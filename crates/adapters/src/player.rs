//! Interruptible audio playback via `cpal`, exposed as an [`AudioPlayer`].
//!
//! Synthesized speech (24 kHz mono f32 from the TTS port) is resampled to the output
//! device rate on the **enqueue** side (never in the real-time callback), pushed into
//! a lock-free [`rtrb`] ring, and drained by the cpal output callback. A barge-in
//! calls [`AudioPlayer::barge_stop`], which sets a flush flag the callback honors on
//! its very next tick (≈ one buffer period, < ~15 ms): it drains the ring and emits
//! silence — an instant audible cut. The cpal `Stream` is `!Send` on CoreAudio, so it
//! lives behind [`SendStream`]; it is only ever built/used/dropped on the one thread
//! that owns this player.
//!
//! Counters (shared atomics):
//! - `enqueued` — real samples pushed into the ring (enqueue side).
//! - `consumed` — samples removed from the ring (played or flush-drained).
//! - `delivered` — real samples actually written to the device (excludes underrun
//!   silence and flush-discarded), i.e. "how much of the agent did the user hear".
//!
//! `is_draining` = `enqueued > consumed`. `played_samples` = `delivered`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use hearsay_core::error::PlaybackError;
use hearsay_core::ports::AudioPlayer;
use rtrb::{Consumer, Producer, RingBuffer};

/// The synthesizer native rate (Kokoro, Mimi codec, outputs 24 kHz).
pub const SYNTH_RATE: u32 = 24_000;

/// Wrapper asserting `Send` for the `!Send` CoreAudio stream. Sound because the
/// stream is only ever touched on the thread that owns this player.
struct SendStream(#[allow(dead_code)] cpal::Stream);
unsafe impl Send for SendStream {}

struct Shared {
    flush: AtomicBool,
    /// When false (after a barge-in), `enqueue` drops samples so a late TTS chunk that
    /// races past `barge_stop` can neither be heard nor latch `is_draining` true.
    /// `resume` re-arms it at the start of the next turn.
    live: AtomicBool,
    enqueued: AtomicU64,
    consumed: AtomicU64,
    delivered: AtomicU64,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            flush: AtomicBool::new(false),
            live: AtomicBool::new(true),
            enqueued: AtomicU64::new(0),
            consumed: AtomicU64::new(0),
            delivered: AtomicU64::new(0),
        }
    }
}

/// The cpal-callback side of the ring: pops samples into the device buffer, honoring
/// the flush flag. Pure enough to unit-test without a device (see tests).
struct PlaybackConsumer {
    cons: Consumer<f32>,
    shared: Arc<Shared>,
    channels: usize,
}

impl PlaybackConsumer {
    /// Fill one device buffer (interleaved by `channels`). Allocation-free, lock-free.
    fn fill<T>(&mut self, out: &mut [T])
    where
        T: SizedSample + FromSample<f32>,
    {
        let silence = T::from_sample(0.0f32);

        if self.shared.flush.swap(false, Ordering::AcqRel) {
            // Barge-in: discard everything pending and emit silence.
            let mut drained = 0u64;
            while self.cons.pop().is_ok() {
                drained += 1;
            }
            self.shared.consumed.fetch_add(drained, Ordering::Relaxed);
            out.fill(silence);
            return;
        }

        let ch = self.channels.max(1);
        let mut consumed = 0u64;
        let mut delivered = 0u64;
        for frame in out.chunks_mut(ch) {
            match self.cons.pop() {
                Ok(v) => {
                    let s = T::from_sample(v);
                    for slot in frame.iter_mut() {
                        *slot = s;
                    }
                    consumed += 1;
                    delivered += 1;
                }
                Err(_) => {
                    // Underrun — emit silence, do not count as delivered/heard.
                    for slot in frame.iter_mut() {
                        *slot = silence;
                    }
                }
            }
        }
        self.shared.consumed.fetch_add(consumed, Ordering::Relaxed);
        self.shared.delivered.fetch_add(delivered, Ordering::Relaxed);
    }
}

/// A stateful linear resampler for **upsampling** speech (24 kHz → device rate, e.g.
/// 48 kHz or 44.1 kHz). The TTS output is already band-limited below 12 kHz, so linear
/// interpolation is artifact-free for this use and far cheaper than a sinc kernel.
/// Chunk-continuous: carries the last input sample + fractional phase across calls.
struct Resampler {
    in_rate: u32,
    out_rate: u32,
    step: f64, // input samples advanced per output sample = in/out
    frac: f64, // phase within the current [prev, cur] interval, in input-sample units
    prev: f32,
    started: bool,
}

impl Resampler {
    fn new(in_rate: u32, out_rate: u32) -> Self {
        let in_rate = in_rate.max(1);
        let out_rate = out_rate.max(1);
        Self {
            in_rate,
            out_rate,
            step: in_rate as f64 / out_rate as f64,
            frac: 0.0,
            prev: 0.0,
            started: false,
        }
    }

    fn reset(&mut self) {
        self.frac = 0.0;
        self.prev = 0.0;
        self.started = false;
    }

    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.in_rate == self.out_rate {
            return input.to_vec();
        }
        // Rough output-size estimate to size the Vec once.
        let mut out = Vec::with_capacity(input.len() * self.out_rate as usize / self.in_rate as usize + 2);
        for &cur in input {
            if !self.started {
                self.prev = cur;
                self.started = true;
            }
            while self.frac < 1.0 {
                let s = self.prev + (cur - self.prev) * self.frac as f32;
                out.push(s);
                self.frac += self.step;
            }
            self.frac -= 1.0;
            self.prev = cur;
        }
        out
    }
}

/// The `AudioPlayer` adapter: owns the producer, resampler, and the live cpal stream.
pub struct CpalPlayer {
    prod: Producer<f32>,
    shared: Arc<Shared>,
    resampler: Resampler,
    _stream: SendStream,
}

impl CpalPlayer {
    /// Open the default output device and start a playing (silent) stream. `input_rate`
    /// is the rate of the PCM that will be `enqueue`d (the synthesizer's rate).
    pub fn open(input_rate: u32) -> Result<Self, PlaybackError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| PlaybackError("no default output device".into()))?;
        let supported = device
            .default_output_config()
            .map_err(|e| PlaybackError(e.to_string()))?;
        let device_rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        let sample_format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();

        // Generous ring: ~30 s of device-rate mono so a fast synthesizer never overruns.
        let capacity = (device_rate as usize * 30).max(48_000);
        let (prod, cons) = RingBuffer::<f32>::new(capacity);
        let shared = Arc::new(Shared::default());

        let consumer = PlaybackConsumer {
            cons,
            shared: shared.clone(),
            channels,
        };
        let stream = build_output_stream(&device, &config, sample_format, consumer)?;
        stream.play().map_err(|e| PlaybackError(e.to_string()))?;

        log::info!("playback started: {device_rate} Hz, {channels} ch (synth {input_rate} Hz)");
        Ok(Self {
            prod,
            shared,
            resampler: Resampler::new(input_rate, device_rate),
            _stream: SendStream(stream),
        })
    }
}

impl AudioPlayer for CpalPlayer {
    fn enqueue(&mut self, pcm: &[f32]) -> Result<(), PlaybackError> {
        // Dropped after a barge-in until the next turn resumes us — closes the race
        // where a late TTS chunk lands just after barge_stop.
        if !self.shared.live.load(Ordering::Acquire) {
            return Ok(());
        }
        let resampled = self.resampler.process(pcm);
        let mut pushed = 0u64;
        for &s in &resampled {
            if self.prod.push(s).is_ok() {
                pushed += 1;
            } else {
                break; // ring full (reply longer than ~30 s) — drop the tail
            }
        }
        self.shared.enqueued.fetch_add(pushed, Ordering::Relaxed);
        if (pushed as usize) < resampled.len() {
            log::warn!(
                "playback ring full; dropped {} samples",
                resampled.len() - pushed as usize
            );
        }
        Ok(())
    }

    fn barge_stop(&mut self) {
        // Stop accepting audio, callback drains + silences on its next tick, reset phase.
        self.shared.live.store(false, Ordering::Release);
        self.shared.flush.store(true, Ordering::Release);
        self.resampler.reset();
    }

    fn resume(&mut self) {
        self.shared.live.store(true, Ordering::Release);
    }

    fn played_samples(&self) -> u64 {
        self.shared.delivered.load(Ordering::Relaxed)
    }

    fn is_draining(&self) -> bool {
        self.shared.enqueued.load(Ordering::Relaxed) > self.shared.consumed.load(Ordering::Relaxed)
    }

    fn input_sample_rate(&self) -> u32 {
        self.resampler.in_rate
    }
}

fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    format: cpal::SampleFormat,
    consumer: PlaybackConsumer,
) -> Result<cpal::Stream, PlaybackError> {
    use cpal::SampleFormat as F;
    match format {
        F::F32 => make_output::<f32>(device, config, consumer),
        F::I16 => make_output::<i16>(device, config, consumer),
        F::U16 => make_output::<u16>(device, config, consumer),
        F::I32 => make_output::<i32>(device, config, consumer),
        F::F64 => make_output::<f64>(device, config, consumer),
        other => Err(PlaybackError(format!("unsupported output format {other:?}"))),
    }
}

fn make_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut consumer: PlaybackConsumer,
) -> Result<cpal::Stream, PlaybackError>
where
    T: SizedSample + FromSample<f32>,
{
    let err_fn = |e| log::warn!("playback stream error: {e}");
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| consumer.fill(data),
            err_fn,
            None,
        )
        .map_err(|e| PlaybackError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(cap: usize, channels: usize) -> (Producer<f32>, PlaybackConsumer) {
        let (prod, cons) = RingBuffer::<f32>::new(cap);
        let shared = Arc::new(Shared::default());
        (
            prod,
            PlaybackConsumer {
                cons,
                shared,
                channels,
            },
        )
    }

    #[test]
    fn drains_in_order_and_counts_delivered() {
        let (mut prod, mut cons) = ring(8, 1);
        for v in [0.1, 0.2, 0.3] {
            prod.push(v).unwrap();
        }
        let mut out = [0.0f32; 2];
        cons.fill(&mut out);
        assert_eq!(out, [0.1, 0.2]);
        let mut out2 = [9.0f32; 2];
        cons.fill(&mut out2);
        assert_eq!(out2, [0.3, 0.0]); // underrun -> silence pad
        // delivered counts only the 3 real samples, not the silence pad
        assert_eq!(cons.shared.delivered.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn barge_stop_flush_discards_pending() {
        let (mut prod, mut cons) = ring(16, 1);
        for v in [1.0, 1.0, 1.0, 1.0] {
            prod.push(v).unwrap();
        }
        cons.shared.flush.store(true, Ordering::Release);
        let mut out = [7.0f32; 2];
        cons.fill(&mut out);
        assert_eq!(out, [0.0, 0.0]); // silence
        // nothing delivered (heard); everything drained from the ring
        assert_eq!(cons.shared.delivered.load(Ordering::Relaxed), 0);
        assert_eq!(cons.shared.consumed.load(Ordering::Relaxed), 4);
        // ring now empty: a subsequent fill is pure silence
        let mut out2 = [3.0f32; 2];
        cons.fill(&mut out2);
        assert_eq!(out2, [0.0, 0.0]);
    }

    #[test]
    fn fill_duplicates_mono_across_channels() {
        let (mut prod, mut cons) = ring(8, 2);
        prod.push(0.5).unwrap();
        let mut out = [0.0f32; 4]; // 2 frames x 2 channels
        cons.fill(&mut out);
        assert_eq!(out, [0.5, 0.5, 0.0, 0.0]); // frame 1 = 0.5 both channels, frame 2 underrun
        assert_eq!(cons.shared.delivered.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn resampler_passthrough_when_rates_match() {
        let mut r = Resampler::new(24_000, 24_000);
        let input = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(r.process(&input), input);
    }

    #[test]
    fn resampler_doubles_length_for_2x_upsample() {
        let mut r = Resampler::new(24_000, 48_000);
        // a long ramp; 2x upsample yields ~2x samples
        let input: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let out = r.process(&input);
        assert!(
            (195..=205).contains(&out.len()),
            "expected ~200 samples, got {}",
            out.len()
        );
        // interpolated midpoints lie between neighbors (monotonic ramp stays monotonic)
        assert!(out.windows(2).all(|w| w[1] >= w[0] - 1e-6));
    }

    #[test]
    fn resampler_is_chunk_continuous() {
        let input: Vec<f32> = (0..200).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut whole = Resampler::new(24_000, 44_100);
        let out_whole = whole.process(&input);
        let mut split = Resampler::new(24_000, 44_100);
        let mut out_split = split.process(&input[..73]);
        out_split.extend(split.process(&input[73..]));
        assert_eq!(out_whole.len(), out_split.len());
        let max_diff = out_whole
            .iter()
            .zip(&out_split)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "chunk variance: {max_diff}");
    }

    /// Regression for the barge-in race (no-mercy B1/B2): after `barge_stop`, a late
    /// `enqueue` must be dropped (not played, not counted) until `resume`.
    #[test]
    #[ignore = "requires an audio output device; run with --ignored"]
    fn enqueue_dropped_after_barge_until_resume() {
        let mut p = CpalPlayer::open(SYNTH_RATE).expect("open output");
        let chunk = vec![0.05f32; 2_400];
        p.enqueue(&chunk).unwrap();
        let baseline = p.shared.enqueued.load(Ordering::Relaxed);
        assert!(baseline > 0, "first enqueue should count");

        p.barge_stop();
        p.enqueue(&chunk).unwrap(); // racing chunk after barge -> must be dropped
        assert_eq!(
            p.shared.enqueued.load(Ordering::Relaxed),
            baseline,
            "enqueue after barge_stop must be dropped"
        );

        p.resume();
        p.enqueue(&chunk).unwrap(); // next turn -> accepted again
        assert!(
            p.shared.enqueued.load(Ordering::Relaxed) > baseline,
            "enqueue must work again after resume"
        );
    }

    /// Manual smoke test: play a 440 Hz tone for ~300 ms on the real device.
    #[test]
    #[ignore = "requires an audio output device; run with --ignored --nocapture"]
    fn plays_a_tone() {
        let mut p = CpalPlayer::open(SYNTH_RATE).expect("open output");
        let tone: Vec<f32> = (0..SYNTH_RATE / 3)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SYNTH_RATE as f32).sin() * 0.2)
            .collect();
        p.enqueue(&tone).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(500));
        assert!(p.played_samples() > 0, "no samples played");
    }
}

//! Hardware **acoustic echo cancellation** via the macOS VoiceProcessingIO AudioUnit
//! (AUVoiceIO) — the same duplex unit FaceTime uses. One unit owns both the mic and the
//! speaker, so the agent's own TTS (played through this unit's render callback) is the
//! echo reference the unit cancels from the mic. That lets you barge-in **open-air**
//! without headphones.
//!
//! [`open`] returns a [`VpioSource`] (`AudioSource`, echo-cancelled mic resampled to 16
//! kHz for whisper) and a [`VpioPlayer`] (`AudioPlayer`, 24 kHz TTS resampled to the
//! unit's hardware rate) that share the single unit. The player keeps the same
//! `live`/`flush` barge-in gating as the cpal player so a late TTS chunk after a
//! barge-in is dropped. The unit runs at the hardware (communications-device) rate; the
//! callbacks are mono, non-interleaved f32 (what VPIO accepts — anything else fails
//! `AudioUnitInitialize` with -10851).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use coreaudio::audio_unit::audio_format::LinearPcmFlags;
use coreaudio::audio_unit::render_callback::{self, data};
use coreaudio::audio_unit::{AudioUnit, Element, IOType, SampleFormat, Scope, StreamFormat};
use hearsay_core::error::{CaptureError, PlaybackError};
use hearsay_core::ports::{AudioPlayer, AudioSource};
use objc2_audio_toolbox::{kAudioOutputUnitProperty_EnableIO, kAudioUnitProperty_StreamFormat};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::player::Resampler;
use crate::resample::Resampler16k;

/// The synthesizer's native rate (Kokoro/Mimi → 24 kHz), the rate `enqueue`d PCM is in.
const SYNTH_RATE: u32 = 24_000;

/// Keeps the `!Send` CoreAudio unit alive; touched only via its owning `Arc<Mutex<…>>`.
struct SendUnit(#[allow(dead_code)] AudioUnit);
unsafe impl Send for SendUnit {}

/// Playback counters/flags shared between [`VpioPlayer`] (control thread) and the render
/// callback (CoreAudio audio thread). Same semantics as the cpal player's `Shared`.
struct PlayShared {
    flush: AtomicBool,
    live: AtomicBool,
    enqueued: AtomicU64,
    consumed: AtomicU64,
    delivered: AtomicU64,
}

impl Default for PlayShared {
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

type SharedUnit = Arc<Mutex<Option<SendUnit>>>;

fn be(e: coreaudio::Error) -> CaptureError {
    CaptureError::Backend(e.to_string())
}

/// Open the VoiceProcessingIO unit; returns the AEC mic source + the speaker player that
/// share it. Initializing the unit triggers the microphone TCC prompt if not yet granted.
pub fn open() -> Result<(VpioSource, VpioPlayer), CaptureError> {
    let mut au = AudioUnit::new_uninitialized(IOType::VoiceProcessingIO)
        .map_err(|e| CaptureError::Backend(format!("VoiceProcessingIO create: {e}")))?;

    // Enable mic (bus 1, input scope) + speaker (bus 0, output scope).
    let enable: u32 = 1;
    au.set_property(kAudioOutputUnitProperty_EnableIO, Scope::Input, Element::Input, Some(&enable))
        .map_err(be)?;
    au.set_property(kAudioOutputUnitProperty_EnableIO, Scope::Output, Element::Output, Some(&enable))
        .map_err(be)?;

    // VPIO runs at the hardware (comms-device) rate; match it (mono f32, non-interleaved).
    let hw_rate = au
        .stream_format(Scope::Output, Element::Output)
        .map(|f| f.sample_rate)
        .unwrap_or(48_000.0);
    let hw = hw_rate as u32;
    let fmt = StreamFormat {
        sample_rate: hw_rate,
        sample_format: SampleFormat::F32,
        flags: LinearPcmFlags::IS_FLOAT | LinearPcmFlags::IS_PACKED | LinearPcmFlags::IS_NON_INTERLEAVED,
        channels: 1,
    }
    .to_asbd();
    au.set_property(kAudioUnitProperty_StreamFormat, Scope::Output, Element::Input, Some(&fmt))
        .map_err(be)?; // mic (near-end)
    au.set_property(kAudioUnitProperty_StreamFormat, Scope::Input, Element::Output, Some(&fmt))
        .map_err(be)?; // speaker (far-end / AEC reference)

    // ~10 s of mic headroom; ~30 s of TTS playback headroom.
    let (mut mic_prod, mic_cons) = RingBuffer::<f32>::new(hw as usize * 10);
    let (tts_prod, mut tts_cons) = RingBuffer::<f32>::new(hw as usize * 30);
    let shared = Arc::new(PlayShared::default());

    type Args = render_callback::Args<data::NonInterleaved<f32>>;

    // Input callback: coreaudio-rs has already AudioUnitRender'd the echo-cancelled mic.
    au.set_input_callback(move |args: Args| {
        let Args { num_frames, mut data, .. } = args;
        if let Some(ch) = data.channels_mut().next() {
            for &s in ch.iter().take(num_frames) {
                let _ = mic_prod.push(s); // drop on overrun — RT-safe
            }
        }
        Ok(())
    })
    .map_err(be)?;

    // Render callback: drain the TTS ring into the speaker, honoring the barge-in flush.
    {
        let shared = shared.clone();
        au.set_render_callback(move |args: Args| {
            let Args { num_frames, mut data, .. } = args;
            let Some(ch) = data.channels_mut().next() else {
                return Ok(());
            };
            if shared.flush.swap(false, Ordering::AcqRel) {
                let mut drained = 0u64;
                while tts_cons.pop().is_ok() {
                    drained += 1;
                }
                shared.consumed.fetch_add(drained, Ordering::Relaxed);
                for s in ch.iter_mut().take(num_frames) {
                    *s = 0.0;
                }
                return Ok(());
            }
            let mut consumed = 0u64;
            let mut delivered = 0u64;
            for s in ch.iter_mut().take(num_frames) {
                match tts_cons.pop() {
                    Ok(v) => {
                        *s = v;
                        consumed += 1;
                        delivered += 1;
                    }
                    Err(_) => *s = 0.0, // underrun -> silence
                }
            }
            shared.consumed.fetch_add(consumed, Ordering::Relaxed);
            shared.delivered.fetch_add(delivered, Ordering::Relaxed);
            Ok(())
        })
        .map_err(be)?;
    }

    au.initialize()
        .map_err(|e| CaptureError::Backend(format!("VoiceProcessingIO init (microphone permission?): {e}")))?;
    au.start().map_err(be)?;
    log::info!("VoiceProcessingIO (hardware AEC) started @ {hw} Hz");

    let unit: SharedUnit = Arc::new(Mutex::new(Some(SendUnit(au))));
    Ok((
        VpioSource {
            cons: mic_cons,
            resampler: Resampler16k::new(hw),
            unit: unit.clone(),
            label: "VoiceProcessingIO mic (AEC)".to_string(),
        },
        VpioPlayer {
            prod: tts_prod,
            shared,
            resampler: Resampler::new(SYNTH_RATE, hw),
            _unit: unit,
        },
    ))
}

/// The mic half of the shared VPIO unit: echo-cancelled audio, resampled to 16 kHz.
pub struct VpioSource {
    cons: Consumer<f32>,
    resampler: Resampler16k,
    unit: SharedUnit,
    label: String,
}

impl AudioSource for VpioSource {
    fn start(&mut self) -> Result<(), CaptureError> {
        Ok(()) // the unit is already running (started in `open`)
    }

    fn poll(&mut self) -> Result<Vec<f32>, CaptureError> {
        let mut hw = Vec::with_capacity(self.cons.slots());
        while let Ok(s) = self.cons.pop() {
            hw.push(s);
        }
        Ok(self.resampler.process(&hw)) // hw rate -> 16 kHz
    }

    fn stop(&mut self) {
        if let Ok(mut g) = self.unit.lock() {
            g.take(); // drop -> AudioUnit uninitialize + dispose (stops the duplex unit)
        }
    }

    fn label(&self) -> &str {
        &self.label
    }
}

/// The speaker half of the shared VPIO unit: 24 kHz TTS resampled to the hardware rate,
/// played through the unit so it becomes the echo-cancellation reference.
pub struct VpioPlayer {
    prod: Producer<f32>,
    shared: Arc<PlayShared>,
    resampler: Resampler,
    _unit: SharedUnit,
}

impl AudioPlayer for VpioPlayer {
    fn enqueue(&mut self, pcm: &[f32]) -> Result<(), PlaybackError> {
        if !self.shared.live.load(Ordering::Acquire) {
            return Ok(()); // dropped after a barge-in until the next turn resumes us
        }
        let out = self.resampler.process(pcm);
        let mut pushed = 0u64;
        for &s in &out {
            if self.prod.push(s).is_ok() {
                pushed += 1;
            } else {
                break;
            }
        }
        self.shared.enqueued.fetch_add(pushed, Ordering::Relaxed);
        Ok(())
    }

    fn barge_stop(&mut self) {
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
        SYNTH_RATE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Manual: open the real VPIO unit, enqueue a tone, confirm the render callback
    /// consumes it (played_samples advances) and the mic delivers echo-cancelled frames.
    #[test]
    #[ignore = "requires audio devices + mic permission; run with --ignored --nocapture"]
    fn duplex_runs() {
        let (mut src, mut player) = open().expect("open VPIO");
        let tone: Vec<f32> = (0..SYNTH_RATE)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SYNTH_RATE as f32).sin() * 0.05)
            .collect();
        player.enqueue(&tone).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(800));
        let mic = src.poll().unwrap();
        println!(
            "played={} mic_samples_16k={}",
            player.played_samples(),
            mic.len()
        );
        assert!(player.played_samples() > 0, "render callback never consumed TTS");
        assert!(!mic.is_empty(), "no mic frames captured");
        src.stop();
    }
}

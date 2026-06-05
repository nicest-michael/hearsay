//! A streaming, anti-aliased sample-rate converter to Hearsay's 16 kHz mono
//! canonical rate.
//!
//! Implemented as a **normalized Blackman-windowed-sinc** resampler (the same core
//! technique as libsamplerate / SoX): each output sample is a windowed-sinc-weighted
//! sum of nearby input samples, with the sinc cutoff placed at the *output* Nyquist
//! when downsampling so high frequencies can't alias into the 0–8 kHz band Whisper
//! cares about. Weights are normalized per output sample, so DC gain is exactly 1
//! and partial windows at the stream start degrade gracefully.
//!
//! It is stateful and chunk-invariant: feeding audio in one call or many produces
//! identical output, because positions are tracked in an absolute input-sample
//! timeline and only no-longer-needed history is dropped.

use hearsay_core::audio::TARGET_RATE;
use std::f64::consts::PI;

/// Resamples an arbitrary input rate to 16 kHz mono. Construct once per source and
/// feed it consecutive chunks with [`Resampler16k::process`].
pub struct Resampler16k {
    in_rate: u32,
    /// Input samples consumed per output sample (`in_rate / 16000`).
    ratio: f64,
    /// Low-pass cutoff in cycles per input sample.
    fc: f64,
    /// Kernel half-width in input samples.
    half: i64,
    /// Sliding window of input history.
    buf: Vec<f32>,
    /// Absolute input index of `buf[0]`.
    base: i64,
    /// Absolute input-coordinate of the next output sample.
    next_out: f64,
}

impl Resampler16k {
    pub fn new(in_rate: u32) -> Self {
        let in_rate = in_rate.max(1);
        let ratio = in_rate as f64 / TARGET_RATE as f64;
        // 8 kHz output-Nyquist expressed in cycles/input-sample, capped at the input
        // Nyquist (0.5) so upsampling doesn't try to synthesize above input bandwidth.
        let fc = (0.5 / ratio).min(0.5);
        // More taps when downsampling hard keeps the transition band tight.
        let half = (16.0 * ratio.max(1.0)).ceil() as i64;
        Self {
            in_rate,
            ratio,
            fc,
            half,
            buf: Vec::new(),
            base: 0,
            next_out: 0.0,
        }
    }

    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    /// Resample a chunk of mono f32 input, returning the 16 kHz output produced so
    /// far. May return empty if more input is needed to fill a kernel window.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.in_rate == TARGET_RATE {
            return input.to_vec(); // exact-rate fast path
        }
        self.buf.extend_from_slice(input);
        let last_abs = self.base + self.buf.len() as i64 - 1;

        let mut out = Vec::new();
        // Produce while the kernel's right edge is covered by available samples.
        while self.next_out.floor() as i64 + self.half <= last_abs {
            out.push(self.sample_at(self.next_out));
            self.next_out += self.ratio;
        }

        // Drop history no future output will read.
        let keep_from = (self.next_out.floor() as i64 - self.half).max(self.base);
        let drop = (keep_from - self.base) as usize;
        if drop > 0 && drop <= self.buf.len() {
            self.buf.drain(..drop);
            self.base += drop as i64;
        }
        out
    }

    fn sample_at(&self, pos: f64) -> f32 {
        let center = pos.floor() as i64;
        let mut acc = 0.0f64;
        let mut norm = 0.0f64;
        for k in -self.half..=self.half {
            let idx = center + k;
            let w = self.kernel(idx as f64 - pos);
            if w != 0.0 {
                acc += w * self.sample(idx) as f64;
                norm += w;
            }
        }
        if norm.abs() > 1e-12 {
            (acc / norm) as f32
        } else {
            0.0
        }
    }

    fn sample(&self, abs_idx: i64) -> f32 {
        if abs_idx < self.base {
            return 0.0; // before the stream began (start warm-up only)
        }
        self.buf
            .get((abs_idx - self.base) as usize)
            .copied()
            .unwrap_or(0.0)
    }

    /// Blackman-windowed sinc low-pass at `fc`.
    fn kernel(&self, x: f64) -> f64 {
        let t = x / self.half as f64;
        if t <= -1.0 || t >= 1.0 {
            return 0.0;
        }
        let window = 0.42 + 0.5 * (PI * t).cos() + 0.08 * (2.0 * PI * t).cos();
        2.0 * self.fc * sinc(2.0 * self.fc * x) * window
    }
}

fn sinc(y: f64) -> f64 {
    if y.abs() < 1e-12 {
        1.0
    } else {
        let p = PI * y;
        p.sin() / p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hearsay_core::audio::rms;

    fn sine(freq: f64, rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / rate as f64).sin() as f32)
            .collect()
    }

    #[test]
    fn downsamples_to_expected_length() {
        let mut r = Resampler16k::new(48_000);
        let out = r.process(&sine(1_000.0, 48_000, 48_000)); // 1 s
                                                             // ~16000 output samples (minus a few for warm-up)
        assert!(
            (15_800..=16_001).contains(&out.len()),
            "len = {}",
            out.len()
        );
    }

    #[test]
    fn passband_tone_is_preserved() {
        let mut r = Resampler16k::new(48_000);
        let input = sine(1_000.0, 48_000, 48_000);
        let out = r.process(&input);
        // a 1 kHz tone is well within the 8 kHz passband; amplitude (RMS ~0.707) survives
        assert!(rms(&out) > 0.6, "passband RMS too low: {}", rms(&out));
    }

    #[test]
    fn out_of_band_tone_is_attenuated() {
        // 12 kHz at 48k is above the 8 kHz output Nyquist -> must be filtered out,
        // not aliased into the audible band.
        let mut r = Resampler16k::new(48_000);
        let out = r.process(&sine(12_000.0, 48_000, 48_000));
        assert!(
            rms(&out) < 0.15,
            "aliasing! out-of-band RMS too high: {}",
            rms(&out)
        );
    }

    #[test]
    fn exact_rate_is_passthrough() {
        let mut r = Resampler16k::new(16_000);
        let input = sine(440.0, 16_000, 1_600);
        assert_eq!(r.process(&input), input);
    }

    #[test]
    fn chunked_input_matches_single_call() {
        let input = sine(1_000.0, 48_000, 24_000);
        let mut whole = Resampler16k::new(48_000);
        let out_whole = whole.process(&input);

        let mut split = Resampler16k::new(48_000);
        let mut out_split = split.process(&input[..10_000]);
        out_split.extend(split.process(&input[10_000..]));

        assert_eq!(out_whole.len(), out_split.len());
        let max_diff = out_whole
            .iter()
            .zip(&out_split)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "chunk-variance: {max_diff}");
    }

    #[test]
    fn handles_44100() {
        let mut r = Resampler16k::new(44_100);
        let out = r.process(&sine(1_000.0, 44_100, 44_100)); // 1 s -> ~16000 samples
        assert!((out.len() as i64 - 16_000).abs() < 100);
        assert!(rms(&out) > 0.6);
    }
}

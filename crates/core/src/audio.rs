//! Audio sample-rate / channel math. Pure functions, no I/O.
//!
//! Hearsay's canonical internal format is **16 kHz mono f32** in `[-1.0, 1.0]` —
//! the format Whisper expects. Adapters resample device audio to this before it
//! reaches the domain.

/// The sample rate Whisper consumes and the domain reasons in.
pub const TARGET_RATE: u32 = 16_000;

/// Number of 16 kHz samples in `ms` milliseconds.
#[inline]
pub fn samples_for_ms(ms: u32) -> usize {
    (TARGET_RATE as u64 * ms as u64 / 1000) as usize
}

/// Number of 16 kHz samples in `cs` centiseconds (hundredths of a second).
#[inline]
pub fn samples_for_cs(cs: u32) -> usize {
    (TARGET_RATE as u64 * cs as u64 / 100) as usize
}

/// Centiseconds represented by `n` samples at 16 kHz (rounds down).
#[inline]
pub fn cs_for_samples(n: usize) -> u32 {
    (n as u64 * 100 / TARGET_RATE as u64) as u32
}

/// Down-mix interleaved multi-channel f32 to mono by averaging the channels.
///
/// `channels` must be >= 1. A single channel is returned as-is.
pub fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    assert!(channels >= 1, "channels must be >= 1");
    if channels == 1 {
        return interleaved.to_vec();
    }
    let inv = 1.0 / channels as f32;
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() * inv)
        .collect()
}

/// Root-mean-square level of a mono buffer in `[0.0, ~1.0]`.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Sum two sample-aligned mono streams, clamping to `[-1.0, 1.0]`.
/// The shorter stream is treated as zero-padded. Kept as a domain utility;
/// Hearsay transcribes mic and system as *separate* sessions rather than mixing
/// (avoids double-transcription when speakers feed the mic).
pub fn mix(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let x = a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0);
            x.clamp(-1.0, 1.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_and_cs_round_trip() {
        assert_eq!(samples_for_ms(1000), 16_000);
        assert_eq!(samples_for_ms(30), 480);
        assert_eq!(samples_for_cs(100), 16_000);
        assert_eq!(samples_for_cs(50), 8_000);
        assert_eq!(cs_for_samples(16_000), 100);
        assert_eq!(cs_for_samples(8_000), 50);
        assert_eq!(cs_for_samples(0), 0);
    }

    #[test]
    fn mono_passthrough() {
        let m = vec![0.1, -0.2, 0.3];
        assert_eq!(downmix_to_mono(&m, 1), m);
    }

    #[test]
    fn stereo_downmix_averages_channels() {
        // L,R interleaved: (1.0,0.0), (0.5,-0.5)
        let stereo = vec![1.0, 0.0, 0.5, -0.5];
        let mono = downmix_to_mono(&stereo, 2);
        assert_eq!(mono, vec![0.5, 0.0]);
    }

    #[test]
    fn downmix_ignores_trailing_partial_frame() {
        // 5 samples, 2 channels -> 2 full frames, last sample dropped
        let stereo = vec![1.0, 1.0, 0.0, 0.0, 0.7];
        assert_eq!(downmix_to_mono(&stereo, 2), vec![1.0, 0.0]);
    }

    #[test]
    fn rms_of_silence_is_zero() {
        assert_eq!(rms(&[0.0; 100]), 0.0);
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn rms_of_full_scale_is_one() {
        assert!((rms(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mix_sums_and_clamps() {
        assert_eq!(mix(&[0.5, 0.5], &[0.25, 0.75]), vec![0.75, 1.0]);
        // clamps below -1.0
        assert_eq!(mix(&[-0.8], &[-0.8]), vec![-1.0]);
    }

    #[test]
    fn mix_zero_pads_shorter() {
        assert_eq!(mix(&[0.1, 0.2, 0.3], &[0.1]), vec![0.2, 0.2, 0.3]);
    }
}

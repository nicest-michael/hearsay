//! An [`AudioSource`] that replays a WAV file as if it were the microphone — for
//! headless end-to-end testing of the conversation pipeline without a human speaking.
//! Converts to 16 kHz mono f32, appends trailing silence so the VAD declares the
//! utterance ended, and streams it in ~100 ms chunks per `poll`.

use hearsay_core::audio::TARGET_RATE;
use hearsay_core::error::CaptureError;
use hearsay_core::ports::AudioSource;

use crate::resample::Resampler16k;

pub struct FileAudioSource {
    samples: Vec<f32>,
    pos: usize,
    chunk: usize,
    label: String,
}

impl FileAudioSource {
    /// Load `path`, append `trailing_silence_ms` of silence, ready to replay.
    pub fn from_wav(path: &str, trailing_silence_ms: u32) -> Result<Self, CaptureError> {
        let mut reader =
            hound::WavReader::open(path).map_err(|e| CaptureError::Backend(e.to_string()))?;
        let spec = reader.spec();
        let channels = spec.channels.max(1) as usize;

        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| s.unwrap_or(0.0))
                .collect(),
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.unwrap_or(0) as f32 / max)
                    .collect()
            }
        };

        // Down-mix to mono.
        let mono: Vec<f32> = if channels <= 1 {
            interleaved
        } else {
            interleaved
                .chunks(channels)
                .map(|f| f.iter().sum::<f32>() / channels as f32)
                .collect()
        };

        // Resample to 16 kHz if needed.
        let mut samples = if spec.sample_rate == TARGET_RATE {
            mono
        } else {
            Resampler16k::new(spec.sample_rate).process(&mono)
        };

        let silence = (TARGET_RATE as u64 * trailing_silence_ms as u64 / 1000) as usize;
        samples.extend(std::iter::repeat(0.0).take(silence));

        Ok(Self {
            samples,
            pos: 0,
            chunk: (TARGET_RATE as usize) / 10, // 100 ms
            label: format!("File: {path}"),
        })
    }

    /// True once the whole clip (incl. trailing silence) has been handed out.
    pub fn exhausted(&self) -> bool {
        self.pos >= self.samples.len()
    }
}

impl AudioSource for FileAudioSource {
    fn start(&mut self) -> Result<(), CaptureError> {
        Ok(())
    }

    fn poll(&mut self) -> Result<Vec<f32>, CaptureError> {
        if self.pos >= self.samples.len() {
            return Ok(Vec::new());
        }
        let end = (self.pos + self.chunk).min(self.samples.len());
        let out = self.samples[self.pos..end].to_vec();
        self.pos = end;
        Ok(out)
    }

    fn stop(&mut self) {}

    fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replays_then_exhausts_with_trailing_silence() {
        // Write a tiny 16 kHz mono i16 WAV (0.1 s tone) to a temp file.
        let path = std::env::temp_dir().join("hearsay_file_src_test.wav");
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::create(&path, spec).unwrap();
            for i in 0..1_600 {
                let v = ((i as f32 * 0.2).sin() * 10_000.0) as i16;
                w.write_sample(v).unwrap();
            }
            w.finalize().unwrap();
        }
        let mut src = FileAudioSource::from_wav(path.to_str().unwrap(), 500).unwrap();
        let mut total = 0;
        while !src.exhausted() {
            total += src.poll().unwrap().len();
        }
        // 1600 samples tone + 8000 samples (500 ms) silence
        assert_eq!(total, 1_600 + 8_000);
        assert!(src.poll().unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }
}

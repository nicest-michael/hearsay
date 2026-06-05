//! Local Whisper transcription via `whisper-rs` (Metal on Apple Silicon).
//!
//! Encodes the council's anti-hallucination decisions (R3): greedy/temperature-0
//! decoding, `no_context` (kills runaway repetition), blank/non-speech-token
//! suppression, and — because `whisper-rs`'s `set_no_speech_thold` is a no-op — we
//! surface `no_speech_probability` and a token-derived `avg_logprob` per segment so
//! the *domain* can gate hallucinated segments out of the committed transcript.

use std::path::Path;
use std::sync::Once;

use hearsay_core::config::ModelChoice;
use hearsay_core::error::TranscribeError;
use hearsay_core::ports::Transcriber;
use hearsay_core::word::{Hypothesis, Segment};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

static LOG_HOOK: Once = Once::new();

pub struct WhisperTranscriber {
    // Kept alive so `state` (which holds an Arc to it) stays valid; both are Send.
    _ctx: WhisperContext,
    state: WhisperState,
    english_only: bool,
    n_threads: i32,
}

impl WhisperTranscriber {
    /// Load a ggml model with GPU (Metal) enabled.
    pub fn load(model_path: &Path, model: ModelChoice) -> Result<Self, TranscribeError> {
        LOG_HOOK.call_once(whisper_rs::install_logging_hooks);

        let mut cparams = WhisperContextParameters::default();
        cparams.use_gpu(true);
        // flash_attn left off: it's an unverifiable perf bet on the vendored
        // whisper.cpp build, and Metal is already real-time at base.en.
        cparams.flash_attn(false);

        let ctx = WhisperContext::new_with_params(model_path, cparams)
            .map_err(|e| TranscribeError::ModelLoad(format!("{}: {e}", model_path.display())))?;
        let state = ctx
            .create_state()
            .map_err(|e| TranscribeError::ModelLoad(e.to_string()))?;

        // Leave 2 cores for capture/UI; whisper.cpp is GPU-bound on Metal anyway.
        let n_threads = std::thread::available_parallelism()
            .map(|n| (n.get().saturating_sub(2)).max(2))
            .unwrap_or(4) as i32;

        Ok(Self {
            _ctx: ctx,
            state,
            english_only: model.is_english_only(),
            n_threads,
        })
    }
}

/// Build decoding params from plain values (no `&self` borrow — so it doesn't
/// conflict with the `&mut self.state` borrow `full` needs). Encodes the
/// anti-hallucination decisions (council R3).
fn build_params<'a>(n_threads: i32, english_only: bool) -> FullParams<'a, 'a> {
    let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    p.set_n_threads(n_threads);
    p.set_translate(false);
    if english_only {
        p.set_language(Some("en"));
    }
    p.set_no_context(true);
    p.set_suppress_blank(true);
    p.set_suppress_nst(true);
    p.set_temperature(0.0);
    p.set_entropy_thold(2.4);
    p.set_logprob_thold(-1.0);
    p.set_token_timestamps(true); // for buffer trimming
    p.set_print_special(false);
    p.set_print_progress(false);
    p.set_print_realtime(false);
    p.set_print_timestamps(false);
    p
}

impl Transcriber for WhisperTranscriber {
    fn transcribe(&mut self, audio_16k_mono: &[f32]) -> Result<Hypothesis, TranscribeError> {
        if audio_16k_mono.is_empty() {
            return Ok(Hypothesis::default());
        }
        let params = build_params(self.n_threads, self.english_only);
        self.state
            .full(params, audio_16k_mono)
            .map_err(|e| TranscribeError::Inference(e.to_string()))?;

        let n = self.state.full_n_segments(); // c_int (i32)
        let mut segments = Vec::new();
        for i in 0..n {
            let Some(seg) = self.state.get_segment(i) else {
                continue;
            };
            let text = seg.to_str_lossy().unwrap_or_default().trim().to_string();
            if text.is_empty() {
                continue;
            }
            segments.push(Segment {
                text,
                t0_cs: seg.start_timestamp().max(0) as u32,
                t1_cs: seg.end_timestamp().max(0) as u32,
                no_speech_prob: seg.no_speech_probability(),
                avg_logprob: avg_logprob(&seg),
            });
        }
        Ok(Hypothesis { segments })
    }
}

/// Mean of token log-probabilities in a segment (whisper-rs exposes only linear
/// token probabilities, so we log them here). Returns 0.0 (neutral) if unavailable.
fn avg_logprob(seg: &whisper_rs::WhisperSegment<'_>) -> f32 {
    let n = seg.n_tokens();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut count = 0u32;
    for i in 0..n {
        if let Some(tok) = seg.get_token(i) {
            let p = tok.token_probability().clamp(1e-10, 1.0);
            sum += (p as f64).ln();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: load a real model (path via HEARSAY_MODEL_PATH) and transcribe
    /// 1 s of silence without panicking. Ignored by default (needs a model file).
    #[test]
    #[ignore = "requires HEARSAY_MODEL_PATH to a ggml model"]
    fn loads_and_runs() {
        let path = std::env::var("HEARSAY_MODEL_PATH").expect("set HEARSAY_MODEL_PATH");
        let mut t = WhisperTranscriber::load(Path::new(&path), ModelChoice::BaseEn).unwrap();
        let hyp = t.transcribe(&vec![0.0f32; 16_000]).unwrap();
        // silence should not produce confident words once gated
        let words = hyp.confident_words(0.6, -1.0);
        println!("silence -> {} confident words", words.len());
    }

    /// End-to-end: transcribe a real 16 kHz mono WAV (generate one with
    /// `say -o fox.wav --data-format=LEF32@16000 "the quick brown fox …"`).
    #[test]
    #[ignore = "requires HEARSAY_MODEL_PATH + HEARSAY_TEST_WAV (16k mono)"]
    fn transcribes_real_speech() {
        let model = std::env::var("HEARSAY_MODEL_PATH").expect("HEARSAY_MODEL_PATH");
        let wav = std::env::var("HEARSAY_TEST_WAV").expect("HEARSAY_TEST_WAV");
        let mut reader = hound::WavReader::open(&wav).unwrap();
        let samples: Vec<f32> = match reader.spec().sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect(),
        };
        let mut t = WhisperTranscriber::load(Path::new(&model), ModelChoice::BaseEn).unwrap();
        let hyp = t.transcribe(&samples).unwrap();
        let text = hyp
            .confident_words(0.6, -1.0)
            .iter()
            .map(|w| w.text.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");
        println!("TRANSCRIPT: {text}");
        assert!(
            text.contains("quick") && text.contains("fox"),
            "expected the fox sentence, got: {text}"
        );
    }
}

//! # hearsay-adapters
//!
//! Concrete implementations of the `hearsay-core` ports against real I/O:
//! microphone capture + interruptible playback (cpal), transcription (whisper-rs +
//! Metal), the local LLM over OpenAI-compatible HTTP, the MisoTTS speech synthesizer
//! over a Unix-domain-socket sidecar, plus model management, a windowed-sinc
//! resampler, and child-process (sidecar) lifecycle. This crate is the only place
//! external audio/ML/network crates appear — the dependency rule keeps them out of
//! the domain.

pub mod mic;
pub mod models;
pub mod resample;
pub mod whisper;

pub use mic::{list_input_devices, CpalMicSource, DeviceInfo};
pub use models::{ensure_model, is_present, model_path, ModelError};
pub use resample::Resampler16k;
pub use whisper::WhisperTranscriber;

/// Compile-time proof the STT port impls satisfy the `Send` bound the worker threads
/// require (they move to the aggregator/inference threads).
const _: () = {
    fn assert_send<T: Send>() {}
    let _ = assert_send::<CpalMicSource>;
    let _ = assert_send::<WhisperTranscriber>;
};

//! # hearsay-adapters
//!
//! Concrete implementations of the `hearsay-core` ports against real I/O:
//! microphone capture + interruptible playback (cpal), transcription (whisper-rs +
//! Metal), the local LLM over OpenAI-compatible HTTP, the MisoTTS speech synthesizer
//! over a Unix-domain-socket sidecar, plus model management, a windowed-sinc
//! resampler, and child-process (sidecar) lifecycle. This crate is the only place
//! external audio/ML/network crates appear — the dependency rule keeps them out of
//! the domain.

pub mod llm;
pub mod mic;
pub mod models;
pub mod player;
pub mod resample;
pub mod sidecar;
pub mod tts;
pub mod whisper;

pub use llm::MlxChat;
pub use mic::{list_input_devices, CpalMicSource, DeviceInfo};
pub use models::{ensure_model, is_present, model_path, ModelError};
pub use player::{CpalPlayer, SYNTH_RATE};
pub use resample::Resampler16k;
pub use sidecar::{sweep_stale, Sidecar};
pub use tts::SidecarTts;
pub use whisper::WhisperTranscriber;

/// Compile-time proof the port impls satisfy the `Send` bound the worker/engine
/// threads require (they move across threads).
const _: () = {
    fn assert_send<T: Send>() {}
    let _ = assert_send::<CpalMicSource>;
    let _ = assert_send::<WhisperTranscriber>;
    let _ = assert_send::<CpalPlayer>;
};

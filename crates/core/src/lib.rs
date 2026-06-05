//! # hearsay-core
//!
//! The **pure domain** of Hearsay and the **ports** (traits) the rest of the system
//! depends on. Zero I/O dependencies by design — the hexagon's dependency rule is
//! compiler-enforced: nothing here can reach a real device, the GPU, the network, the
//! filesystem, or the UI. Everything is deterministic and unit-testable.
//!
//! ## Hearing (STT)
//! - [`audio`] — sample-rate / channel math, mono downmix, RMS.
//! - [`vad`] — energy voice-activity detection with an adaptive noise floor (inherited
//!   from earshot); drives utterance onset (barge-in) and end-of-turn detection.
//! - [`word`] — [`word::Word`] / [`word::Segment`] / [`word::Hypothesis`]: a
//!   transcriber's output, with the confidence filter that drops hallucinations.
//!
//! ## Conversing
//! - [`dialogue`] — [`dialogue::Conversation`] history + the low-latency [`dialogue::SentenceChunker`].
//! - [`conversation`] — the pure dialog FSM with turn-id barge-in invalidation.
//!
//! ## Shared
//! - [`config`] — pure configuration value types.
//! - [`error`] — domain error types.
//! - [`ports`] — the traits adapters implement (capture, transcribe, LLM, TTS, playback).

pub mod audio;
pub mod config;
pub mod conversation;
pub mod dialogue;
pub mod error;
pub mod ports;
pub mod vad;
pub mod word;

pub use audio::TARGET_RATE;

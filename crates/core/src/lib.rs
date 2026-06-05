//! # hearsay-core
//!
//! The **pure domain** of Hearsay and the **ports** (traits) the rest of the system
//! depends on. Zero I/O dependencies by design — the hexagon's dependency rule is
//! compiler-enforced: nothing here can reach a real device, the GPU, the network, the
//! filesystem, or the UI. Everything is deterministic and unit-testable.
//!
//! ## The STT half (inherited from earshot)
//! - [`audio`] — sample-rate / channel math, mono downmix, stream mixing.
//! - [`word`] — [`word::Word`] / [`word::Segment`] / [`word::Hypothesis`]: a transcriber's output.
//! - [`transcript`] — [`transcript::TranscriptUpdate`]: stabilized transcript deltas.
//! - [`vad`] — energy voice-activity detection with an adaptive noise floor.
//! - [`stabilize`] — LocalAgreement-2 word stabilization with a no-speech commit gate.
//! - [`aggregator`] — the sliding-window FSM: buffering, windowing, timestamp trimming.
//!
//! ## The conversation half (new)
//! - [`dialogue`] — [`dialogue::Conversation`] history + the low-latency [`dialogue::SentenceChunker`].
//! - [`conversation`] — the pure dialog FSM with turn-id barge-in invalidation.
//!
//! ## Shared
//! - [`config`] — pure configuration value types.
//! - [`error`] — domain error types.
//! - [`ports`] — the traits adapters implement (STT + LLM + TTS + playback).

pub mod aggregator;
pub mod audio;
pub mod config;
pub mod conversation;
pub mod dialogue;
pub mod error;
pub mod ports;
pub mod stabilize;
pub mod transcript;
pub mod vad;
pub mod word;

pub use audio::TARGET_RATE;

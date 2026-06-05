//! # hearsay-app
//!
//! The application/orchestration layer. Depends only on `hearsay-core`'s ports, so it
//! knows nothing about cpal, whisper, the network, or the OS. It wires a source →
//! aggregator → transcriber → stabilizer → sink into the STT half of the pipeline;
//! the conversation orchestration (LLM/TTS/playback) lives in `hearsay-engine`.
//!
//! - [`pipeline::Pipeline`] — single-threaded STT driver (tests).
//! - [`pipeline::infer_window`] / [`pipeline::flush`] — the reusable steps the
//!   engine's two-thread worker calls across the aggregator/inference split.

pub mod error;
pub mod pipeline;

pub use error::PipelineError;
pub use pipeline::{flush, infer_window, Pipeline, StepOutcome};

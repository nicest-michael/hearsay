//! # hearsay-engine
//!
//! The UI-agnostic runtime: the conversation loop that wires the STT half
//! (capture → VAD → whisper) to the dialog half (LLM → sentence chunker → Kokoro
//! TTS → interruptible playback) around the pure
//! [`hearsay_core::conversation::Dialog`] FSM, with barge-in. The desktop shell
//! (Tauri) is a thin adapter over this.

pub mod conversation_loop;
pub mod events;

pub use conversation_loop::{spawn, Ports};
pub use events::{ConvState, EngineConfig, TurnRole, UiCommand, UiEvent};

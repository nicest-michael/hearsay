//! # hearsay-engine
//!
//! The UI-agnostic runtime: the conversation loop that wires the STT half
//! (capture → VAD → whisper) to the dialog half (LLM → sentence chunker → MisoTTS →
//! interruptible playback) around the pure [`hearsay_core::conversation::Dialog`] FSM,
//! with barge-in cancellation. The desktop shell (Tauri) is a thin adapter over this.
//!
//! Built out in Task 8 (`conversation_loop`, `events`). This stub keeps the workspace
//! green during scaffolding.

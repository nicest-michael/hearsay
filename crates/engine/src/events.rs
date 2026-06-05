//! Types crossing the engine boundary: what a UI sends ([`UiCommand`]), what the
//! engine emits ([`UiEvent`]), and the run config ([`EngineConfig`]). The Tauri shell
//! maps these to/from camelCase wire DTOs — domain/engine types never cross the IPC
//! boundary directly.

use std::path::PathBuf;

use hearsay_core::config::{ModelChoice, SessionConfig};
use hearsay_core::vad::VadConfig;

/// Everything the engine needs to run a conversation.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Microphone device name; `None` = system default input.
    pub mic: Option<String>,
    pub whisper_model: ModelChoice,
    pub model_dir: PathBuf,
    pub session: SessionConfig,
    pub vad: VadConfig,
    /// Base URL of the OpenAI-compatible LLM server, e.g. `http://127.0.0.1:8080`.
    pub llm_base: String,
    /// LLM model id the server was started with.
    pub llm_model: String,
    /// The system persona prompt.
    pub persona: String,
    /// Keep the system turn + this many recent turns in history.
    pub history_keep: usize,
}

/// UI → engine.
#[derive(Clone, Debug)]
pub enum UiCommand {
    /// Stop the conversation (threads exit; sidecars stay warm — owned by the shell).
    Stop,
    /// Stop and tear down the engine.
    Shutdown,
}

/// Which side of the conversation a committed turn belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnRole {
    User,
    Assistant,
}

/// Coarse conversation state for the UI (mirrors the FSM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvState {
    Idle,
    Listening,
    Thinking,
    Speaking,
}

/// engine → UI.
#[derive(Clone, Debug)]
pub enum UiEvent {
    /// Free-text status during startup ("Loading models…").
    Status(String),
    /// Capture started and models are ready — the agent is listening.
    Started,
    /// FSM state changed.
    State(ConvState),
    /// A finalized turn to append to the transcript.
    Turn { role: TurnRole, text: String },
    /// Live input level (linear RMS ~0.0..1.0) for the mic meter.
    Level(f32),
    /// A recoverable error to surface.
    Error(String),
    /// The engine stopped.
    Stopped,
}

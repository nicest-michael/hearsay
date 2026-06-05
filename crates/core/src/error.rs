//! Domain error types. Hand-rolled (no `thiserror`) so the core stays
//! dependency-free and the "zero I/O deps" property is ironclad.

use std::fmt;

/// Something went wrong capturing audio from a source (mic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    /// The requested device/source could not be found.
    DeviceNotFound(String),
    /// The platform/backend does not support this capture mode.
    Unsupported(String),
    /// The OS denied permission (TCC) to capture.
    PermissionDenied(String),
    /// A lower-level backend error, stringified at the adapter boundary.
    Backend(String),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CaptureError::DeviceNotFound(s) => write!(f, "audio device not found: {s}"),
            CaptureError::Unsupported(s) => write!(f, "capture unsupported: {s}"),
            CaptureError::PermissionDenied(s) => write!(f, "permission denied: {s}"),
            CaptureError::Backend(s) => write!(f, "audio backend error: {s}"),
        }
    }
}
impl std::error::Error for CaptureError {}

/// Something went wrong turning audio into text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscribeError {
    /// The model failed to load.
    ModelLoad(String),
    /// Inference failed.
    Inference(String),
}

impl fmt::Display for TranscribeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranscribeError::ModelLoad(s) => write!(f, "model load failed: {s}"),
            TranscribeError::Inference(s) => write!(f, "transcription failed: {s}"),
        }
    }
}
impl std::error::Error for TranscribeError {}

/// Something went wrong talking to the language model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmError {
    /// Could not reach the local model server (not started / wrong port).
    Unreachable(String),
    /// The server returned an error status or unparseable stream.
    Protocol(String),
    /// The request was cancelled by a barge-in (not a real failure).
    Cancelled,
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::Unreachable(s) => write!(f, "llm unreachable: {s}"),
            LlmError::Protocol(s) => write!(f, "llm protocol error: {s}"),
            LlmError::Cancelled => write!(f, "llm request cancelled"),
        }
    }
}
impl std::error::Error for LlmError {}

/// Something went wrong synthesizing speech.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynthError {
    /// Could not reach the TTS sidecar (not started / socket gone).
    Unreachable(String),
    /// The sidecar returned an error or unframeable stream.
    Protocol(String),
    /// Synthesis was cancelled by a barge-in (not a real failure).
    Cancelled,
}

impl fmt::Display for SynthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SynthError::Unreachable(s) => write!(f, "tts unreachable: {s}"),
            SynthError::Protocol(s) => write!(f, "tts protocol error: {s}"),
            SynthError::Cancelled => write!(f, "tts synthesis cancelled"),
        }
    }
}
impl std::error::Error for SynthError {}

/// Something went wrong sending audio to the output device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaybackError(pub String);

impl fmt::Display for PlaybackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "playback error: {}", self.0)
    }
}
impl std::error::Error for PlaybackError {}

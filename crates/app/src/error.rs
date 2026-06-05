//! Application-layer error: a capture or a transcription failure during a tick.

use hearsay_core::error::{CaptureError, TranscribeError};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineError {
    Capture(CaptureError),
    Transcribe(TranscribeError),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::Capture(e) => write!(f, "{e}"),
            PipelineError::Transcribe(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for PipelineError {}

impl From<CaptureError> for PipelineError {
    fn from(e: CaptureError) -> Self {
        PipelineError::Capture(e)
    }
}
impl From<TranscribeError> for PipelineError {
    fn from(e: TranscribeError) -> Self {
        PipelineError::Transcribe(e)
    }
}

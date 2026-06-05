//! Ports: the interfaces the app/domain depend on, implemented by adapters in the
//! outer layers. **Every method speaks only in domain types** — no cpal, whisper,
//! reqwest, socket, or process types leak across this boundary. This is the seam that
//! keeps the dependency rule pointing inward.

use std::sync::atomic::AtomicBool;

use crate::dialogue::Turn;
use crate::error::{CaptureError, LlmError, PlaybackError, SynthError, TranscribeError};
use crate::vad::SpeechState;
use crate::word::Hypothesis;

// ---------------------------------------------------------------------------
// STT half (inherited from earshot)
// ---------------------------------------------------------------------------

/// A driven port that yields captured audio already normalized to **16 kHz mono
/// f32**. Non-blocking pull model: the adapter hides any real-time callback /
/// ring buffer behind [`AudioSource::poll`].
pub trait AudioSource: Send {
    /// Begin capture. Idempotent — calling while already running is a no-op.
    fn start(&mut self) -> Result<(), CaptureError>;
    /// Return audio captured since the last poll (16 kHz mono), empty if none yet.
    /// Never blocks.
    fn poll(&mut self) -> Result<Vec<f32>, CaptureError>;
    /// Stop capture and release the device.
    fn stop(&mut self);
    /// Human label, e.g. "MacBook Pro Microphone".
    fn label(&self) -> &str;
}

/// A driven port that turns a window of 16 kHz mono audio into a word-level
/// [`Hypothesis`] with timestamps and per-segment confidence.
pub trait Transcriber: Send {
    fn transcribe(&mut self, audio_16k_mono: &[f32]) -> Result<Hypothesis, TranscribeError>;
}

/// A driven port for voice-activity detection. A pure energy impl lives in
/// [`crate::vad`]; a Silero/ONNX adapter can replace it behind a feature flag.
pub trait SpeechDetector: Send {
    fn observe(&mut self, frame_16k_mono: &[f32]) -> SpeechState;
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// Conversation half (new)
// ---------------------------------------------------------------------------

/// A driven port: a streaming chat LLM. `reply` streams assistant text deltas to
/// `on_delta` until the model is done or `cancel` flips true (the adapter checks it
/// between tokens and aborts). Returns the full assistant text produced so far.
pub trait LlmClient: Send {
    fn reply(
        &self,
        history: &[Turn],
        cancel: &AtomicBool,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String, LlmError>;
}

/// A driven port: text → mono f32 audio at [`SpeechSynthesizer::sample_rate`],
/// delivered in chunks via `on_pcm` as they are synthesized (not buffered to the
/// end). `cancel` is checked between chunks/frames so a barge-in aborts ASAP.
pub trait SpeechSynthesizer: Send {
    fn speak(
        &mut self,
        text: &str,
        cancel: &AtomicBool,
        on_pcm: &mut dyn FnMut(&[f32]),
    ) -> Result<(), SynthError>;
    /// Native sample rate of the synthesized PCM (MisoTTS/Mimi → 24 kHz).
    fn sample_rate(&self) -> u32;
}

/// A driven port: an interruptible audio sink. `enqueue` appends f32 frames at the
/// synthesizer's sample rate (the adapter resamples to the device); `barge_stop`
/// flushes everything pending immediately for an instant audible cut; `played_samples`
/// reports what actually reached the device (for "how much did they hear").
pub trait AudioPlayer: Send {
    fn enqueue(&mut self, pcm: &[f32]) -> Result<(), PlaybackError>;
    fn barge_stop(&mut self);
    fn played_samples(&self) -> u64;
    fn is_draining(&self) -> bool;
    /// Sample rate the player expects `enqueue`d PCM to be in.
    fn input_sample_rate(&self) -> u32;
}

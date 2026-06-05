//! Pure configuration value types shared across layers. No I/O, no serde here —
//! adapters/binary own persistence so the domain stays framework-free.

/// Which kind of audio a source captures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceKind {
    /// A microphone / input device.
    Microphone,
    /// The computer's own output ("system audio").
    SystemAudio,
}

impl SourceKind {
    /// Short label for UI/overlay/clipboard headings.
    pub fn label(self) -> &'static str {
        match self {
            SourceKind::Microphone => "Mic",
            SourceKind::SystemAudio => "System",
        }
    }
}

/// A Whisper ggml model the user can pick. Fast → accurate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ModelChoice {
    TinyEn,
    #[default]
    BaseEn,
    SmallEn,
    LargeV3Turbo,
}

impl ModelChoice {
    pub const ALL: [ModelChoice; 4] = [
        ModelChoice::TinyEn,
        ModelChoice::BaseEn,
        ModelChoice::SmallEn,
        ModelChoice::LargeV3Turbo,
    ];

    /// Human label with a speed/accuracy hint.
    pub fn label(self) -> &'static str {
        match self {
            ModelChoice::TinyEn => "Tiny (fastest)",
            ModelChoice::BaseEn => "Base (recommended)",
            ModelChoice::SmallEn => "Small (more accurate)",
            ModelChoice::LargeV3Turbo => "Large v3 Turbo (most accurate)",
        }
    }

    /// Whether the model is English-only (`.en`) — lets the transcriber force `en`.
    pub fn is_english_only(self) -> bool {
        !matches!(self, ModelChoice::LargeV3Turbo)
    }
}

/// Sliding-window / commit policy for a transcription session. Centiseconds (cs)
/// throughout for parity with Whisper timestamps.
#[derive(Clone, Copy, Debug)]
pub struct SessionConfig {
    /// Emit a window for transcription roughly every `step_cs`.
    pub step_cs: u32,
    /// Hard cap on buffered audio (safety; trimming normally keeps it shorter).
    pub window_max_cs: u32,
    /// Once buffered audio is older than this, trim it at committed boundaries.
    pub buffer_trimming_cs: u32,
    /// Audio kept *before* the committed boundary as decoder context.
    pub preroll_cs: u32,
    /// Silence this long marks utterance end → flush volatile tail.
    pub utterance_silence_ms: u32,
    /// Drop segments with `no_speech_prob` above this (anti-hallucination gate).
    pub no_speech_max: f32,
    /// Drop segments with `avg_logprob` below this.
    pub logprob_min: f32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            step_cs: 100,            // 1.0 s
            window_max_cs: 1200,     // 12 s hard cap
            buffer_trimming_cs: 800, // trim once buffer > 8 s
            preroll_cs: 50,          // keep 0.5 s context behind the boundary
            utterance_silence_ms: 700,
            no_speech_max: 0.6,
            logprob_min: -1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_flags() {
        assert!(ModelChoice::BaseEn.is_english_only());
        assert!(!ModelChoice::LargeV3Turbo.is_english_only());
        assert_eq!(ModelChoice::default(), ModelChoice::BaseEn);
        assert_eq!(ModelChoice::ALL.len(), 4);
    }

    #[test]
    fn source_kind_labels() {
        assert_eq!(SourceKind::Microphone.label(), "Mic");
        assert_eq!(SourceKind::SystemAudio.label(), "System");
    }

    #[test]
    fn default_session_config_is_sane() {
        let c = SessionConfig::default();
        assert!(c.buffer_trimming_cs < c.window_max_cs);
        assert!(c.preroll_cs < c.buffer_trimming_cs);
    }
}

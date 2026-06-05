//! The sliding-window FSM: buffer 16 kHz mono audio, run VAD, hand out windows to
//! transcribe, and **trim the buffer at committed boundaries** so re-transcription
//! cost stays bounded (council revision R2 — the canonical whisper_streaming fix).
//!
//! Pure and single-threaded. In the running app one instance lives on the
//! aggregator thread per source; the inference thread feeds committed times back
//! via [`WindowAggregator::trim_to`].

use crate::audio::{cs_for_samples, samples_for_cs};
use crate::config::SessionConfig;
use crate::vad::{SpeechState, Vad, VadConfig};

/// 30 ms of 16 kHz audio — the VAD analysis frame.
const VAD_FRAME: usize = 480;

/// A chunk of audio to transcribe, tagged with its absolute start time so the
/// resulting hypothesis can be shifted to the global timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub samples: Vec<f32>,
    pub offset_cs: u32,
}

/// Buffers audio for one source and decides when to transcribe / trim.
#[derive(Clone, Debug)]
pub struct WindowAggregator {
    cfg: SessionConfig,
    vad: Vad,
    buf: Vec<f32>,
    /// Absolute sample index of `buf[0]` (source of truth for time; avoids drift).
    offset_samples: usize,
    samples_since_step: usize,
    vad_carry: Vec<f32>,
    /// Any speech observed since the last trim/flush (gate: don't transcribe pure silence).
    speech_in_window: bool,
}

impl WindowAggregator {
    pub fn new(cfg: SessionConfig, vad_cfg: VadConfig) -> Self {
        Self {
            cfg,
            vad: Vad::new(vad_cfg),
            buf: Vec::new(),
            offset_samples: 0,
            samples_since_step: 0,
            vad_carry: Vec::new(),
            speech_in_window: false,
        }
    }

    fn step_samples(&self) -> usize {
        samples_for_cs(self.cfg.step_cs).max(1)
    }
    fn window_max_samples(&self) -> usize {
        samples_for_cs(self.cfg.window_max_cs).max(1)
    }

    /// Absolute centisecond time of `buf[0]`.
    pub fn offset_cs(&self) -> u32 {
        cs_for_samples(self.offset_samples)
    }

    /// Current buffered duration in centiseconds.
    pub fn buffer_len_cs(&self) -> u32 {
        cs_for_samples(self.buf.len())
    }

    pub fn vad_state(&self) -> SpeechState {
        self.vad.state()
    }
    pub fn vad_enter_threshold(&self) -> f32 {
        self.vad.enter_threshold()
    }
    /// Current input RMS level (most recent frame) — for a UI meter.
    pub fn level(&self) -> f32 {
        self.vad.last_rms()
    }

    /// Append captured 16 kHz mono audio, updating VAD and the hard size cap.
    pub fn push_audio(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        self.buf.extend_from_slice(samples);
        self.samples_since_step += samples.len();
        self.feed_vad(samples);
        self.enforce_hard_cap();
    }

    fn feed_vad(&mut self, samples: &[f32]) {
        self.vad_carry.extend_from_slice(samples);
        while self.vad_carry.len() >= VAD_FRAME {
            let rest = self.vad_carry.split_off(VAD_FRAME);
            let frame = std::mem::replace(&mut self.vad_carry, rest);
            if self.vad.observe(&frame) == SpeechState::Speech {
                self.speech_in_window = true;
            }
        }
    }

    /// Never let the buffer exceed `window_max_cs` (safety net independent of trimming).
    fn enforce_hard_cap(&mut self) {
        let cap = self.window_max_samples();
        if self.buf.len() > cap {
            let drop = self.buf.len() - cap;
            self.buf.drain(..drop);
            self.offset_samples += drop;
        }
    }

    /// If a step's worth of new audio has arrived and the buffer contains speech,
    /// hand out the current buffer as a window to transcribe.
    pub fn poll_window(&mut self) -> Option<Window> {
        if self.buf.is_empty() || !self.speech_in_window {
            return None;
        }
        if self.samples_since_step < self.step_samples() {
            return None;
        }
        self.samples_since_step = 0;
        Some(Window {
            samples: self.buf.clone(),
            offset_cs: self.offset_cs(),
        })
    }

    /// Trim the buffer up to the committed absolute time (minus pre-roll), but only
    /// once the buffer has grown past `buffer_trimming_cs`. Monotonic and idempotent.
    pub fn trim_to(&mut self, committed_abs_cs: u32) {
        if self.buffer_len_cs() < self.cfg.buffer_trimming_cs {
            return;
        }
        let keep_from_cs = committed_abs_cs.saturating_sub(self.cfg.preroll_cs);
        let keep_from_sample = samples_for_cs(keep_from_cs);
        if keep_from_sample <= self.offset_samples {
            return; // already trimmed past here
        }
        let drop = (keep_from_sample - self.offset_samples).min(self.buf.len());
        self.buf.drain(..drop);
        self.offset_samples += drop;
    }

    /// True once we've heard enough trailing silence after speech to call the
    /// utterance finished (→ the app should flush the volatile tail).
    pub fn utterance_ended(&self) -> bool {
        self.speech_in_window && self.vad.silence_run_ms() >= self.cfg.utterance_silence_ms
    }

    /// Reset after the app flushed an utterance: drop all but `preroll_cs` of audio,
    /// clear the speech gate, and ready the VAD for the next utterance.
    pub fn note_flushed(&mut self) {
        let keep = samples_for_cs(self.cfg.preroll_cs).min(self.buf.len());
        let drop = self.buf.len() - keep;
        if drop > 0 {
            self.buf.drain(..drop);
            self.offset_samples += drop;
        }
        self.speech_in_window = false;
        self.samples_since_step = 0;
        // Drop any partial VAD frame: leftover loud samples from the finished
        // utterance must not bleed into the next frame and re-trigger speech.
        self.vad_carry.clear();
        self.vad.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen(level: f32, cs: u32) -> Vec<f32> {
        let n = samples_for_cs(cs);
        (0..n)
            .map(|i| if i % 2 == 0 { level } else { -level })
            .collect()
    }

    fn agg() -> WindowAggregator {
        WindowAggregator::new(SessionConfig::default(), VadConfig::default())
    }

    #[test]
    fn no_window_before_a_step_elapses() {
        let mut a = agg();
        a.push_audio(&gen(0.3, 50)); // 0.5 s speech, < 1 s step
        assert!(a.poll_window().is_none());
    }

    #[test]
    fn emits_window_after_step_with_speech() {
        let mut a = agg();
        a.push_audio(&gen(0.3, 120)); // 1.2 s of speech
        let win = a.poll_window().expect("window after a step of speech");
        assert_eq!(win.offset_cs, 0);
        assert!(!win.samples.is_empty());
        // immediately polling again yields nothing (step counter reset)
        assert!(a.poll_window().is_none());
    }

    #[test]
    fn pure_silence_never_emits_a_window() {
        let mut a = agg();
        a.push_audio(&gen(0.00001, 300)); // 3 s of silence
        assert!(a.poll_window().is_none());
    }

    #[test]
    fn hard_cap_bounds_the_buffer_and_advances_offset() {
        let mut a = agg();
        // push 20 s of speech in 1 s chunks; cap is 12 s
        for _ in 0..20 {
            a.push_audio(&gen(0.3, 100));
        }
        assert!(a.buffer_len_cs() <= SessionConfig::default().window_max_cs);
        assert!(
            a.offset_cs() > 0,
            "offset should advance as oldest audio is dropped"
        );
    }

    #[test]
    fn trim_to_drops_committed_audio_keeping_preroll() {
        let mut a = agg();
        // 10 s buffered (> 8 s trimming threshold)
        a.push_audio(&gen(0.3, 1000));
        let before = a.buffer_len_cs();
        a.trim_to(600); // committed up to 6 s
        let after = a.buffer_len_cs();
        assert!(after < before, "buffer should shrink after trim");
        // offset should be ~ (600 - 50 preroll) = 550 cs
        assert!(
            a.offset_cs() >= 540 && a.offset_cs() <= 560,
            "offset_cs = {}",
            a.offset_cs()
        );
    }

    #[test]
    fn trim_is_noop_when_buffer_is_short() {
        let mut a = agg();
        a.push_audio(&gen(0.3, 300)); // 3 s, < 8 s threshold
        let before = a.offset_cs();
        a.trim_to(200);
        assert_eq!(a.offset_cs(), before, "short buffers are not trimmed");
    }

    #[test]
    fn trim_is_monotonic() {
        let mut a = agg();
        a.push_audio(&gen(0.3, 1000));
        a.trim_to(600);
        let off = a.offset_cs();
        a.trim_to(300); // earlier than already trimmed -> ignored
        assert_eq!(a.offset_cs(), off);
    }

    #[test]
    fn utterance_end_after_trailing_silence() {
        let mut a = agg();
        a.push_audio(&gen(0.3, 100)); // speech
        assert!(!a.utterance_ended());
        a.push_audio(&gen(0.00001, 100)); // 1 s silence > 700 ms
        assert!(a.utterance_ended());
    }

    #[test]
    fn note_flushed_clears_speech_and_shrinks_to_preroll() {
        let mut a = agg();
        a.push_audio(&gen(0.3, 500));
        a.note_flushed();
        assert!(a.buffer_len_cs() <= SessionConfig::default().preroll_cs + 5);
        // after flush, fresh silence does not immediately re-emit
        a.push_audio(&gen(0.00001, 200));
        assert!(a.poll_window().is_none());
    }
}

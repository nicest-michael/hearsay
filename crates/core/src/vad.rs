//! Energy voice-activity detection with an **adaptive noise floor** and hysteresis.
//!
//! Per council revision R3, energy VAD is used only to (a) skip silence and
//! (b) detect utterance boundaries — *not* as the anti-hallucination mechanism
//! (that's the no-speech gate in [`crate::word::Hypothesis::confident_words`]).
//!
//! A fixed RMS threshold mis-fires across devices/rooms, so the threshold tracks a
//! rolling noise floor estimated during silence: `enter = floor × enter_mult`,
//! with a lower `exit = floor × exit_mult` so brief dips don't chop speech.

use crate::audio::{cs_for_samples, rms};

/// Speech vs silence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeechState {
    Speech,
    Silence,
}

/// Tunables for [`Vad`].
#[derive(Clone, Copy, Debug)]
pub struct VadConfig {
    /// Initial noise-floor estimate (RMS).
    pub noise_floor_init: f32,
    /// Lower bound on the noise floor, so the threshold never collapses to ~0
    /// in perfect digital silence (which would let any dither trigger speech).
    pub floor_min: f32,
    /// Enter-speech threshold multiplier over the noise floor.
    pub enter_mult: f32,
    /// Exit-speech threshold multiplier (< `enter_mult` → hysteresis).
    pub exit_mult: f32,
    /// How long RMS must stay below the exit threshold before we call it silence.
    pub hangover_ms: u32,
    /// EMA smoothing for the noise-floor estimate (updated only during silence).
    pub ema_alpha: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            noise_floor_init: 0.003,
            floor_min: 0.0008,
            enter_mult: 3.5,
            exit_mult: 2.0,
            hangover_ms: 320,
            ema_alpha: 0.05,
        }
    }
}

/// Stateful energy VAD. Feed it consecutive frames of 16 kHz mono audio.
#[derive(Clone, Debug)]
pub struct Vad {
    cfg: VadConfig,
    noise_floor: f32,
    silence_run_ms: u32,
    state: SpeechState,
    last_rms: f32,
}

impl Vad {
    pub fn new(cfg: VadConfig) -> Self {
        Self {
            noise_floor: cfg.noise_floor_init.max(cfg.floor_min),
            silence_run_ms: 0,
            state: SpeechState::Silence,
            last_rms: 0.0,
            cfg,
        }
    }

    /// RMS level of the most recently observed frame (for a UI meter).
    pub fn last_rms(&self) -> f32 {
        self.last_rms
    }

    /// Current enter-speech threshold (RMS).
    pub fn enter_threshold(&self) -> f32 {
        self.noise_floor * self.cfg.enter_mult
    }

    /// Observe one frame of 16 kHz mono audio; returns the updated state.
    pub fn observe(&mut self, frame: &[f32]) -> SpeechState {
        let level = rms(frame);
        self.last_rms = level;
        let frame_ms = (cs_for_samples(frame.len()) * 10).max(1); // cs → ms, min 1
        let enter = self.noise_floor * self.cfg.enter_mult;
        let exit = self.noise_floor * self.cfg.exit_mult;

        match self.state {
            SpeechState::Silence => {
                if level >= enter {
                    self.state = SpeechState::Speech;
                    self.silence_run_ms = 0;
                } else {
                    // Track the noise floor only while we believe it's silence.
                    self.noise_floor = ((1.0 - self.cfg.ema_alpha) * self.noise_floor
                        + self.cfg.ema_alpha * level)
                        .max(self.cfg.floor_min);
                    self.silence_run_ms = self.silence_run_ms.saturating_add(frame_ms);
                }
            }
            SpeechState::Speech => {
                if level < exit {
                    self.silence_run_ms = self.silence_run_ms.saturating_add(frame_ms);
                    if self.silence_run_ms >= self.cfg.hangover_ms {
                        self.state = SpeechState::Silence;
                    }
                } else {
                    self.silence_run_ms = 0;
                }
            }
        }
        self.state
    }

    pub fn state(&self) -> SpeechState {
        self.state
    }

    /// How long we've observed sub-threshold audio (ms). Used to detect
    /// utterance end (flush + trim).
    pub fn silence_run_ms(&self) -> u32 {
        self.silence_run_ms
    }

    pub fn reset(&mut self) {
        self.silence_run_ms = 0;
        self.state = SpeechState::Silence;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(level: f32, n: usize) -> Vec<f32> {
        // alternating ±level gives rms == level
        (0..n)
            .map(|i| if i % 2 == 0 { level } else { -level })
            .collect()
    }

    #[test]
    fn loud_frame_enters_speech_immediately() {
        let mut v = Vad::new(VadConfig::default());
        assert_eq!(v.observe(&frame(0.2, 480)), SpeechState::Speech);
    }

    #[test]
    fn silence_stays_silence() {
        let mut v = Vad::new(VadConfig::default());
        for _ in 0..50 {
            assert_eq!(v.observe(&frame(0.0001, 480)), SpeechState::Silence);
        }
    }

    #[test]
    fn hangover_holds_then_drops_to_silence() {
        let mut v = Vad::new(VadConfig::default());
        v.observe(&frame(0.3, 480)); // -> Speech
                                     // 30ms frames below exit threshold; hangover is 320ms
        let mut transitions = 0;
        for _ in 0..(320 / 30 + 2) {
            if v.observe(&frame(0.00001, 480)) == SpeechState::Silence {
                transitions += 1;
            }
        }
        assert!(
            transitions >= 1,
            "should eventually return to silence after hangover"
        );
    }

    #[test]
    fn brief_dip_does_not_chop_speech() {
        let mut v = Vad::new(VadConfig::default());
        v.observe(&frame(0.3, 480)); // Speech
                                     // one short quiet frame (< hangover) keeps speech
        assert_eq!(v.observe(&frame(0.00001, 160)), SpeechState::Speech); // 10ms
        assert_eq!(v.observe(&frame(0.3, 480)), SpeechState::Speech);
    }

    #[test]
    fn noise_floor_adapts_upward_in_steady_noise() {
        let mut v = Vad::new(VadConfig::default());
        let start = v.enter_threshold();
        // steady noise ABOVE the initial floor (0.003) but below the enter
        // threshold (~0.0105) -> the floor (and threshold) should climb toward it.
        for _ in 0..200 {
            v.observe(&frame(0.006, 480));
        }
        assert!(
            v.enter_threshold() > start,
            "threshold should rise as noise floor adapts: {} !> {}",
            v.enter_threshold(),
            start
        );
    }

    #[test]
    fn reset_returns_to_silence() {
        let mut v = Vad::new(VadConfig::default());
        v.observe(&frame(0.5, 480));
        v.reset();
        assert_eq!(v.state(), SpeechState::Silence);
        assert_eq!(v.silence_run_ms(), 0);
    }
}

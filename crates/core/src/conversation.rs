//! The dialog brain: a **pure** turn-taking state machine. `Dialog::on(event)` maps
//! `(State, Event) -> Vec<Effect>` with no I/O, no threads, no clock. The engine
//! (`hearsay-engine`) executes the `Effect`s against the real ports (LLM, TTS, player)
//! and feeds results back as `Event`s. A monotonic `turn` id is bumped on every new
//! user utterance; any `Event` carrying a stale turn id is ignored, which is how
//! barge-in invalidates in-flight LLM/TTS output without races.
//!
//! ## States
//! - `Idle` — nothing happening; mic is hot, waiting for speech.
//! - `Listening` — the user is speaking; STT is accumulating their utterance.
//! - `Thinking` — the LLM is generating; no audio has played yet.
//! - `Speaking` — the agent's TTS audio is playing (mic still hot for barge-in).
//!
//! ## Barge-in
//! `UserSpeechStarted` while `Thinking`/`Speaking` is the interruption: emit
//! `CancelLlm` + `CancelTts` + `StopPlayback` (the engine flushes playback for an
//! instant audible cut and frees the GPU), and return to `Listening` for the
//! interruption utterance.

/// The conversation state. The user always wins; the agent never blocks listening.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Idle,
    Listening,
    Thinking,
    Speaking,
}

/// Monotonic per-utterance id. Output (LLM tokens, TTS audio) is tagged with the turn
/// it belongs to; the FSM drops anything that doesn't match the current turn.
pub type TurnId = u64;

/// Inputs to the FSM, produced by the engine from the ports.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Event {
    /// VAD detected speech onset. In `Idle` this begins a turn; in `Thinking`/
    /// `Speaking` it is a **barge-in**.
    UserSpeechStarted,
    /// STT finalized the user's utterance (endpoint reached). Empty = false start.
    UserUtterance(String),
    /// A speakable chunk of the assistant reply is ready to synthesize.
    SpeakableChunk { turn: TurnId, text: String },
    /// The first audio of `turn` reached the output device.
    PlaybackStarted { turn: TurnId },
    /// The engine's aggregated "this turn is fully done": LLM finished **and** all
    /// synthesized audio has drained from the player (or the reply was empty).
    AssistantComplete { turn: TurnId },
}

/// Side effects the engine must carry out. Pure data — the FSM performs no I/O.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Effect {
    /// Begin streaming an LLM reply for `turn` to the user's text.
    StartLlm { turn: TurnId, user_text: String },
    /// Synthesize `text` (a sentence/chunk) for `turn` and enqueue it for playback.
    Synthesize { turn: TurnId, text: String },
    /// Abort the in-flight LLM stream for `turn`.
    CancelLlm { turn: TurnId },
    /// Abort in-flight TTS synthesis for `turn`.
    CancelTts { turn: TurnId },
    /// Flush the audio output immediately (instant audible cut).
    StopPlayback,
    /// Append the user's turn to the conversation history.
    CommitUser { text: String },
    /// Append the assistant's turn (the text the engine accumulated for `turn`).
    CommitAssistant { turn: TurnId },
}

/// The pure turn-taking state machine.
#[derive(Debug)]
pub struct Dialog {
    state: State,
    turn: TurnId,
}

impl Dialog {
    pub fn new() -> Self {
        Self {
            state: State::Idle,
            turn: 0,
        }
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn turn(&self) -> TurnId {
        self.turn
    }

    /// Apply an event, returning the effects the engine must perform.
    pub fn on(&mut self, ev: Event) -> Vec<Effect> {
        match (self.state, ev) {
            // ---- Idle: speech onset begins listening ----
            (State::Idle, Event::UserSpeechStarted) => {
                self.state = State::Listening;
                vec![]
            }

            // ---- Listening: the utterance is finalized ----
            (State::Listening, Event::UserUtterance(text)) => {
                let text = text.trim().to_string();
                if text.is_empty() {
                    // false start — back to idle, no turn consumed
                    self.state = State::Idle;
                    return vec![];
                }
                self.turn += 1;
                self.state = State::Thinking;
                vec![
                    Effect::CommitUser { text: text.clone() },
                    Effect::StartLlm {
                        turn: self.turn,
                        user_text: text,
                    },
                ]
            }
            // A second speech-onset while already listening is a no-op.
            (State::Listening, Event::UserSpeechStarted) => vec![],

            // ---- Thinking/Speaking: stream LLM -> TTS, gated by turn id ----
            (State::Thinking | State::Speaking, Event::SpeakableChunk { turn, text })
                if turn == self.turn =>
            {
                vec![Effect::Synthesize { turn, text }]
            }
            (State::Thinking | State::Speaking, Event::PlaybackStarted { turn })
                if turn == self.turn =>
            {
                self.state = State::Speaking;
                vec![]
            }
            (State::Thinking | State::Speaking, Event::AssistantComplete { turn })
                if turn == self.turn =>
            {
                self.state = State::Idle;
                vec![Effect::CommitAssistant { turn }]
            }

            // ---- BARGE-IN: user speaks during Thinking or Speaking ----
            (State::Thinking | State::Speaking, Event::UserSpeechStarted) => {
                let stale = self.turn;
                self.state = State::Listening;
                vec![
                    Effect::CancelLlm { turn: stale },
                    Effect::CancelTts { turn: stale },
                    Effect::StopPlayback,
                ]
            }

            // ---- Everything else (stale turn ids, irrelevant transitions): ignore ----
            _ => vec![],
        }
    }
}

impl Default for Dialog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d() -> Dialog {
        Dialog::new()
    }

    #[test]
    fn happy_path_listen_think_speak_idle() {
        let mut d = d();
        assert_eq!(d.state(), State::Idle);

        assert!(d.on(Event::UserSpeechStarted).is_empty());
        assert_eq!(d.state(), State::Listening);

        let eff = d.on(Event::UserUtterance("what's the time".into()));
        assert_eq!(d.state(), State::Thinking);
        assert_eq!(d.turn(), 1);
        assert_eq!(
            eff,
            vec![
                Effect::CommitUser {
                    text: "what's the time".into()
                },
                Effect::StartLlm {
                    turn: 1,
                    user_text: "what's the time".into()
                },
            ]
        );

        let eff = d.on(Event::SpeakableChunk {
            turn: 1,
            text: "It's noon.".into(),
        });
        assert_eq!(
            eff,
            vec![Effect::Synthesize {
                turn: 1,
                text: "It's noon.".into()
            }]
        );

        assert!(d.on(Event::PlaybackStarted { turn: 1 }).is_empty());
        assert_eq!(d.state(), State::Speaking);

        let eff = d.on(Event::AssistantComplete { turn: 1 });
        assert_eq!(d.state(), State::Idle);
        assert_eq!(eff, vec![Effect::CommitAssistant { turn: 1 }]);
    }

    #[test]
    fn empty_utterance_is_a_false_start() {
        let mut d = d();
        d.on(Event::UserSpeechStarted);
        let eff = d.on(Event::UserUtterance("   ".into()));
        assert_eq!(d.state(), State::Idle);
        assert_eq!(d.turn(), 0);
        assert!(eff.is_empty());
    }

    #[test]
    fn barge_in_during_speaking_cancels_and_relistens() {
        let mut d = d();
        d.on(Event::UserSpeechStarted);
        d.on(Event::UserUtterance("tell me a long story".into()));
        d.on(Event::SpeakableChunk {
            turn: 1,
            text: "Once upon a time,".into(),
        });
        d.on(Event::PlaybackStarted { turn: 1 });
        assert_eq!(d.state(), State::Speaking);

        let eff = d.on(Event::UserSpeechStarted); // BARGE-IN
        assert_eq!(d.state(), State::Listening);
        assert_eq!(
            eff,
            vec![
                Effect::CancelLlm { turn: 1 },
                Effect::CancelTts { turn: 1 },
                Effect::StopPlayback,
            ]
        );
    }

    #[test]
    fn barge_in_during_thinking_before_audio() {
        let mut d = d();
        d.on(Event::UserSpeechStarted);
        d.on(Event::UserUtterance("hi".into()));
        assert_eq!(d.state(), State::Thinking);
        let eff = d.on(Event::UserSpeechStarted); // barge-in before any audio
        assert_eq!(d.state(), State::Listening);
        assert!(eff.contains(&Effect::CancelLlm { turn: 1 }));
        assert!(eff.contains(&Effect::StopPlayback));
    }

    #[test]
    fn stale_output_after_bargein_is_ignored() {
        let mut d = d();
        d.on(Event::UserSpeechStarted);
        d.on(Event::UserUtterance("a".into())); // turn 1
        d.on(Event::PlaybackStarted { turn: 1 });
        d.on(Event::UserSpeechStarted); // barge-in -> Listening
        d.on(Event::UserUtterance("b".into())); // turn 2
        assert_eq!(d.turn(), 2);
        // late turn-1 output produces nothing
        assert!(d
            .on(Event::SpeakableChunk {
                turn: 1,
                text: "late".into()
            })
            .is_empty());
        assert!(d.on(Event::PlaybackStarted { turn: 1 }).is_empty());
        assert!(d.on(Event::AssistantComplete { turn: 1 }).is_empty());
        // turn 2 still active
        assert_eq!(d.state(), State::Thinking);
    }

    #[test]
    fn empty_reply_completes_to_idle() {
        let mut d = d();
        d.on(Event::UserSpeechStarted);
        d.on(Event::UserUtterance("hi".into()));
        // no chunks, no playback — engine signals completion directly
        let eff = d.on(Event::AssistantComplete { turn: 1 });
        assert_eq!(d.state(), State::Idle);
        assert_eq!(eff, vec![Effect::CommitAssistant { turn: 1 }]);
    }

    #[test]
    fn turn_ids_increment_per_utterance() {
        let mut d = d();
        for n in 1..=3 {
            d.on(Event::UserSpeechStarted);
            d.on(Event::UserUtterance(format!("msg {n}")));
            assert_eq!(d.turn(), n);
            d.on(Event::AssistantComplete { turn: n });
            assert_eq!(d.state(), State::Idle);
        }
    }

    #[test]
    fn speech_onset_while_listening_is_noop() {
        let mut d = d();
        d.on(Event::UserSpeechStarted);
        assert_eq!(d.state(), State::Listening);
        assert!(d.on(Event::UserSpeechStarted).is_empty());
        assert_eq!(d.state(), State::Listening);
    }

    #[test]
    fn utterance_while_idle_is_ignored() {
        let mut d = d();
        let eff = d.on(Event::UserUtterance("stray".into()));
        assert_eq!(d.state(), State::Idle);
        assert!(eff.is_empty());
    }
}

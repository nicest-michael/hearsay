//! The conversation runtime: wires the STT half (mic → VAD → whisper) and the dialog
//! half (LLM → sentence chunker → TTS → interruptible playback) around the pure
//! [`Dialog`] FSM, with barge-in.
//!
//! Threads (all spawned by [`spawn`], joined on Stop/Shutdown):
//! - **capture** — polls the mic, runs VAD; emits `SpeechStarted` on onset (the
//!   barge-in trigger) and ships each finished utterance's audio to inference.
//! - **inference** — whisper-transcribes an utterance → `Utterance(text)`.
//! - **llm** — streams the reply, splits it into speakable chunks → `Chunk` / `LlmDone`.
//! - **tts** — synthesizes a chunk and enqueues it to the player → `PlaybackStarted` /
//!   `SynthDone`.
//! - **controller** (this thread) — owns the FSM + history, executes the FSM's effects
//!   against the worker threads/ports, and decides when a turn is complete.
//!
//! The controller is the single authority: only it touches the `Dialog`. Worker output
//! is tagged with a turn id; anything not matching the current turn is dropped, which is
//! how barge-in invalidates in-flight LLM/TTS work without races.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{select, tick, unbounded, Receiver, Sender};

use hearsay_core::audio::samples_for_cs;
use hearsay_core::conversation::{Dialog, Effect, Event, State};
use hearsay_core::dialogue::{Conversation, SentenceChunker, Turn};
use hearsay_core::ports::{AudioPlayer, AudioSource, LlmClient, SpeechSynthesizer, Transcriber};
use hearsay_core::vad::{SpeechState, Vad};
use hearsay_core::word::join_words;

use crate::events::{ConvState, EngineConfig, TurnRole, UiCommand, UiEvent};

/// 30 ms of 16 kHz audio — the VAD analysis frame (matches the STT aggregator).
const VAD_FRAME: usize = 480;

/// The concrete ports the engine drives (dependency-injected so tests use fakes).
pub struct Ports {
    pub source: Box<dyn AudioSource>,
    pub transcriber: Box<dyn Transcriber>,
    pub llm: Box<dyn LlmClient>,
    pub tts: Box<dyn SpeechSynthesizer>,
    pub player: Box<dyn AudioPlayer>,
}

type SharedPlayer = Arc<Mutex<Box<dyn AudioPlayer>>>;

/// Controller-bound messages from the worker threads.
#[derive(Clone, Debug)]
pub(crate) enum Msg {
    SpeechStarted,
    Utterance(String),
    Chunk { turn: u64, text: String },
    LlmDone { turn: u64, full: String },
    PlaybackStarted { turn: u64 },
    SynthDone { turn: u64 },
}

pub(crate) struct LlmJob {
    turn: u64,
    history: Vec<Turn>,
    cancel: Arc<AtomicBool>,
}

pub(crate) struct SynthJob {
    turn: u64,
    text: String,
    cancel: Arc<AtomicBool>,
}

/// Spawn the conversation. Returns the controller thread's join handle; drive it with
/// `cmd_rx` and observe it via `evt_tx`.
pub fn spawn(
    ports: Ports,
    cfg: EngineConfig,
    cmd_rx: Receiver<UiCommand>,
    evt_tx: Sender<UiEvent>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("hearsay-controller".into())
        .spawn(move || run(ports, cfg, cmd_rx, evt_tx))
        .expect("spawn controller")
}

fn run(ports: Ports, cfg: EngineConfig, cmd_rx: Receiver<UiCommand>, evt_tx: Sender<UiEvent>) {
    let Ports {
        source,
        transcriber,
        llm,
        tts,
        player,
    } = ports;
    let player: SharedPlayer = Arc::new(Mutex::new(player));
    let stop = Arc::new(AtomicBool::new(false));

    let (ctrl_tx, ctrl_rx) = unbounded::<Msg>();
    let (infer_tx, infer_rx) = unbounded::<Vec<f32>>();
    let (job_tx, job_rx) = unbounded::<LlmJob>();
    let (synth_tx, synth_rx) = unbounded::<SynthJob>();

    let handles = vec![
        spawn_capture(
            source,
            cfg.clone(),
            stop.clone(),
            ctrl_tx.clone(),
            infer_tx,
            evt_tx.clone(),
        ),
        spawn_inference(
            transcriber,
            cfg.clone(),
            infer_rx,
            ctrl_tx.clone(),
            evt_tx.clone(),
        ),
        spawn_llm(llm, job_rx, ctrl_tx.clone()),
        spawn_tts(tts, player.clone(), synth_rx, ctrl_tx),
    ];

    let _ = evt_tx.send(UiEvent::Started);
    let _ = evt_tx.send(UiEvent::State(ConvState::Idle));

    let mut ctl = Controller::new(player, job_tx, synth_tx, evt_tx.clone(), &cfg.persona, cfg.history_keep);
    let ticker = tick(Duration::from_millis(50));
    loop {
        select! {
            recv(cmd_rx) -> cmd => match cmd {
                Ok(UiCommand::Stop) | Ok(UiCommand::Shutdown) | Err(_) => break,
            },
            recv(ctrl_rx) -> msg => if let Ok(m) = msg { ctl.handle(m); },
            recv(ticker) -> _ => ctl.check_drain(),
        }
    }

    stop.store(true, Ordering::Release);
    // Cancel any in-flight turn so a worker mid-speak()/reply() bails on its next
    // read-timeout instead of blocking join() on a slow/hung sidecar.
    ctl.cur_cancel.store(true, Ordering::Release);
    drop(ctl); // drops job_tx/synth_tx -> llm/tts workers exit; capture exit drops infer_tx
    for h in handles {
        let _ = h.join();
    }
    let _ = evt_tx.send(UiEvent::Stopped);
}

// ---------------------------------------------------------------------------
// Capture: mic -> VAD -> onset (barge-in) + per-utterance audio to inference
// ---------------------------------------------------------------------------

fn spawn_capture(
    mut source: Box<dyn AudioSource>,
    cfg: EngineConfig,
    stop: Arc<AtomicBool>,
    ctrl_tx: Sender<Msg>,
    infer_tx: Sender<Vec<f32>>,
    evt_tx: Sender<UiEvent>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("hearsay-capture".into())
        .spawn(move || {
            if let Err(e) = source.start() {
                let _ = evt_tx.send(UiEvent::Error(format!("microphone: {e}")));
                return;
            }
            let mut vad = Vad::new(cfg.vad);
            let mut carry: Vec<f32> = Vec::new();
            let mut utter: Vec<f32> = Vec::new();
            let mut preroll: Vec<f32> = Vec::new();
            let preroll_cap = samples_for_cs(cfg.session.preroll_cs).max(VAD_FRAME);
            let mut in_speech = false;
            let mut tick: u64 = 0;

            while !stop.load(Ordering::Relaxed) {
                let frames = match source.poll() {
                    Ok(f) => f,
                    Err(e) => {
                        let _ = evt_tx.send(UiEvent::Error(format!("capture: {e}")));
                        break;
                    }
                };
                carry.extend_from_slice(&frames);
                while carry.len() >= VAD_FRAME {
                    let frame: Vec<f32> = carry.drain(..VAD_FRAME).collect();
                    match vad.observe(&frame) {
                        SpeechState::Speech => {
                            if !in_speech {
                                in_speech = true;
                                utter.clear();
                                utter.extend_from_slice(&preroll); // lead-in so the first word isn't clipped
                                let _ = ctrl_tx.send(Msg::SpeechStarted);
                            }
                            utter.extend_from_slice(&frame);
                        }
                        SpeechState::Silence => {
                            if in_speech {
                                utter.extend_from_slice(&frame);
                                if vad.silence_run_ms() >= cfg.session.utterance_silence_ms {
                                    let _ = infer_tx.send(std::mem::take(&mut utter));
                                    in_speech = false;
                                    vad.reset();
                                    preroll.clear();
                                }
                            } else {
                                preroll.extend_from_slice(&frame);
                                if preroll.len() > preroll_cap {
                                    let drop = preroll.len() - preroll_cap;
                                    preroll.drain(..drop);
                                }
                            }
                        }
                    }
                }
                tick = tick.wrapping_add(1);
                if tick.is_multiple_of(3) {
                    let _ = evt_tx.send(UiEvent::Level(vad.last_rms()));
                }
                thread::sleep(Duration::from_millis(10));
            }
            source.stop();
        })
        .expect("spawn capture")
}

fn spawn_inference(
    mut transcriber: Box<dyn Transcriber>,
    cfg: EngineConfig,
    infer_rx: Receiver<Vec<f32>>,
    ctrl_tx: Sender<Msg>,
    evt_tx: Sender<UiEvent>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("hearsay-inference".into())
        .spawn(move || {
            while let Ok(audio) = infer_rx.recv() {
                let text = match transcriber.transcribe(&audio) {
                    Ok(hyp) => {
                        let words =
                            hyp.confident_words(cfg.session.no_speech_max, cfg.session.logprob_min);
                        join_words(&words)
                    }
                    Err(e) => {
                        let _ = evt_tx.send(UiEvent::Error(format!("transcription: {e}")));
                        String::new()
                    }
                };
                let _ = ctrl_tx.send(Msg::Utterance(text));
            }
        })
        .expect("spawn inference")
}

fn spawn_llm(
    llm: Box<dyn LlmClient>,
    job_rx: Receiver<LlmJob>,
    ctrl_tx: Sender<Msg>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("hearsay-llm".into())
        .spawn(move || {
            while let Ok(job) = job_rx.recv() {
                let LlmJob {
                    turn,
                    history,
                    cancel,
                } = job;
                let mut chunker = SentenceChunker::new();
                let mut on_delta = |tok: &str| {
                    if let Some(chunk) = chunker.push(tok) {
                        let _ = ctrl_tx.send(Msg::Chunk { turn, text: chunk });
                    }
                };
                let full = llm
                    .reply(&history, &cancel, &mut on_delta)
                    .unwrap_or_default();
                if let Some(tail) = chunker.flush() {
                    let _ = ctrl_tx.send(Msg::Chunk { turn, text: tail });
                }
                let _ = ctrl_tx.send(Msg::LlmDone { turn, full });
            }
        })
        .expect("spawn llm")
}

fn spawn_tts(
    mut tts: Box<dyn SpeechSynthesizer>,
    player: SharedPlayer,
    synth_rx: Receiver<SynthJob>,
    ctrl_tx: Sender<Msg>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("hearsay-tts".into())
        .spawn(move || {
            while let Ok(job) = synth_rx.recv() {
                let SynthJob { turn, text, cancel } = job;
                let mut started = false;
                let res = tts.speak(&text, &cancel, &mut |pcm| {
                    if !started {
                        started = true;
                        let _ = ctrl_tx.send(Msg::PlaybackStarted { turn });
                    }
                    if let Ok(mut p) = player.lock() {
                        let _ = p.enqueue(pcm);
                    }
                });
                if let Err(e) = res {
                    log::warn!("tts synth: {e}");
                }
                let _ = ctrl_tx.send(Msg::SynthDone { turn });
            }
        })
        .expect("spawn tts")
}

// ---------------------------------------------------------------------------
// Controller: the single FSM authority + turn completion bookkeeping
// ---------------------------------------------------------------------------

pub(crate) struct Controller {
    dialog: Dialog,
    conv: Conversation,
    cur_turn: u64,
    cur_cancel: Arc<AtomicBool>,
    assistant_text: String,
    llm_done: bool,
    pending_synth: i64,
    awaiting_drain: bool,
    last_state: State,
    player: SharedPlayer,
    job_tx: Sender<LlmJob>,
    synth_tx: Sender<SynthJob>,
    evt_tx: Sender<UiEvent>,
    history_keep: usize,
}

fn conv_state(s: State) -> ConvState {
    match s {
        State::Idle => ConvState::Idle,
        State::Listening => ConvState::Listening,
        State::Thinking => ConvState::Thinking,
        State::Speaking => ConvState::Speaking,
    }
}

impl Controller {
    pub(crate) fn new(
        player: SharedPlayer,
        job_tx: Sender<LlmJob>,
        synth_tx: Sender<SynthJob>,
        evt_tx: Sender<UiEvent>,
        persona: &str,
        history_keep: usize,
    ) -> Self {
        Self {
            dialog: Dialog::new(),
            conv: Conversation::new(persona),
            cur_turn: 0,
            cur_cancel: Arc::new(AtomicBool::new(false)),
            assistant_text: String::new(),
            llm_done: false,
            pending_synth: 0,
            awaiting_drain: false,
            last_state: State::Idle,
            player,
            job_tx,
            synth_tx,
            evt_tx,
            history_keep,
        }
    }

    pub(crate) fn handle(&mut self, m: Msg) {
        match m {
            Msg::SpeechStarted => self.event(Event::UserSpeechStarted),
            Msg::Utterance(t) => self.event(Event::UserUtterance(t)),
            Msg::Chunk { turn, text } => {
                if turn == self.cur_turn {
                    self.event(Event::SpeakableChunk { turn, text });
                }
            }
            Msg::PlaybackStarted { turn } => {
                if turn == self.cur_turn {
                    self.event(Event::PlaybackStarted { turn });
                }
            }
            Msg::LlmDone { turn, full } => {
                if turn == self.cur_turn {
                    self.llm_done = true;
                    if !full.trim().is_empty() {
                        self.assistant_text = full;
                    }
                    self.maybe_complete();
                }
            }
            Msg::SynthDone { turn } => {
                if turn == self.cur_turn {
                    self.pending_synth -= 1;
                    self.maybe_complete();
                }
            }
        }
        // Don't let turn completion hinge on the ticker winning the select! race.
        self.check_drain();
    }

    fn event(&mut self, ev: Event) {
        let effects = self.dialog.on(ev);
        for e in effects {
            self.exec(e);
        }
        self.emit_state();
    }

    fn exec(&mut self, e: Effect) {
        match e {
            Effect::StartLlm { turn, user_text } => {
                self.cur_turn = turn;
                self.cur_cancel = Arc::new(AtomicBool::new(false));
                self.assistant_text.clear();
                self.llm_done = false;
                self.pending_synth = 0;
                self.awaiting_drain = false;
                // Re-arm the player after any prior barge-in so this turn's audio plays.
                if let Ok(mut p) = self.player.lock() {
                    p.resume();
                }
                // `user_text` is already committed via the preceding CommitUser effect.
                let _ = user_text;
                let _ = self.job_tx.send(LlmJob {
                    turn,
                    history: self.conv.turns().to_vec(),
                    cancel: self.cur_cancel.clone(),
                });
            }
            Effect::Synthesize { turn, text } => {
                self.pending_synth += 1;
                let _ = self.synth_tx.send(SynthJob {
                    turn,
                    text,
                    cancel: self.cur_cancel.clone(),
                });
            }
            Effect::CancelLlm { .. } | Effect::CancelTts { .. } => {
                self.cur_cancel.store(true, Ordering::Release);
            }
            Effect::StopPlayback => {
                if let Ok(mut p) = self.player.lock() {
                    p.barge_stop();
                }
            }
            Effect::CommitUser { text } => {
                self.conv.add_user(&text);
                let _ = self.evt_tx.send(UiEvent::Turn {
                    role: TurnRole::User,
                    text,
                });
            }
            Effect::CommitAssistant { .. } => {
                let text = std::mem::take(&mut self.assistant_text);
                if !text.trim().is_empty() {
                    self.conv.add_assistant(&text);
                    let _ = self.evt_tx.send(UiEvent::Turn {
                        role: TurnRole::Assistant,
                        text,
                    });
                }
                self.conv.truncate_keep_last(self.history_keep);
                self.awaiting_drain = false;
            }
        }
    }

    fn maybe_complete(&mut self) {
        if self.llm_done && self.pending_synth <= 0 {
            self.awaiting_drain = true;
            self.check_drain();
        }
    }

    /// Called on the ticker: once the LLM is done and all synth is enqueued, complete
    /// the turn as soon as the player has drained (so the commit reflects what was heard).
    pub(crate) fn check_drain(&mut self) {
        if !self.awaiting_drain {
            return;
        }
        let draining = self
            .player
            .lock()
            .map(|p| p.is_draining())
            .unwrap_or(false);
        if !draining {
            self.event(Event::AssistantComplete {
                turn: self.cur_turn,
            });
        }
    }

    fn emit_state(&mut self) {
        let s = self.dialog.state();
        if s != self.last_state {
            self.last_state = s;
            let _ = self.evt_tx.send(UiEvent::State(conv_state(s)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::Receiver;
    use hearsay_core::error::PlaybackError;

    #[derive(Default)]
    struct FakeState {
        enqueued: u64,
        consumed: u64,
        barge_stops: u32,
    }
    #[derive(Clone, Default)]
    struct FakePlayer {
        st: Arc<Mutex<FakeState>>,
    }
    impl AudioPlayer for FakePlayer {
        fn enqueue(&mut self, pcm: &[f32]) -> Result<(), PlaybackError> {
            self.st.lock().unwrap().enqueued += pcm.len() as u64;
            Ok(())
        }
        fn barge_stop(&mut self) {
            let mut s = self.st.lock().unwrap();
            s.barge_stops += 1;
            s.consumed = s.enqueued; // flush -> nothing pending
        }
        fn resume(&mut self) {}
        fn played_samples(&self) -> u64 {
            self.st.lock().unwrap().consumed
        }
        fn is_draining(&self) -> bool {
            let s = self.st.lock().unwrap();
            s.enqueued > s.consumed
        }
        fn input_sample_rate(&self) -> u32 {
            24_000
        }
    }

    struct Harness {
        ctl: Controller,
        jobs: Receiver<LlmJob>,
        synths: Receiver<SynthJob>,
        events: Receiver<UiEvent>,
        fake: Arc<Mutex<FakeState>>,
    }

    fn harness() -> Harness {
        let (job_tx, jobs) = unbounded();
        let (synth_tx, synths) = unbounded();
        let (evt_tx, events) = unbounded();
        let fp = FakePlayer::default();
        let fake = fp.st.clone();
        let player: SharedPlayer = Arc::new(Mutex::new(Box::new(fp)));
        let ctl = Controller::new(player, job_tx, synth_tx, evt_tx, "persona", 8);
        Harness {
            ctl,
            jobs,
            synths,
            events,
            fake,
        }
    }

    fn states(events: &Receiver<UiEvent>) -> Vec<ConvState> {
        events
            .try_iter()
            .filter_map(|e| match e {
                UiEvent::State(s) => Some(s),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn full_turn_commits_user_and_assistant() {
        let mut h = harness();
        h.ctl.handle(Msg::SpeechStarted);
        h.ctl.handle(Msg::Utterance("what's up".into()));
        // an LLM job was dispatched with the user turn in history
        let job = h.jobs.try_recv().expect("llm job");
        assert_eq!(job.turn, 1);
        assert_eq!(job.history.last().unwrap().text, "what's up");
        // LLM streams a chunk -> a synth job is dispatched
        h.ctl.handle(Msg::Chunk {
            turn: 1,
            text: "Not much, you?".into(),
        });
        let synth = h.synths.try_recv().expect("synth job");
        assert_eq!(synth.text, "Not much, you?");
        h.ctl.handle(Msg::PlaybackStarted { turn: 1 });
        h.ctl.handle(Msg::LlmDone {
            turn: 1,
            full: "Not much, you?".into(),
        });
        h.ctl.handle(Msg::SynthDone { turn: 1 });
        // player drained (fake: barge flushed nothing; enqueued==0 here since fake synth
        // didn't enqueue) -> completes
        h.ctl.check_drain();
        assert_eq!(h.ctl.dialog.state(), State::Idle);
        let turns: Vec<_> = h
            .events
            .try_iter()
            .filter_map(|e| match e {
                UiEvent::Turn { role, text } => Some((role, text)),
                _ => None,
            })
            .collect();
        assert!(turns.contains(&(TurnRole::User, "what's up".to_string())));
        assert!(turns.contains(&(TurnRole::Assistant, "Not much, you?".to_string())));
    }

    #[test]
    fn barge_in_flushes_player_and_cancels() {
        let mut h = harness();
        h.ctl.handle(Msg::SpeechStarted);
        h.ctl.handle(Msg::Utterance("tell me a story".into()));
        let _ = h.jobs.try_recv();
        h.ctl.handle(Msg::Chunk {
            turn: 1,
            text: "Once upon a time,".into(),
        });
        let _ = h.synths.try_recv();
        h.ctl.handle(Msg::PlaybackStarted { turn: 1 });
        assert_eq!(h.ctl.dialog.state(), State::Speaking);
        let cancel = h.ctl.cur_cancel.clone();

        // BARGE-IN
        h.ctl.handle(Msg::SpeechStarted);
        assert_eq!(h.ctl.dialog.state(), State::Listening);
        assert!(cancel.load(Ordering::Relaxed), "turn-1 cancel flag set");
        assert_eq!(h.fake.lock().unwrap().barge_stops, 1, "player flushed");

        // the interruption becomes a new turn
        h.ctl.handle(Msg::Utterance("never mind".into()));
        let job2 = h.jobs.try_recv().expect("second llm job");
        assert_eq!(job2.turn, 2);
        // late output from turn 1 is ignored
        h.ctl.handle(Msg::Chunk {
            turn: 1,
            text: "the end.".into(),
        });
        assert!(h.synths.try_recv().is_err(), "stale chunk must not synth");
    }

    #[test]
    fn empty_utterance_is_a_false_start() {
        let mut h = harness();
        h.ctl.handle(Msg::SpeechStarted);
        h.ctl.handle(Msg::Utterance("   ".into()));
        assert_eq!(h.ctl.dialog.state(), State::Idle);
        assert!(h.jobs.try_recv().is_err(), "no llm job for empty utterance");
    }

    #[test]
    fn emits_state_transitions() {
        let mut h = harness();
        h.ctl.handle(Msg::SpeechStarted);
        h.ctl.handle(Msg::Utterance("hi".into()));
        h.ctl.handle(Msg::Chunk {
            turn: 1,
            text: "Hey.".into(),
        });
        h.ctl.handle(Msg::PlaybackStarted { turn: 1 });
        let seen = states(&h.events);
        assert!(seen.contains(&ConvState::Listening));
        assert!(seen.contains(&ConvState::Thinking));
        assert!(seen.contains(&ConvState::Speaking));
    }
}

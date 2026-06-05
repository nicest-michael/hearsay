//! Headless end-to-end proof of the conversation pipeline with the REAL models
//! (whisper + mlx Qwen + Kokoro), no human needed: replay a recorded utterance WAV as
//! the microphone and verify it produces a transcript → reply → synthesized speech.
//! The agent's spoken reply is recorded to `/tmp/hearsay_reply.wav`.
//!
//! Run: `cargo run -p hearsay-engine --example e2e -- /tmp/utter.wav`
//! Needs the sidecar venvs (see Makefile `setup`). HEARSAY_ROOT overrides the repo path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::unbounded;
use hearsay_adapters::{
    ensure_model, sweep_stale, FileAudioSource, MlxChat, Sidecar, SidecarTts, WhisperTranscriber,
};
use hearsay_core::config::{ModelChoice, SessionConfig};
use hearsay_core::error::PlaybackError;
use hearsay_core::ports::AudioPlayer;
use hearsay_core::vad::VadConfig;
use hearsay_engine::{EngineConfig, Ports, TurnRole, UiCommand, UiEvent};

const LLM_PORT: u16 = 8799;
const TTS_SOCK: &str = "/tmp/hearsay_e2e_tts.sock";
const LLM_MODEL: &str = "mlx-community/Qwen2.5-3B-Instruct-4bit";

/// Records everything enqueued so we can prove (and save) the agent's speech.
#[derive(Clone, Default)]
struct RecordingPlayer {
    samples: Arc<Mutex<Vec<f32>>>,
}
impl AudioPlayer for RecordingPlayer {
    fn enqueue(&mut self, pcm: &[f32]) -> Result<(), PlaybackError> {
        self.samples.lock().unwrap().extend_from_slice(pcm);
        Ok(())
    }
    fn barge_stop(&mut self) {}
    fn played_samples(&self) -> u64 {
        self.samples.lock().unwrap().len() as u64
    }
    fn is_draining(&self) -> bool {
        false // recorder never "plays", so a turn completes as soon as synth is done
    }
    fn input_sample_rate(&self) -> u32 {
        24_000
    }
}

fn repo_root() -> String {
    std::env::var("HEARSAY_ROOT").unwrap_or_else(|_| "/Users/michael/Repos/hearsay".to_string())
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let wav = std::env::args().nth(1).expect("usage: e2e <utterance.wav>");
    let root = repo_root();

    println!("[e2e] sweeping stale sidecars + starting fresh ones…");
    sweep_stale();
    std::thread::sleep(Duration::from_secs(1));
    let _llm = Sidecar::mlx_llm(&root, LLM_MODEL, LLM_PORT).expect("spawn llm");
    let _tts = Sidecar::kokoro(&root, TTS_SOCK, "af_heart").expect("spawn tts");

    let base = format!("http://127.0.0.1:{LLM_PORT}");
    println!("[e2e] waiting for LLM…");
    let chat = MlxChat::new(&base, LLM_MODEL);
    let t0 = Instant::now();
    while !chat.health() {
        assert!(t0.elapsed() < Duration::from_secs(180), "LLM never came up");
        std::thread::sleep(Duration::from_millis(500));
    }
    println!("[e2e] LLM ready in {:.0}s. Warming TTS…", t0.elapsed().as_secs_f32());
    let tts = {
        let t0 = Instant::now();
        loop {
            match SidecarTts::connect(TTS_SOCK) {
                Ok(t) => break t,
                Err(_) if t0.elapsed() < Duration::from_secs(120) => {
                    std::thread::sleep(Duration::from_millis(500))
                }
                Err(e) => panic!("TTS never ready: {e}"),
            }
        }
    };
    println!("[e2e] TTS ready. Loading whisper…");
    let dir = std::path::PathBuf::from("/tmp/hearsay-models");
    let model_path = ensure_model(&dir, ModelChoice::BaseEn, |_, _| {}).expect("whisper model");
    let transcriber = WhisperTranscriber::load(&model_path, ModelChoice::BaseEn).expect("whisper");

    let rec = RecordingPlayer::default();
    let recorded = rec.samples.clone();
    let ports = Ports {
        source: Box::new(FileAudioSource::from_wav(&wav, 1200).expect("wav")),
        transcriber: Box::new(transcriber),
        llm: Box::new(MlxChat::new(&base, LLM_MODEL)),
        tts: Box::new(tts),
        player: Box::new(rec),
    };
    let cfg = EngineConfig {
        mic: None,
        whisper_model: ModelChoice::BaseEn,
        model_dir: dir,
        session: SessionConfig::default(),
        vad: VadConfig::default(),
        llm_base: base,
        llm_model: LLM_MODEL.to_string(),
        persona: "You are Hearsay, a warm, witty buddy. Reply in one short, spoken sentence. \
No markdown or emoji."
            .to_string(),
        history_keep: 8,
    };

    let (cmd_tx, cmd_rx) = unbounded();
    let (evt_tx, evt_rx) = unbounded();
    println!("[e2e] running the conversation (replaying {wav})…");
    let handle = hearsay_engine::spawn(ports, cfg, cmd_rx, evt_tx);

    let done = Arc::new(AtomicBool::new(false));
    let mut user_text = String::new();
    let mut assistant_text = String::new();
    let deadline = Instant::now() + Duration::from_secs(45);
    while Instant::now() < deadline {
        match evt_rx.recv_timeout(Duration::from_millis(500)) {
            Ok(UiEvent::State(s)) => println!("[e2e]   state -> {s:?}"),
            Ok(UiEvent::Turn { role, text }) => {
                match role {
                    TurnRole::User => {
                        println!("[e2e]   YOU: {text}");
                        user_text = text;
                    }
                    TurnRole::Assistant => {
                        println!("[e2e]   HEARSAY: {text}");
                        assistant_text = text;
                        // give playback enqueue a beat, then finish
                        std::thread::sleep(Duration::from_millis(500));
                        done.store(true, Ordering::Relaxed);
                    }
                }
            }
            Ok(UiEvent::Error(e)) => println!("[e2e]   error: {e}"),
            Ok(_) => {}
            Err(_) => {}
        }
        if done.load(Ordering::Relaxed) {
            break;
        }
    }

    let _ = cmd_tx.send(UiCommand::Shutdown);
    let _ = handle.join();

    // Save the agent's spoken reply.
    let samples = recorded.lock().unwrap().clone();
    if !samples.is_empty() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create("/tmp/hearsay_reply.wav", spec).unwrap();
        for s in &samples {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
        }
        w.finalize().unwrap();
    }

    println!("\n===== E2E RESULT =====");
    println!("user transcript : {user_text:?}");
    println!("assistant reply : {assistant_text:?}");
    println!(
        "reply audio     : {} samples = {:.2}s -> /tmp/hearsay_reply.wav",
        samples.len(),
        samples.len() as f32 / 24_000.0
    );
    let pass = !user_text.trim().is_empty()
        && !assistant_text.trim().is_empty()
        && samples.len() > 10_000;
    println!("VERDICT         : {}", if pass { "PASS" } else { "FAIL" });
    std::process::exit(if pass { 0 } else { 1 });
}

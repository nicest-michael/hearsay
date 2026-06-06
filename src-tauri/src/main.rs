//! Hearsay — the Tauri desktop shell (a driving adapter over `hearsay-engine`).
//!
//! Click **Go**: this spawns the model sidecars (Kokoro TTS + mlx LLM) the app owns,
//! waits for them to warm up, builds the engine ports, and starts the conversation
//! engine; engine events are fanned out to the webview. **Stop** ends the conversation
//! but keeps the sidecars warm for an instant resume; closing the window or the idle
//! timer kills the sidecars (frees the model RAM) — so models never stay loaded.
//!
//! Domain/engine types never cross the IPC boundary: this file maps `UiEvent` to
//! camelCase JSON the webview understands.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{unbounded, Sender};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State, WindowEvent};

use hearsay_adapters::{
    ensure_model, sweep_stale, CpalMicSource, CpalPlayer, FileAudioSource, MlxChat, Sidecar,
    SidecarTts, WhisperTranscriber, SYNTH_RATE,
};
use hearsay_core::config::{ModelChoice, SessionConfig};
use hearsay_core::ports::{AudioPlayer, AudioSource};
use hearsay_core::vad::VadConfig;
use hearsay_engine::{ConvState, EngineConfig, Ports, TurnRole, UiCommand, UiEvent};

const LLM_PORT: u16 = 8765;
const TTS_SOCK: &str = "/tmp/hearsay_kokoro.sock";
const LLM_MODEL: &str = "mlx-community/Qwen2.5-3B-Instruct-4bit";
const DEFAULT_VOICE: &str = "af_heart";
const IDLE_UNLOAD: Duration = Duration::from_secs(300); // free model RAM after 5 min idle
const PERSONA: &str = "You are Hearsay, a warm, witty conversational buddy having a spoken \
chat. Keep replies short — one or two sentences. Be natural, friendly, and a little \
playful. Never use markdown, bullet points, numbered lists, stage directions, or emoji; \
just talk like a person.";

/// Where the sidecar scripts + venvs live. Override with HEARSAY_ROOT.
fn repo_root() -> String {
    std::env::var("HEARSAY_ROOT").unwrap_or_else(|_| "/Users/michael/Repos/hearsay".to_string())
}

fn model_dir() -> std::path::PathBuf {
    directories::ProjectDirs::from("ai", "nicest", "hearsay")
        .map(|d| d.data_dir().join("models"))
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp/hearsay-models"))
}

struct Sidecars {
    llm: Sidecar,
    tts: Sidecar,
}

struct EngineRun {
    cmd_tx: Sender<UiCommand>,
    handle: JoinHandle<()>,
}

#[derive(Default)]
struct Shared {
    sidecars: Option<Sidecars>,
    engine: Option<EngineRun>,
    starting: bool,
    last_active: Option<Instant>,
}

struct AppState(Arc<Mutex<Shared>>);

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TurnPayload {
    role: &'static str,
    text: String,
}

fn conv_state_str(s: ConvState) -> &'static str {
    match s {
        ConvState::Idle => "idle",
        ConvState::Listening => "listening",
        ConvState::Thinking => "thinking",
        ConvState::Speaking => "speaking",
    }
}

/// Fan engine events out to the webview as camelCase events.
fn forward_events(app: AppHandle, evt_rx: crossbeam_channel::Receiver<UiEvent>) {
    while let Ok(ev) = evt_rx.recv() {
        match ev {
            UiEvent::Status(s) => emit(&app, "status", s),
            UiEvent::Started => emit(&app, "started", ()),
            UiEvent::State(cs) => {
                log::info!("[ui] state: {}", conv_state_str(cs));
                emit(&app, "state", conv_state_str(cs));
            }
            UiEvent::Turn { role, text } => {
                let role = match role {
                    TurnRole::User => "user",
                    TurnRole::Assistant => "assistant",
                };
                log::info!("[ui] turn {role}: {text}");
                emit(&app, "turn", TurnPayload { role, text });
            }
            UiEvent::Level(rms) => emit(&app, "level", rms),
            UiEvent::Error(s) => emit(&app, "error", s),
            UiEvent::Stopped => {
                emit(&app, "stopped", ());
                break;
            }
        }
    }
}

fn emit<S: Serialize + Clone>(app: &AppHandle, event: &str, payload: S) {
    let _ = app.emit(event, payload);
}

/// Build ports + start the engine. Runs on a background thread (sidecar warmup blocks).
fn start_conversation(app: AppHandle, shared: Arc<Mutex<Shared>>) {
    let root = repo_root();

    // 1. Spawn the sidecars if not already warm. Killing stale ones first avoids a
    //    port/socket clash from a previous crashed run.
    {
        let g = shared.lock().unwrap();
        if g.sidecars.is_none() {
            drop(g);
            sweep_stale();
            // A fast relaunch (or a crash + reopen) can race a dying previous instance
            // that still holds the LLM port / TTS socket. Wait for them to actually free.
            wait_port_free(LLM_PORT, Duration::from_secs(6));
            let _ = std::fs::remove_file(TTS_SOCK);
            emit(&app, "status", "Starting models… (first Go takes ~15s)".to_string());
            let llm = match Sidecar::mlx_llm(&root, LLM_MODEL, LLM_PORT) {
                Ok(s) => s,
                Err(e) => return fail(&app, &shared, format!("Couldn't start the language model: {e}\n{}", SETUP_HINT)),
            };
            let tts = match Sidecar::kokoro(&root, TTS_SOCK, DEFAULT_VOICE) {
                Ok(s) => s,
                Err(e) => return fail(&app, &shared, format!("Couldn't start the voice: {e}\n{}", SETUP_HINT)),
            };
            shared.lock().unwrap().sidecars = Some(Sidecars { llm, tts });
        }
    }

    // 2. Wait for both to be ready — bail fast with the sidecar's own log if one crashes,
    //    instead of hanging on the full timeout.
    let base = format!("http://127.0.0.1:{LLM_PORT}");
    let chat = MlxChat::new(&base, LLM_MODEL);
    if let Err(e) = await_ready(
        &app,
        &shared,
        Which::Llm,
        "Loading language model",
        || chat.health(),
        Duration::from_secs(120),
    ) {
        return fail(&app, &shared, e);
    }
    let tts = match await_tts(&app, &shared, Duration::from_secs(90)) {
        Ok(t) => t,
        Err(e) => return fail(&app, &shared, e),
    };

    // 3. Speech recognition (downloads the Whisper model on first run).
    emit(&app, "status", "Loading speech recognition…".to_string());
    let dir = model_dir();
    let app2 = app.clone();
    let model_path = match ensure_model(&dir, ModelChoice::BaseEn, move |done, total| {
        if total > 0 {
            emit(
                &app2,
                "status",
                format!("Downloading speech model… {}%", done * 100 / total),
            );
        }
    }) {
        Ok(p) => p,
        Err(e) => return fail(&app, &shared, format!("speech model: {e}")),
    };
    let transcriber = match WhisperTranscriber::load(&model_path, ModelChoice::BaseEn) {
        Ok(t) => t,
        Err(e) => return fail(&app, &shared, format!("whisper load: {e}")),
    };

    // 4. Audio I/O. By default one macOS VoiceProcessingIO unit does mic + speaker with
    //    hardware echo cancellation, so the mic doesn't hear the TTS and barge-in works
    //    open-air. HEARSAY_NO_AEC, or a replay WAV (testing), uses separate cpal streams.
    let replay = std::env::var("HEARSAY_REPLAY_WAV").ok().filter(|p| !p.is_empty());
    let no_aec = std::env::var("HEARSAY_NO_AEC").is_ok() || replay.is_some();

    let open_cpal = |app: &AppHandle, shared: &Arc<Mutex<Shared>>| -> Option<Box<dyn AudioPlayer>> {
        match CpalPlayer::open(SYNTH_RATE) {
            Ok(p) => Some(Box::new(p)),
            Err(e) => {
                fail(app, shared, format!("audio output: {e}"));
                None
            }
        }
    };

    let (source, player): (Box<dyn AudioSource>, Box<dyn AudioPlayer>) = if no_aec {
        let Some(player) = open_cpal(&app, &shared) else {
            return;
        };
        let source: Box<dyn AudioSource> = match &replay {
            Some(path) => match FileAudioSource::from_wav(path, 1200) {
                Ok(s) => Box::new(s),
                Err(e) => return fail(&app, &shared, format!("replay wav: {e}")),
            },
            None => Box::new(CpalMicSource::new(None)),
        };
        (source, player)
    } else {
        match hearsay_adapters::vpio::open() {
            Ok((s, p)) => (Box::new(s), Box::new(p)),
            Err(e) => {
                // AEC unavailable (e.g. mic denied) — fall back to a plain mic + speaker.
                log::warn!("echo cancellation unavailable, using a plain mic: {e}");
                emit(&app, "status", "Echo cancellation unavailable — wear headphones.".to_string());
                let Some(player) = open_cpal(&app, &shared) else {
                    return;
                };
                (Box::new(CpalMicSource::new(None)), player)
            }
        }
    };

    let ports = Ports {
        source,
        transcriber: Box::new(transcriber),
        llm: Box::new(MlxChat::new(&base, LLM_MODEL)),
        tts: Box::new(tts),
        player,
    };
    let cfg = EngineConfig {
        mic: None,
        whisper_model: ModelChoice::BaseEn,
        model_dir: dir,
        session: SessionConfig::default(),
        vad: VadConfig::default(),
        llm_base: base,
        llm_model: LLM_MODEL.to_string(),
        persona: PERSONA.to_string(),
        history_keep: 8,
    };

    // 5. Start the engine and fan its events to the webview.
    let (cmd_tx, cmd_rx) = unbounded();
    let (evt_tx, evt_rx) = unbounded();
    let handle = hearsay_engine::spawn(ports, cfg, cmd_rx, evt_tx);
    let app3 = app.clone();
    thread::spawn(move || forward_events(app3, evt_rx));

    let mut g = shared.lock().unwrap();
    g.engine = Some(EngineRun { cmd_tx, handle });
    g.starting = false;
    g.last_active = Some(Instant::now());
}

fn fail(app: &AppHandle, shared: &Arc<Mutex<Shared>>, msg: String) {
    log::error!("{msg}");
    emit(app, "error", msg);
    emit(app, "stopped", ());
    let mut g = shared.lock().unwrap();
    g.starting = false;
    // Mark idle so the unloader still reaps any warm sidecars left by a partial start.
    g.last_active = Some(Instant::now());
}

/// Block until `port` on 127.0.0.1 can be bound (i.e. is free), or `timeout` elapses.
/// Used after `sweep_stale` so a freshly-spawned LLM server doesn't race a dying one.
fn wait_port_free(port: u16, timeout: Duration) {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(200));
    }
}

const SETUP_HINT: &str =
    "If this is the first run, open the repo in a terminal once and run: make setup";

#[derive(Clone, Copy)]
enum Which {
    Llm,
    Tts,
}

/// (has the sidecar exited?, the tail of its log). Locks `shared` briefly.
fn sidecar_status(shared: &Arc<Mutex<Shared>>, which: Which) -> (bool, String) {
    let mut g = shared.lock().unwrap();
    match g.sidecars.as_mut() {
        Some(s) => {
            let sc = match which {
                Which::Llm => &mut s.llm,
                Which::Tts => &mut s.tts,
            };
            (sc.has_exited(), sc.log_tail())
        }
        None => (true, "(no sidecar running)".to_string()),
    }
}

/// Poll `ready` until true, emitting elapsed-time status. Returns a detailed error (with
/// the sidecar's own log) if the sidecar crashes or the timeout elapses.
fn await_ready<F: Fn() -> bool>(
    app: &AppHandle,
    shared: &Arc<Mutex<Shared>>,
    which: Which,
    label: &str,
    ready: F,
    timeout: Duration,
) -> Result<(), String> {
    let start = Instant::now();
    loop {
        if ready() {
            return Ok(());
        }
        let (exited, tail) = sidecar_status(shared, which);
        if exited {
            return Err(format!("{label} crashed during startup.\n\n{tail}"));
        }
        if start.elapsed() >= timeout {
            return Err(format!(
                "{label} didn't come up in {}s.\n\n{tail}",
                timeout.as_secs()
            ));
        }
        emit(app, "status", format!("{label}… {}s", start.elapsed().as_secs()));
        thread::sleep(Duration::from_millis(400));
    }
}

/// Like [`await_ready`], but the "ready" check is connecting to the TTS socket, and it
/// returns the live connection.
fn await_tts(
    app: &AppHandle,
    shared: &Arc<Mutex<Shared>>,
    timeout: Duration,
) -> Result<SidecarTts, String> {
    let start = Instant::now();
    loop {
        if let Ok(t) = SidecarTts::connect(TTS_SOCK) {
            return Ok(t);
        }
        let (exited, tail) = sidecar_status(shared, Which::Tts);
        if exited {
            return Err(format!("The voice model crashed during startup.\n\n{tail}"));
        }
        if start.elapsed() >= timeout {
            return Err(format!(
                "The voice didn't warm up in {}s.\n\n{tail}",
                timeout.as_secs()
            ));
        }
        emit(
            app,
            "status",
            format!("Warming up the voice… {}s", start.elapsed().as_secs()),
        );
        thread::sleep(Duration::from_millis(400));
    }
}

fn stop_engine(shared: &Arc<Mutex<Shared>>) {
    let run = shared.lock().unwrap().engine.take();
    if let Some(run) = run {
        let _ = run.cmd_tx.send(UiCommand::Stop);
        let _ = run.handle.join();
    }
    shared.lock().unwrap().last_active = Some(Instant::now());
}

/// Begin a conversation (idempotent — a no-op if one is already starting/running).
fn begin(app: AppHandle, shared: Arc<Mutex<Shared>>) {
    {
        let mut g = shared.lock().unwrap();
        if g.starting || g.engine.is_some() {
            return;
        }
        g.starting = true;
    }
    thread::spawn(move || start_conversation(app, shared));
}

#[tauri::command]
fn go(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    begin(app, state.0.clone());
    Ok(())
}

#[tauri::command]
fn stop(state: State<'_, AppState>) -> Result<(), String> {
    stop_engine(&state.0);
    Ok(())
}

/// Background unloader: free the model sidecars after the app has been idle a while.
fn spawn_idle_unloader(shared: Arc<Mutex<Shared>>, app: AppHandle) {
    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(30));
        let mut g = shared.lock().unwrap();
        let idle = g.engine.is_none()
            && !g.starting
            && g.sidecars.is_some()
            && g.last_active.map(|t| t.elapsed() >= IDLE_UNLOAD).unwrap_or(false);
        if idle {
            g.sidecars = None; // Sidecar::drop kills the model processes
            drop(g);
            emit(&app, "status", "Models unloaded (idle). Click Go to resume.".to_string());
        }
    });
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let shared = Arc::new(Mutex::new(Shared::default()));
    let state = AppState(shared.clone());

    tauri::Builder::default()
        .manage(state)
        .setup(move |app| {
            spawn_idle_unloader(shared.clone(), app.handle().clone());
            // Headless GUI testing: auto-click Go on launch (pairs with HEARSAY_REPLAY_WAV).
            if std::env::var("HEARSAY_AUTOGO").is_ok() {
                let (sh, ah) = (shared.clone(), app.handle().clone());
                thread::spawn(move || {
                    thread::sleep(Duration::from_millis(800));
                    begin(ah, sh);
                });
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                // Free models on close (Sidecar::drop kills the processes).
                let state = window.state::<AppState>();
                stop_engine(&state.0);
                state.0.lock().unwrap().sidecars = None;
            }
        })
        .invoke_handler(tauri::generate_handler![go, stop])
        .run(tauri::generate_context!())
        .expect("error while running Hearsay");
}

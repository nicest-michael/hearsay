# Hearsay

A **local, on-device voice agent** for macOS (Apple Silicon). Click **Go**, talk, and
have a spoken back-and-forth with an AI that you can **interrupt mid-sentence** — it
stops talking and listens, like a real conversation. Nothing leaves your Mac.

It hears you (Whisper), thinks (a local Qwen), and says something back (a local neural
voice) — *hear / say*.

```
  mic ─▶ VAD ─▶ Whisper ─▶ │ Dialog FSM │ ─▶ Qwen (mlx) ─▶ sentence chunks ─▶ TTS ─▶ speaker
         └───── barge-in onset ──▶ │ (turn-ids) │ ◀── cancel LLM+TTS+flush playback ─┘
```

## The voice: Kokoro

The project was *named* for talking in the **MisoTTS 8B** "Miso" emotive voice, but we
measured it honestly on this hardware (M5 Pro): MisoTTS 8B runs at **~10× slower than
real-time** on Apple MPS (its 32-deep sequential RVQ decode per frame is kernel-launch-
bound on Metal) and is a **32 GB** download. A reply would lag ~10× its length — that
isn't a conversation — so Miso was evaluated and **dropped** (`docs/adr/0001-architecture.md`,
`docs/spike-evidence/`).

The voice is **Kokoro 82M** via `mlx-audio` — verified **~0.05× RTF** (20× *faster* than
real-time), which is what makes the conversation real-time and interruptible. It lives
behind a `SpeechSynthesizer` port, so if Miso Labs ever ships their promised hosted API
(110 ms), it drops in there with no other changes.

## Architecture (hexagonal — forked from `earshot`)

A Rust workspace + a Tauri 2 desktop shell. The dependency rule points inward.

- **`crates/core`** — pure domain, zero I/O deps: the VAD, sliding-window aggregator and
  Whisper stabilizer (inherited from earshot), the `dialogue` types + low-latency
  sentence chunker, the pure **`conversation::Dialog` FSM** (Idle/Listening/Thinking/
  Speaking with monotonic turn-ids for race-free barge-in), and the **ports**
  (`AudioSource`, `Transcriber`, `LlmClient`, `SpeechSynthesizer`, `AudioPlayer`).
- **`crates/adapters`** — the only place external SDKs appear: the audio I/O (a macOS
  `VoiceProcessingIO` duplex unit for hardware AEC by default, or separate cpal mic +
  interruptible cpal playback), `whisper-rs` (Metal), the `mlx_lm.server` SSE client, the
  TTS Unix-socket client, child-process sidecar lifecycle, and a WAV `FileAudioSource`
  for headless tests.
- **`crates/engine`** — the conversation runtime: five threads (capture, inference, llm,
  tts, controller) around the FSM, with the barge-in cancellation wiring.
- **`src-tauri`** — the desktop shell: Go/Stop, the sidecar lifecycle, and the
  camelCase IPC seam (domain types never cross the wire).
- **`sidecars/`** — Python model servers (`kokoro_server.py`, `llm_server.py`) speaking
  a small Unix-socket protocol (`sidecars/protocol.md`).

## Memory: models never stay loaded

The LLM and TTS run as **child processes the app owns**. **Stop** keeps them warm for an
instant resume; **closing the window** or **5 minutes idle** frees the model RAM. Backstops
for crashes: the Python sidecars self-exit when the parent dies (`os.getppid()` watchdog),
and the app sweeps stale sidecars on startup.

## Run it

Prereqs: macOS on Apple Silicon, Rust 1.94, Node, and [`uv`](https://astral.sh/uv).

```bash
make setup     # installs web deps + the Python sidecar venv (mlx-audio, mlx-lm, misaki)
make dev       # opens the Hearsay window (Vite + Tauri, hot reload)
```

Click **Go**, grant the microphone prompt, **relaunch** (macOS TCC), click **Go** again, and
talk — **open-air, no headphones needed**. The mic and speaker share one macOS
**VoiceProcessingIO** unit, so hardware echo cancellation keeps the mic from hearing the
agent's own voice and barge-in works over the speakers. (`HEARSAY_NO_AEC=1` falls back to
a plain mic + headphones.)

First run downloads the Whisper model (~140 MB), Qwen2.5-3B-Instruct-4bit (~1.8 GB), and
Kokoro (~330 MB).

```bash
make app       # signed release bundle -> target/release/bundle/macos/Hearsay.app
make test      # core + adapter + engine tests
make lint      # clippy -D warnings
```

The bundled `.app` shells out to `uv`/the sidecar venv, so it expects the repo present at
`$HEARSAY_ROOT` (default `~/Repos/hearsay`). This is a personal tool, not a notarized
distributable — see `DEPLOY.md`.

## Verifying without a microphone

`cargo run -p hearsay-engine --example e2e -- /tmp/utter.wav` replays a recorded
utterance through the *real* whisper + Qwen + Kokoro pipeline and saves the agent's spoken
reply to `/tmp/hearsay_reply.wav` — an end-to-end proof with no human in the loop.

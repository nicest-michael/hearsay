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

## The voice: Kokoro by default, Miso optionally

This project was built to talk in the **MisoTTS 8B** "Miso" emotive voice. We measured it
honestly on this hardware (M5 Pro): MisoTTS 8B runs at **~10× slower than real-time** on
Apple MPS — its 32-deep sequential RVQ decode per audio frame is kernel-launch-bound on
Metal (see `docs/adr/0001-architecture.md` and `docs/spike-evidence/`). A reply would lag
~10× its length, which isn't a conversation.

So Hearsay ships **two voices behind one `SpeechSynthesizer` port**:

- **Kokoro 82M** (default) — verified **~0.05× RTF** (20× *faster* than real-time) via
  `mlx-audio`. This is what makes the conversation real-time and interruptible.
- **MisoTTS 8B** (optional) — the real emotive Miso voice, available but **not
  real-time**. Pick it when you want to hear Miso speak and don't mind waiting.

If Miso Labs ships their promised hosted API (110 ms) or a quantized/MLX build, it drops
in behind the same port with no other changes.

## Architecture (hexagonal — forked from `earshot`)

A Rust workspace + a Tauri 2 desktop shell. The dependency rule points inward.

- **`crates/core`** — pure domain, zero I/O deps: the VAD, sliding-window aggregator and
  Whisper stabilizer (inherited from earshot), the `dialogue` types + low-latency
  sentence chunker, the pure **`conversation::Dialog` FSM** (Idle/Listening/Thinking/
  Speaking with monotonic turn-ids for race-free barge-in), and the **ports**
  (`AudioSource`, `Transcriber`, `LlmClient`, `SpeechSynthesizer`, `AudioPlayer`).
- **`crates/adapters`** — the only place external SDKs appear: cpal mic + interruptible
  cpal playback, `whisper-rs` (Metal), the `mlx_lm.server` SSE client, the TTS
  Unix-socket client, child-process sidecar lifecycle, and a WAV `FileAudioSource` for
  headless tests.
- **`crates/engine`** — the conversation runtime: five threads (capture, inference, llm,
  tts, controller) around the FSM, with the barge-in cancellation wiring.
- **`src-tauri`** — the desktop shell: Go/Stop, the sidecar lifecycle, and the
  camelCase IPC seam (domain types never cross the wire).
- **`sidecars/`** — Python model servers (`kokoro_server.py`, `miso_server.py`) speaking
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
talk. **Wear headphones** so the mic doesn't hear the speakers (v1 has no acoustic echo
cancellation — barge-in onset is plain VAD; headphones keep it from hearing itself).

First run downloads the Whisper model (~140 MB), Qwen2.5-3B-Instruct-4bit (~1.8 GB), and
Kokoro (~330 MB). The Miso voice additionally pulls the 32 GB MisoTTS weights on first use.

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

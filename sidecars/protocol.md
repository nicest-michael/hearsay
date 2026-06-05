# Hearsay TTS sidecar protocol

A Hearsay TTS sidecar is a long-lived process that loads a speech model **once** (warm),
then serves synthesis over a **Unix domain socket** (path passed as argv[1]). The Rust
`SidecarTts` adapter is the client. The same protocol is implemented by both backends:

- `kokoro_server.py` — Kokoro 82M via mlx-audio (default, real-time, RTF ~0.05).
- `miso_server.py`   — MisoTTS 8B via PyTorch/MPS (optional, "highest quality, NOT real-time").

## Framing

Control messages are **newline-delimited JSON** (UTF-8). Audio is **length-prefixed
binary**: an `audio` JSON header line, immediately followed by exactly `bytes` bytes of
**little-endian float32 mono PCM** at the model's `sample_rate`.

## Server → client

- `{"type":"ready","sample_rate":24000}`  — sent once, **after the model is loaded and
  warmed** (first synth compiles kernels). The client must wait for this before sending.
- `{"type":"audio","turn":N,"bytes":B}\n` + `B` raw bytes — a chunk of synthesized PCM
  for turn `N` (one per sentence/segment).
- `{"type":"done","turn":N}` — turn `N` fully synthesized (or cancelled).
- `{"type":"error","turn":N,"message":"..."}` — synthesis failed for turn `N`.

## Client → server

- `{"type":"speak","turn":N,"text":"..."}` — synthesize `text` for turn `N`. The server
  splits into sentences and streams `audio` chunks, then `done`.
- `{"type":"cancel","turn":N}` — stop emitting for any turn `<= N` ASAP (checked between
  sentences). Playback is flushed client-side for the instant audible cut; this frees the
  synthesizer for the next turn.
- `{"type":"shutdown"}` or EOF — unload and exit (frees model RAM).

## Lifecycle / safety

- The server runs a **parent-death watchdog**: it polls `os.getppid()` and `os._exit(0)`
  if the parent (the Tauri app) dies, so a crash can never leak the model process.
- Exactly one client connection is served; on disconnect the server exits.

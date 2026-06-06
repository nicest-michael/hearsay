# Deploying Hearsay

Hearsay is a **personal, local** macOS app (Apple Silicon), not a notarized
distributable. "Deploy" = build the signed `.app` and confirm it launches and holds a
conversation. The `.app` shells out to the repo's Python sidecar venv, so it is bound to
this machine + repo, not portable.

## Prerequisites (one time)

- macOS 13+ on Apple Silicon, Rust 1.94 (`rustup`), Node, and [`uv`](https://astral.sh/uv).
- The sidecar Python env and web deps:

  ```bash
  make setup
  ```

  This runs `npm install` in `web/` and creates `sidecars/.venv` with `mlx-audio`,
  `mlx-lm`, `misaki[en]`, `en_core_web_sm`, and `soundfile` (the Kokoro voice + the
  Qwen LLM). `en_core_web_sm` is required at runtime by Kokoro's text processing — without
  it the TTS sidecar crashes trying to download it.

## Build

```bash
make app
# -> target/release/bundle/macos/Hearsay.app  (~6 min cold; web is built first)
```

The bundle is **ad-hoc signed** (no signing identity is set in `tauri.conf.json`). That is
fine for personal use; macOS will re-prompt for Microphone access after a rebuild because
the designated requirement changes. For a stable TCC grant across rebuilds, create a
persistent self-signed identity and set `bundle.macOS.signingIdentity` in
`src-tauri/tauri.conf.json` (same approach as the sibling `earshot` project).

## Run

```bash
open target/release/bundle/macos/Hearsay.app
```

- The app resolves the sidecar scripts + venv from `$HEARSAY_ROOT` (default
  `~/Repos/hearsay`). If the repo lives elsewhere, launch with that set:
  `HEARSAY_ROOT=/path/to/hearsay open -a target/release/bundle/macos/Hearsay.app`.
- Click **Go**, grant the Microphone prompt, **quit and relaunch** (macOS TCC only applies
  the grant on next launch), click **Go**, and talk.
- **Wear headphones.** v1 has no acoustic echo cancellation; the barge-in trigger is plain
  VAD, so open speakers make the agent hear itself.
- First run downloads the models: Whisper base.en (~140 MB) to
  `~/Library/Application Support/ai.nicest.hearsay/models`, Qwen2.5-3B-Instruct-4bit
  (~1.8 GB) and Kokoro (~330 MB) to the HF cache. The Miso voice pulls 32 GB on first use.

## Verify (no microphone needed)

```bash
say --data-format=LEI16@16000 -o /tmp/utter.wav "Hey, can you hear me? What should I call you?"
cargo run -p hearsay-engine --example e2e -- /tmp/utter.wav
```

Expected: it transcribes the utterance, prints a `HEARSAY:` reply, writes
`/tmp/hearsay_reply.wav` (the spoken reply), and prints `VERDICT : PASS`.

## Confirm models don't linger

After quitting the app (or 5 minutes idle), no sidecars should remain:

```bash
pgrep -fl "kokoro_server|llm_server|mlx_lm.server"   # -> nothing
```

If a hard crash ever leaks one, the next **Go** sweeps it (`sweep_stale`), and each sidecar
also self-exits when the app process dies.

## Logs

Sidecar stdout/stderr go to `/tmp/hearsay-{kokoro,llm}.log`. The app logs to the
console it was launched from (`RUST_LOG=debug` for more).

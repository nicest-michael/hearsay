# Hearsay — build & run. Run the Tauri CLI from the repo root.

WEB := ./web/node_modules/.bin
TAURI := $(WEB)/tauri

.PHONY: setup dev app test lint clean sidecars

## Install frontend deps + the Python sidecar venv (Kokoro TTS + Qwen LLM).
setup:
	cd web && npm install
	cd sidecars && uv venv --python 3.12 .venv
	cd sidecars && uv pip install --python .venv mlx-audio mlx-lm "misaki[en]" soundfile
	# Kokoro's text processing (misaki -> spaCy) needs this model AT RUNTIME, or the TTS
	# sidecar crashes trying to download it (no pip/uv when launched from the .app).
	cd sidecars && uv pip install --python .venv "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

## Vite + Tauri dev window (hot reload).
dev:
	$(TAURI) dev

## Signed release bundle -> target/release/bundle/macos/Hearsay.app
app:
	$(TAURI) build

## Domain + adapter + engine tests (excludes the Tauri shell).
test:
	cargo test

## Clippy across the whole workspace, warnings as errors.
lint:
	cargo clippy --workspace --all-targets -- -D warnings

clean:
	cargo clean
	rm -rf web/dist web/node_modules

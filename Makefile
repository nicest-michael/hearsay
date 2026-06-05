# Hearsay — build & run. Run the Tauri CLI from the repo root.

WEB := ./web/node_modules/.bin
TAURI := $(WEB)/tauri

.PHONY: setup dev app test lint clean sidecars

## Install frontend deps + the Python sidecar venvs.
setup:
	cd web && npm install
	cd sidecars && uv venv --python 3.12 .venv && uv pip install --python .venv mlx-audio mlx-lm "misaki[en]" soundfile

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

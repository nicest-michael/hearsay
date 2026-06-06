//! `SpeechSynthesizer` over the Hearsay Kokoro TTS sidecar speaking the
//! Unix-domain-socket protocol in `sidecars/protocol.md`.
//!
//! The adapter owns one connection. `speak` sends a `speak` request and streams the
//! returned f32 PCM to `on_pcm`. When `cancel` flips (barge-in), it sends a `cancel`
//! and stops forwarding audio (still draining to the terminating `done`, so the
//! connection stays in sync for the next turn). Playback is flushed engine-side, so the
//! audible cut is instant regardless of how long the drain takes.

use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use hearsay_core::error::SynthError;
use hearsay_core::ports::SpeechSynthesizer;
use serde_json::{json, Value};

/// A silent sidecar for this long is treated as dead, so a hung TTS process can't
/// block the worker thread (and thus shutdown `join()`) forever.
const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// Reject an `audio` header claiming more than this (guards against a bad header
/// triggering a huge allocation + indefinite read). 30 s of 24 kHz f32 mono.
const MAX_PCM_BYTES: usize = 24_000 * 4 * 30;

fn is_timeout(e: &std::io::Error) -> bool {
    matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut)
}

pub struct SidecarTts {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
    sample_rate: u32,
    next_id: u64,
}

impl SidecarTts {
    /// Connect to a sidecar socket and wait for its `ready` (model loaded + warmed).
    pub fn connect(sock_path: &str) -> Result<Self, SynthError> {
        let stream =
            UnixStream::connect(sock_path).map_err(|e| SynthError::Unreachable(e.to_string()))?;
        stream
            .set_read_timeout(Some(READ_TIMEOUT))
            .map_err(|e| SynthError::Unreachable(e.to_string()))?;
        let writer = stream
            .try_clone()
            .map_err(|e| SynthError::Unreachable(e.to_string()))?;
        let mut reader = BufReader::new(stream);

        let mut line = String::new();
        reader
            .read_line(&mut line)
            .map_err(|e| SynthError::Protocol(e.to_string()))?;
        let v: Value = serde_json::from_str(line.trim())
            .map_err(|e| SynthError::Protocol(format!("bad ready line {line:?}: {e}")))?;
        if v["type"] != "ready" {
            return Err(SynthError::Protocol(format!("expected ready, got {line:?}")));
        }
        let sample_rate = v["sample_rate"].as_u64().unwrap_or(24_000) as u32;
        Ok(Self {
            reader,
            writer,
            sample_rate,
            next_id: 0,
        })
    }

    fn send(&mut self, obj: &Value) -> Result<(), SynthError> {
        let mut bytes = serde_json::to_vec(obj).map_err(|e| SynthError::Protocol(e.to_string()))?;
        bytes.push(b'\n');
        self.writer
            .write_all(&bytes)
            .and_then(|_| self.writer.flush())
            .map_err(|e| SynthError::Unreachable(e.to_string()))
    }
}

/// Decode little-endian f32 PCM bytes into samples.
fn decode_pcm(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl SpeechSynthesizer for SidecarTts {
    fn speak(
        &mut self,
        text: &str,
        cancel: &AtomicBool,
        on_pcm: &mut dyn FnMut(&[f32]),
    ) -> Result<(), SynthError> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"type": "speak", "turn": id, "text": text}))?;

        let mut cancelled = false;
        let mut line = String::new();
        loop {
            line.clear();
            let n = match self.reader.read_line(&mut line) {
                Ok(n) => n,
                // A read timeout while cancelled is a clean bail; otherwise the sidecar
                // is hung — fail rather than block the worker (and shutdown) forever.
                Err(e) if is_timeout(&e) => {
                    return if cancel.load(Ordering::Relaxed) {
                        Ok(())
                    } else {
                        Err(SynthError::Unreachable("tts read timed out".into()))
                    };
                }
                Err(e) => return Err(SynthError::Protocol(e.to_string())),
            };
            if n == 0 {
                return Err(SynthError::Unreachable("sidecar closed connection".into()));
            }
            let v: Value = serde_json::from_str(line.trim())
                .map_err(|e| SynthError::Protocol(format!("bad line {line:?}: {e}")))?;
            match v["type"].as_str() {
                Some("audio") => {
                    let len = v["bytes"].as_u64().unwrap_or(0) as usize;
                    if len > MAX_PCM_BYTES {
                        return Err(SynthError::Protocol(format!("audio chunk too large: {len}")));
                    }
                    let mut buf = vec![0u8; len];
                    if let Err(e) = self.reader.read_exact(&mut buf) {
                        return if is_timeout(&e) && cancel.load(Ordering::Relaxed) {
                            Ok(())
                        } else {
                            Err(SynthError::Protocol(e.to_string()))
                        };
                    }
                    if cancelled {
                        continue; // draining post-cancel
                    }
                    if cancel.load(Ordering::Relaxed) {
                        cancelled = true;
                        self.send(&json!({"type": "cancel", "turn": id}))?;
                    } else {
                        on_pcm(&decode_pcm(&buf));
                    }
                }
                Some("done") => break,
                Some("error") => {
                    return Err(SynthError::Protocol(
                        v["message"].as_str().unwrap_or("tts error").to_string(),
                    ))
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_le_f32_pcm() {
        let samples = [0.0f32, 1.0, -0.5, 0.25];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        assert_eq!(decode_pcm(&bytes), samples);
    }

    #[test]
    fn decode_ignores_trailing_partial_sample() {
        let mut bytes = 1.0f32.to_le_bytes().to_vec();
        bytes.push(0xAB); // a stray byte
        assert_eq!(decode_pcm(&bytes), vec![1.0]);
    }

    /// End-to-end against the real Kokoro sidecar (spawns Python). Manual.
    #[test]
    #[ignore = "spawns the Kokoro sidecar; run with --ignored --nocapture"]
    fn speaks_via_real_sidecar() {
        use std::process::Command;
        use std::sync::atomic::AtomicBool;
        let root = env!("CARGO_MANIFEST_DIR").to_string() + "/../..";
        let sock = "/tmp/hearsay_tts_rusttest.sock";
        let _ = std::fs::remove_file(sock);
        let mut child = Command::new(format!("{root}/sidecars/.venv/bin/python"))
            .arg(format!("{root}/sidecars/kokoro_server.py"))
            .arg(sock)
            .spawn()
            .expect("spawn sidecar");
        // wait for warmup + bind
        let mut tts = None;
        for _ in 0..120 {
            std::thread::sleep(std::time::Duration::from_millis(500));
            if let Ok(t) = SidecarTts::connect(sock) {
                tts = Some(t);
                break;
            }
        }
        let mut tts = tts.expect("connect to sidecar");
        assert_eq!(tts.sample_rate(), 24_000);
        let mut total = 0usize;
        let cancel = AtomicBool::new(false);
        tts.speak("Hello from Hearsay.", &cancel, &mut |pcm| total += pcm.len())
            .expect("speak");
        assert!(total > 10_000, "expected speech PCM, got {total} samples");
        let _ = child.kill();
        let _ = child.wait(); // reap
    }
}

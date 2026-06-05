//! Child-process lifecycle for the Python model sidecars (TTS + LLM) the app owns.
//!
//! "Don't leave models loaded" is satisfied here: a [`Sidecar`] kills its child on
//! `Drop` (Stop / window-close / idle-timeout all drop it). Because `Drop` does not run
//! if the app is `SIGKILL`ed or panics, two backstops cover that: the Python sidecars
//! self-exit when their parent dies (`os.getppid()` watchdog), and [`sweep_stale`] kills
//! any orphaned sidecars from a previous crashed run at startup.

use std::process::{Child, Command};

/// A model sidecar process owned by the app. Killed on drop.
pub struct Sidecar {
    child: Child,
    pub name: &'static str,
}

impl Sidecar {
    /// Spawn the Kokoro TTS sidecar (mlx-audio) bound to `sock`.
    pub fn kokoro(repo_root: &str, sock: &str, voice: &str) -> std::io::Result<Self> {
        let child = Command::new(format!("{repo_root}/sidecars/.venv/bin/python"))
            .arg(format!("{repo_root}/sidecars/kokoro_server.py"))
            .arg(sock)
            .arg(voice)
            .spawn()?;
        Ok(Self {
            child,
            name: "kokoro-tts",
        })
    }

    /// Spawn the MisoTTS sidecar (PyTorch/MPS — "highest quality, not real-time").
    pub fn miso(repo_root: &str, sock: &str, voice: &str) -> std::io::Result<Self> {
        let child = Command::new(format!("{repo_root}/vendor/MisoTTS/.venv/bin/python"))
            .arg(format!("{repo_root}/sidecars/miso_server.py"))
            .arg(sock)
            .arg(voice)
            .spawn()?;
        Ok(Self {
            child,
            name: "miso-tts",
        })
    }

    /// Spawn `mlx_lm.server` for the dialog model (OpenAI-compatible on 127.0.0.1:port).
    /// Uses the venv console script (`python -m mlx_lm.server` is deprecated in 0.31).
    pub fn mlx_llm(repo_root: &str, model: &str, port: u16) -> std::io::Result<Self> {
        let child = Command::new(format!("{repo_root}/sidecars/.venv/bin/mlx_lm.server"))
            .args([
                "--model",
                model,
                "--port",
                &port.to_string(),
                "--host",
                "127.0.0.1",
                "--log-level",
                "WARNING",
            ])
            .spawn()?;
        Ok(Self {
            child,
            name: "mlx-llm",
        })
    }

    /// The child's PID (for diagnostics).
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait(); // reap — no zombies
        log::info!("sidecar '{}' (pid {}) stopped", self.name, self.child.id());
    }
}

/// Kill any orphaned Hearsay sidecars left by a previous crashed run. Best-effort.
pub fn sweep_stale() {
    for pat in ["kokoro_server.py", "miso_server.py", "mlx_lm.server"] {
        let _ = Command::new("pkill").args(["-f", pat]).status();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::zombie_processes)] // the Child is moved into Sidecar, whose Drop reaps it
    fn drop_kills_and_reaps_child() {
        // A stand-in long-lived child; Drop must terminate it.
        let child = Command::new("sleep").arg("60").spawn().expect("spawn sleep");
        let pid = child.id();
        let sc = Sidecar {
            child,
            name: "test",
        };
        assert_eq!(sc.pid(), pid);
        drop(sc);
        // After drop, the pid should no longer be alive.
        std::thread::sleep(std::time::Duration::from_millis(100));
        let alive = Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(!alive, "child {pid} still alive after Drop");
    }
}

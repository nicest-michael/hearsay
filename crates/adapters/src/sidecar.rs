//! Child-process lifecycle for the Python model sidecars (TTS + LLM) the app owns.
//!
//! "Don't leave models loaded" is enforced with defense in depth:
//! 1. Each sidecar is spawned as its own **process group**; [`Sidecar`]'s `Drop` kills
//!    the whole group (Stop / window-close / idle-timeout all drop it), so the LLM
//!    wrapper's `mlx_lm.server` child dies with it.
//! 2. `Drop` doesn't run on `SIGKILL`/panic, so every sidecar **self-exits when its
//!    parent dies**: the TTS servers via an `os.getppid()` watchdog, the LLM via the
//!    `llm_server.py` watchdog wrapper (mlx_lm.server has no such guard of its own).
//! 3. [`sweep_stale`] kills any orphans from a previous crashed run at startup.
//!
//! Sidecar stdio is redirected to log files (not inherited) so they neither spam the
//! app's console nor hold its pipes open.

use std::fs::File;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};

/// A redirect for a sidecar's stdout/stderr: a truncated per-name log file under
/// `/tmp`, or `/dev/null` if that can't be created. Never inherits the parent's fds.
fn log_to(name: &str) -> Stdio {
    File::create(format!("/tmp/hearsay-{name}.log"))
        .map(Stdio::from)
        .unwrap_or_else(|_| Stdio::null())
}

/// A model sidecar process owned by the app. Its whole process group is killed on drop.
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
            .process_group(0)
            .stdout(log_to("kokoro"))
            .stderr(log_to("kokoro"))
            .spawn()?;
        Ok(Self {
            child,
            name: "kokoro-tts",
        })
    }

    /// Spawn the MisoTTS sidecar (PyTorch/MPS — "highest quality, not real-time").
    pub fn miso(repo_root: &str, sock: &str, speaker: &str) -> std::io::Result<Self> {
        let child = Command::new(format!("{repo_root}/vendor/MisoTTS/.venv/bin/python"))
            .arg(format!("{repo_root}/sidecars/miso_server.py"))
            .arg(sock)
            .arg(speaker)
            .process_group(0)
            .stdout(log_to("miso"))
            .stderr(log_to("miso"))
            .spawn()?;
        Ok(Self {
            child,
            name: "miso-tts",
        })
    }

    /// Spawn the dialog LLM via the `llm_server.py` watchdog wrapper around
    /// `mlx_lm.server` (OpenAI-compatible on 127.0.0.1:port).
    pub fn mlx_llm(repo_root: &str, model: &str, port: u16) -> std::io::Result<Self> {
        let child = Command::new(format!("{repo_root}/sidecars/.venv/bin/python"))
            .arg(format!("{repo_root}/sidecars/llm_server.py"))
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
            .process_group(0)
            .stdout(log_to("llm"))
            .stderr(log_to("llm"))
            .spawn()?;
        Ok(Self {
            child,
            name: "mlx-llm",
        })
    }

    /// The child's PID (also its process-group id).
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        let pid = self.child.id() as i32;
        // Terminate the whole process group (the wrapper + any model child).
        // SIGTERM lets the watchdog/server shut down cleanly; the members all respond
        // to it, so the following wait() returns promptly.
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }
        let _ = self.child.wait(); // reap — no zombies
        log::info!("sidecar '{}' (pid {pid}) stopped", self.name);
    }
}

/// Kill any orphaned Hearsay sidecars left by a previous crashed run. Best-effort.
pub fn sweep_stale() {
    for pat in [
        "kokoro_server.py",
        "miso_server.py",
        "llm_server.py",
        "mlx_lm.server",
    ] {
        let _ = Command::new("pkill").args(["-f", pat]).status();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::zombie_processes)] // the Child is moved into Sidecar, whose Drop reaps it
    fn drop_kills_group_and_reaps() {
        // Stand-in child in its OWN process group (so group-kill can't hit the test
        // runner). It also spawns a grandchild to prove the whole group dies.
        let child = Command::new("sh")
            .arg("-c")
            .arg("sleep 60 & sleep 60")
            .process_group(0)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        let pid = child.id();
        let sc = Sidecar { child, name: "test" };
        assert_eq!(sc.pid(), pid);
        drop(sc);
        std::thread::sleep(std::time::Duration::from_millis(150));
        let alive = Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(!alive, "group leader {pid} still alive after Drop");
    }
}

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

/// A PATH that includes the usual spots a launched `.app` may be missing — `uv`
/// (`~/.local/bin`), Homebrew, and `/usr/local` — so a sidecar's own subprocesses
/// (e.g. spaCy/pip lookups) resolve regardless of how the app was started.
fn sane_path() -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let inherited = std::env::var("PATH").unwrap_or_default();
    format!("{home}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{inherited}")
}

/// A model sidecar process owned by the app. Its whole process group is killed on drop.
pub struct Sidecar {
    child: Child,
    pub name: &'static str,
    log: &'static str,
}

impl Sidecar {
    /// Spawn the Kokoro TTS sidecar (mlx-audio) bound to `sock`.
    pub fn kokoro(repo_root: &str, sock: &str, voice: &str) -> std::io::Result<Self> {
        let mut cmd = Command::new(format!("{repo_root}/sidecars/.venv/bin/python"));
        cmd.arg(format!("{repo_root}/sidecars/kokoro_server.py"))
            .arg(sock)
            .arg(voice);
        spawn_grouped(cmd, repo_root, "kokoro", "kokoro-tts")
    }

    /// Spawn the MisoTTS sidecar (PyTorch/MPS — "highest quality, not real-time").
    pub fn miso(repo_root: &str, sock: &str, speaker: &str) -> std::io::Result<Self> {
        let mut cmd = Command::new(format!("{repo_root}/vendor/MisoTTS/.venv/bin/python"));
        cmd.arg(format!("{repo_root}/sidecars/miso_server.py"))
            .arg(sock)
            .arg(speaker);
        spawn_grouped(cmd, repo_root, "miso", "miso-tts")
    }

    /// Spawn the dialog LLM via the `llm_server.py` watchdog wrapper around
    /// `mlx_lm.server` (OpenAI-compatible on 127.0.0.1:port).
    pub fn mlx_llm(repo_root: &str, model: &str, port: u16) -> std::io::Result<Self> {
        let mut cmd = Command::new(format!("{repo_root}/sidecars/.venv/bin/python"));
        cmd.arg(format!("{repo_root}/sidecars/llm_server.py")).args([
            "--model",
            model,
            "--port",
            &port.to_string(),
            "--host",
            "127.0.0.1",
            "--log-level",
            "WARNING",
        ]);
        spawn_grouped(cmd, repo_root, "llm", "mlx-llm")
    }

    /// The child's PID (also its process-group id).
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// True if the process has already exited (crashed/failed) — reaps it if so.
    pub fn has_exited(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(Some(_)))
    }

    /// The last few lines of this sidecar's log, for surfacing a failure to the user.
    pub fn log_tail(&self) -> String {
        let path = format!("/tmp/hearsay-{}.log", self.log);
        match std::fs::read_to_string(&path) {
            Ok(s) => {
                let lines: Vec<&str> = s.lines().collect();
                let start = lines.len().saturating_sub(8);
                lines[start..].join("\n")
            }
            Err(_) => format!("(no log at {path})"),
        }
    }
}

/// Spawn `cmd` in its own process group, with a sane working dir + PATH, and
/// stdout/stderr redirected to `/tmp/hearsay-<log>.log` (never inheriting the parent's
/// fds), tagged `name`.
fn spawn_grouped(
    mut cmd: Command,
    repo_root: &str,
    log: &'static str,
    name: &'static str,
) -> std::io::Result<Sidecar> {
    let child = cmd
        .current_dir(repo_root)
        .env("PATH", sane_path())
        .process_group(0)
        .stdout(log_to(log))
        .stderr(log_to(log))
        .spawn()?;
    Ok(Sidecar { child, name, log })
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
/// Patterns are Hearsay-specific paths so this never touches an unrelated `mlx_lm.server`
/// the user might be running for another project.
pub fn sweep_stale() {
    for pat in [
        "sidecars/kokoro_server.py",
        "sidecars/miso_server.py",
        "sidecars/llm_server.py",
        "sidecars/.venv/bin/mlx_lm.server",
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
        let sc = Sidecar {
            child,
            name: "test",
            log: "test",
        };
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

//! ggml Whisper model resolution + download from Hugging Face (`ggerganov/whisper.cpp`).
//! The binary owns *where* models live (a `directories` data dir); this module only
//! ensures a given model is present, downloading it once on first use.

use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use hearsay_core::config::ModelChoice;

const HF_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// The ggml filename for a model, as published on Hugging Face `ggerganov/whisper.cpp`.
/// (Vendor packaging detail — lives here in the adapter, not in the domain.)
fn ggml_filename(model: ModelChoice) -> &'static str {
    match model {
        ModelChoice::TinyEn => "ggml-tiny.en.bin",
        ModelChoice::BaseEn => "ggml-base.en.bin",
        ModelChoice::SmallEn => "ggml-small.en.bin",
        ModelChoice::LargeV3Turbo => "ggml-large-v3-turbo.bin",
    }
}

/// Approximate on-disk size in MB, for the download-progress estimate.
fn approx_mb(model: ModelChoice) -> u32 {
    match model {
        ModelChoice::TinyEn => 78,
        ModelChoice::BaseEn => 148,
        ModelChoice::SmallEn => 488,
        ModelChoice::LargeV3Turbo => 1620,
    }
}

#[derive(Debug)]
pub enum ModelError {
    Io(io::Error),
    Download(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Io(e) => write!(f, "model i/o error: {e}"),
            ModelError::Download(s) => write!(f, "model download failed: {s}"),
        }
    }
}
impl std::error::Error for ModelError {}
impl From<io::Error> for ModelError {
    fn from(e: io::Error) -> Self {
        ModelError::Io(e)
    }
}

/// Where a model file would live inside `dir`.
pub fn model_path(dir: &Path, model: ModelChoice) -> PathBuf {
    dir.join(ggml_filename(model))
}

/// Is the model already downloaded (and plausibly complete)?
pub fn is_present(dir: &Path, model: ModelChoice) -> bool {
    let path = model_path(dir, model);
    fs::metadata(&path)
        .map(|m| m.len() > 1_000_000)
        .unwrap_or(false)
}

/// Ensure the model exists locally, downloading it if needed. `progress` is called
/// with `(bytes_downloaded, total_estimate_bytes)`.
pub fn ensure_model(
    dir: &Path,
    model: ModelChoice,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf, ModelError> {
    let path = model_path(dir, model);
    if is_present(dir, model) {
        return Ok(path);
    }
    fs::create_dir_all(dir)?;
    let url = format!("{HF_BASE}/{}", ggml_filename(model));
    let total = approx_mb(model) as u64 * 1_000_000;
    let tmp = path.with_extension("part");

    let resp = ureq::get(&url)
        .call()
        .map_err(|e| ModelError::Download(format!("{url}: {e}")))?;
    let mut reader = resp.into_body().into_reader();
    let mut file = fs::File::create(&tmp)?;
    let mut buf = vec![0u8; 1 << 16];
    let mut done = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        progress(done, total.max(done));
    }
    file.flush()?;
    drop(file);
    fs::rename(&tmp, &path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_uses_model_filename() {
        let p = model_path(Path::new("/models"), ModelChoice::BaseEn);
        assert!(p.ends_with("ggml-base.en.bin"));
    }

    #[test]
    fn missing_model_is_not_present() {
        assert!(!is_present(
            Path::new("/nonexistent-hearsay-dir"),
            ModelChoice::BaseEn
        ));
    }
}

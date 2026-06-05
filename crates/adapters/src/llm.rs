//! `LlmClient` over an OpenAI-compatible local server (`mlx_lm.server`), streaming
//! Server-Sent Events. `reply` POSTs the conversation with `stream: true`, forwards
//! each `choices[0].delta.content` token to `on_delta`, and aborts (drops the
//! connection, which stops server-side generation) as soon as `cancel` flips.

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};

use hearsay_core::dialogue::{Role, Turn};
use hearsay_core::error::LlmError;
use hearsay_core::ports::LlmClient;
use serde_json::{json, Value};

pub struct MlxChat {
    base: String, // e.g. http://127.0.0.1:8080
    model: String,
    max_tokens: u32,
    temperature: f32,
}

impl MlxChat {
    pub fn new(base: &str, model: &str) -> Self {
        Self {
            base: base.trim_end_matches('/').to_string(),
            model: model.to_string(),
            max_tokens: 200, // terse, spoken replies
            temperature: 0.7,
        }
    }

    pub fn with_sampling(mut self, max_tokens: u32, temperature: f32) -> Self {
        self.max_tokens = max_tokens;
        self.temperature = temperature;
        self
    }

    /// True once the server is up and serving models.
    pub fn health(&self) -> bool {
        ureq::get(format!("{}/v1/models", self.base))
            .call()
            .is_ok()
    }
}

fn role_str(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// Parse one SSE line into a token delta. Returns:
/// - `Ok(Some(tok))` for a content delta,
/// - `Ok(None)` for blanks / non-content events,
/// - `Err(())` for the `[DONE]` sentinel (end of stream).
fn parse_sse_line(line: &str) -> Result<Option<String>, ()> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    let data = line.strip_prefix("data:").map(str::trim).unwrap_or(line);
    if data == "[DONE]" {
        return Err(());
    }
    let v: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    match v["choices"][0]["delta"]["content"].as_str() {
        Some(tok) if !tok.is_empty() => Ok(Some(tok.to_string())),
        _ => Ok(None),
    }
}

impl LlmClient for MlxChat {
    fn reply(
        &self,
        history: &[Turn],
        cancel: &AtomicBool,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String, LlmError> {
        let messages: Vec<Value> = history
            .iter()
            .map(|t| json!({"role": role_str(t.role), "content": t.text}))
            .collect();
        let body = json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        });

        let resp = ureq::post(format!("{}/v1/chat/completions", self.base))
            .send_json(&body)
            .map_err(|e| LlmError::Unreachable(e.to_string()))?;
        let mut reader = BufReader::new(resp.into_body().into_reader());

        let mut full = String::new();
        let mut line = String::new();
        loop {
            if cancel.load(Ordering::Relaxed) {
                // Dropping the reader closes the connection -> server stops generating.
                return Ok(full);
            }
            line.clear();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| LlmError::Protocol(e.to_string()))?;
            if n == 0 {
                break; // stream closed
            }
            match parse_sse_line(&line) {
                Ok(Some(tok)) => {
                    full.push_str(&tok);
                    on_delta(&tok);
                }
                Ok(None) => {}
                Err(()) => break, // [DONE]
            }
        }
        Ok(full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_delta() {
        let l = r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#;
        assert_eq!(parse_sse_line(l), Ok(Some("Hello".to_string())));
    }

    #[test]
    fn done_sentinel_ends_stream() {
        assert_eq!(parse_sse_line("data: [DONE]"), Err(()));
    }

    #[test]
    fn blank_and_roleonly_deltas_are_none() {
        assert_eq!(parse_sse_line(""), Ok(None));
        assert_eq!(
            parse_sse_line(r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#),
            Ok(None)
        );
        assert_eq!(
            parse_sse_line(r#"data: {"choices":[{"delta":{"content":""}}]}"#),
            Ok(None)
        );
    }

    #[test]
    fn tolerates_unparseable_lines() {
        assert_eq!(parse_sse_line("data: not json"), Ok(None));
        assert_eq!(parse_sse_line(": keep-alive comment"), Ok(None));
    }

    #[test]
    fn role_strings() {
        assert_eq!(role_str(Role::System), "system");
        assert_eq!(role_str(Role::User), "user");
        assert_eq!(role_str(Role::Assistant), "assistant");
    }
}

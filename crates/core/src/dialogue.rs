//! Conversation history and the low-latency sentence chunker.
//!
//! [`Conversation`] is the message log the engine maintains and hands to the
//! [`crate::ports::LlmClient`]. [`SentenceChunker`] splits a streaming LLM token feed
//! into speakable chunks so the synthesizer can start on the first phrase instead of
//! waiting for the whole reply — the first chunk may break early at a comma (lower
//! time-to-first-audio), later chunks break on sentence terminators.

/// Who produced a turn.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// One message in the conversation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Turn {
    pub role: Role,
    pub text: String,
}

/// The running conversation: a system turn followed by alternating user/assistant
/// turns. The engine owns one of these per session.
#[derive(Clone, Debug)]
pub struct Conversation {
    turns: Vec<Turn>,
}

impl Conversation {
    /// Start a conversation with a system persona.
    pub fn new(system: &str) -> Self {
        Self {
            turns: vec![Turn {
                role: Role::System,
                text: system.to_string(),
            }],
        }
    }

    pub fn add_user(&mut self, text: &str) {
        self.turns.push(Turn {
            role: Role::User,
            text: text.to_string(),
        });
    }

    pub fn add_assistant(&mut self, text: &str) {
        self.turns.push(Turn {
            role: Role::Assistant,
            text: text.to_string(),
        });
    }

    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// Keep the system turn plus the most recent `n` turns (caps context + RAM).
    pub fn truncate_keep_last(&mut self, n: usize) {
        if self.turns.len() <= 1 + n {
            return;
        }
        let start = self.turns.len() - n;
        let tail: Vec<Turn> = self.turns.drain(start..).collect();
        self.turns.truncate(1); // keep only the system turn
        self.turns.extend(tail);
    }
}

const TERMINATORS: [char; 5] = ['.', '!', '?', '\n', '…'];

/// Splits a streaming token feed into speakable chunks.
///
/// The **first** chunk of a reply may break early at a comma once it is at least
/// `MIN_FIRST` characters long, to minimize time-to-first-audio. After the first
/// chunk, breaks happen only at sentence terminators. Call [`SentenceChunker::flush`]
/// at end-of-stream to emit any trailing partial sentence.
#[derive(Debug, Default)]
pub struct SentenceChunker {
    buf: String,
    first_done: bool,
}

impl SentenceChunker {
    /// Minimum length of the first chunk before an early comma-break is allowed.
    const MIN_FIRST: usize = 12;

    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a token/delta. Returns a speakable chunk if a break point was reached.
    pub fn push(&mut self, delta: &str) -> Option<String> {
        self.buf.push_str(delta);

        let term = self
            .buf
            .char_indices()
            .find(|&(_, c)| TERMINATORS.contains(&c));
        let comma = if self.first_done {
            None
        } else {
            self.buf
                .char_indices()
                .find(|&(i, c)| c == ',' && i >= Self::MIN_FIRST)
        };

        let pick = match (term, comma) {
            (Some(t), Some(c)) => Some(if t.0 <= c.0 { t } else { c }),
            (Some(t), None) => Some(t),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        };

        pick.and_then(|(i, ch)| self.take_through(i + ch.len_utf8()))
    }

    /// Emit any remaining buffered text as a final chunk (end of stream).
    pub fn flush(&mut self) -> Option<String> {
        let s = self.buf.trim().to_string();
        self.buf.clear();
        self.first_done = true;
        (!s.is_empty()).then_some(s)
    }

    fn take_through(&mut self, end: usize) -> Option<String> {
        let chunk: String = self.buf.drain(..end).collect();
        self.first_done = true;
        let s = chunk.trim().to_string();
        (!s.is_empty()).then_some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_emits_on_sentence_end() {
        let mut c = SentenceChunker::new();
        assert_eq!(c.push("Hello"), None);
        assert_eq!(c.push(" there."), Some("Hello there.".to_string()));
        assert_eq!(c.push(" How are"), None);
        assert_eq!(c.push(" you?"), Some("How are you?".to_string()));
        assert_eq!(c.flush(), None);
    }

    #[test]
    fn chunker_flushes_tail_without_terminator() {
        let mut c = SentenceChunker::new();
        c.push("no period here");
        assert_eq!(c.flush(), Some("no period here".to_string()));
    }

    #[test]
    fn first_chunk_breaks_early_at_comma_past_min() {
        let mut c = SentenceChunker::new();
        // first comma (idx 4) is too early; the comma after "sure" (idx 14) qualifies
        assert_eq!(
            c.push("Yeah, for sure, that makes total sense."),
            Some("Yeah, for sure,".to_string())
        );
        // the remainder breaks on the terminator, no longer comma-breaking
        assert_eq!(c.flush(), Some("that makes total sense.".to_string()));
    }

    #[test]
    fn later_chunks_do_not_comma_break() {
        let mut c = SentenceChunker::new();
        assert_eq!(c.push("First one is here."), Some("First one is here.".to_string()));
        // now first_done; a comma alone must NOT trigger a break
        assert_eq!(c.push(" second, with a comma"), None);
        assert_eq!(c.push(" and an end."), Some("second, with a comma and an end.".to_string()));
    }

    #[test]
    fn handles_multibyte_terminator() {
        let mut c = SentenceChunker::new();
        assert_eq!(c.push("well…"), Some("well…".to_string()));
    }

    #[test]
    fn conversation_appends_and_orders() {
        let mut conv = Conversation::new("You are a friendly conversational partner.");
        conv.add_user("hi");
        conv.add_assistant("hey!");
        let t = conv.turns();
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].role, Role::System);
        assert_eq!((t[1].role, t[1].text.as_str()), (Role::User, "hi"));
        assert_eq!((t[2].role, t[2].text.as_str()), (Role::Assistant, "hey!"));
    }

    #[test]
    fn truncate_keeps_system_plus_last_n() {
        let mut conv = Conversation::new("sys");
        for i in 0..10 {
            conv.add_user(&format!("u{i}"));
            conv.add_assistant(&format!("a{i}"));
        }
        conv.truncate_keep_last(4);
        let t = conv.turns();
        assert_eq!(t.len(), 5); // system + last 4
        assert_eq!(t[0].role, Role::System);
        // last 4 of u0,a0,...,u9,a9 are u8,a8,u9,a9
        assert_eq!(t[1].text, "u8");
        assert_eq!(t[4].text, "a9");
    }
}

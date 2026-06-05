//! What flows to the UI: a streaming **delta** event ([`TranscriptUpdate`]) plus a
//! pure accumulator ([`TranscriptView`]) the UI applies it to. Emitting deltas (not
//! the whole transcript) keeps per-step cost proportional to new words, so hour-long
//! sessions stay cheap.

use crate::word::{join_words, Word};

/// A streaming transcript event for one source.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptUpdate {
    /// Short human label for the source, e.g. "Mic" or "System".
    pub source: String,
    /// Words promoted to committed since the last update (final).
    pub newly_committed: Vec<Word>,
    /// The current volatile tail (replaces the previous tail; may change again).
    pub volatile: Vec<Word>,
}

/// The accumulated transcript for one source, built by applying updates in order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptView {
    pub committed: Vec<Word>,
    pub volatile: Vec<Word>,
}

impl TranscriptView {
    /// Fold in one update: append the newly-committed words, replace the tail.
    pub fn apply(&mut self, update: &TranscriptUpdate) {
        self.committed
            .extend(update.newly_committed.iter().cloned());
        self.volatile = update.volatile.clone();
    }

    /// Stable text only.
    pub fn committed_text(&self) -> String {
        join_words(&self.committed)
    }

    /// Committed + volatile tail (what's visually on screen).
    pub fn full_text(&self) -> String {
        let c = self.committed_text();
        let v = join_words(&self.volatile);
        match (c.is_empty(), v.is_empty()) {
            (true, _) => v,
            (false, true) => c,
            (false, false) => format!("{c} {v}"),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.committed.is_empty() && self.volatile.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(t: &str) -> Word {
        Word::new(t, 0, 1)
    }

    #[test]
    fn view_accumulates_committed_and_replaces_volatile() {
        let mut view = TranscriptView::default();
        view.apply(&TranscriptUpdate {
            source: "Mic".into(),
            newly_committed: vec![w("the"), w("quick")],
            volatile: vec![w("brown")],
        });
        assert_eq!(view.committed_text(), "the quick");
        assert_eq!(view.full_text(), "the quick brown");

        // next update commits "brown", revises tail to "fox"
        view.apply(&TranscriptUpdate {
            source: "Mic".into(),
            newly_committed: vec![w("brown")],
            volatile: vec![w("fox")],
        });
        assert_eq!(view.committed_text(), "the quick brown");
        assert_eq!(view.full_text(), "the quick brown fox");
    }

    #[test]
    fn empty_view_is_empty() {
        assert!(TranscriptView::default().is_empty());
        assert_eq!(TranscriptView::default().full_text(), "");
    }
}

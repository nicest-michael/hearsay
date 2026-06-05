//! LocalAgreement-2 word stabilization (the streaming-Whisper correctness core).
//!
//! Whisper is a batch model; we fake streaming by re-transcribing an overlapping
//! window each step and only **committing** words once two consecutive hypotheses
//! agree on them. Committed words never change; the unconfirmed tail is "volatile"
//! and may be revised next step. This is the LocalAgreement-2 policy from
//! ufal/whisper_streaming, adapted to carry centisecond timestamps so the audio
//! buffer can be trimmed at committed boundaries (see [`crate::aggregator`]).
//!
//! Inputs are **absolute-time** [`Word`]s (the app shifts window-relative
//! hypotheses by the window offset before calling [`Stabilizer::observe`]).

use crate::word::Word;

/// Result of feeding one hypothesis.
#[derive(Clone, Debug, PartialEq)]
pub struct StabilizeOutcome {
    /// Words promoted to "committed" this step (final, never revised).
    pub newly_committed: Vec<Word>,
    /// The current volatile tail (shown dimmer; may change next step).
    pub volatile: Vec<Word>,
    /// Absolute centisecond end-time of the last committed word — the audio
    /// buffer may be trimmed up to here.
    pub committed_up_to_cs: u32,
}

/// Accumulates committed words and stabilizes via two-window agreement.
#[derive(Clone, Debug)]
pub struct Stabilizer {
    committed: Vec<Word>,
    /// The previous hypothesis's uncommitted tail (absolute time).
    prev_tail: Vec<Word>,
    last_committed_cs: u32,
    max_overlap_ngram: usize,
}

impl Default for Stabilizer {
    fn default() -> Self {
        Self {
            committed: Vec::new(),
            prev_tail: Vec::new(),
            last_committed_cs: 0,
            max_overlap_ngram: 5,
        }
    }
}

impl Stabilizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Full committed transcript so far.
    pub fn committed(&self) -> &[Word] {
        &self.committed
    }

    pub fn last_committed_cs(&self) -> u32 {
        self.last_committed_cs
    }

    /// Feed the latest window hypothesis (absolute-time words). Returns what was
    /// newly committed and the current volatile tail.
    pub fn observe(&mut self, words: Vec<Word>) -> StabilizeOutcome {
        // 1. Drop words that end at/before the last committed time — already final.
        let last = self.last_committed_cs;
        let mut new: Vec<Word> = words.into_iter().filter(|w| w.t1_cs > last).collect();

        // An empty hypothesis carries no agreement information; keep the volatile
        // tail intact rather than wiping it (a silent window must not erase the
        // words still pending from the previous one).
        if new.is_empty() {
            return StabilizeOutcome {
                newly_committed: Vec::new(),
                volatile: self.prev_tail.clone(),
                committed_up_to_cs: self.last_committed_cs,
            };
        }

        // 2. Remove n-gram overlap: if the head of `new` repeats the tail of the
        //    committed transcript (window pre-roll re-emits boundary words), drop it.
        self.strip_committed_overlap(&mut new);

        // 3. LocalAgreement-2: commit the longest key-equal prefix of (prev_tail, new).
        let n = common_prefix_len(&self.prev_tail, &new);
        let newly: Vec<Word> = new[..n].to_vec();
        if let Some(last_word) = newly.last() {
            self.last_committed_cs = last_word.t1_cs;
        }
        self.committed.extend(newly.iter().cloned());
        self.prev_tail = new.split_off(n);

        StabilizeOutcome {
            newly_committed: newly,
            volatile: self.prev_tail.clone(),
            committed_up_to_cs: self.last_committed_cs,
        }
    }

    /// Commit the entire volatile tail (call on utterance end or stop). Returns
    /// the flushed words.
    pub fn flush(&mut self) -> Vec<Word> {
        let flushed = std::mem::take(&mut self.prev_tail);
        if let Some(last_word) = flushed.last() {
            self.last_committed_cs = last_word.t1_cs;
        }
        self.committed.extend(flushed.iter().cloned());
        flushed
    }

    /// Largest `n` (1..=max) where the committed tail equals the head of `new` by
    /// key; drop those `n` head words from `new`.
    fn strip_committed_overlap(&self, new: &mut Vec<Word>) {
        let max = self
            .max_overlap_ngram
            .min(self.committed.len())
            .min(new.len());
        let mut best = 0;
        for n in 1..=max {
            let c_tail = &self.committed[self.committed.len() - n..];
            let n_head = &new[..n];
            if keys_equal(c_tail, n_head) {
                best = n;
            }
        }
        if best > 0 {
            new.drain(..best);
        }
    }
}

fn common_prefix_len(a: &[Word], b: &[Word]) -> usize {
    a.iter()
        .zip(b)
        .take_while(|(x, y)| x.key() == y.key())
        .count()
}

fn keys_equal(a: &[Word], b: &[Word]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.key() == y.key())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::word::join_words;

    /// Helper: words with sequential 10cs spans starting at `start_cs`.
    fn words(texts: &[&str], start_cs: u32) -> Vec<Word> {
        texts
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let t0 = start_cs + i as u32 * 10;
                Word::new(*t, t0, t0 + 10)
            })
            .collect()
    }

    #[test]
    fn first_hypothesis_commits_nothing() {
        let mut s = Stabilizer::new();
        let out = s.observe(words(&["the", "quick", "brown"], 0));
        assert!(out.newly_committed.is_empty());
        assert_eq!(join_words(&out.volatile), "the quick brown");
    }

    #[test]
    fn agreement_commits_the_agreed_prefix() {
        let mut s = Stabilizer::new();
        s.observe(words(&["the", "quick", "brown"], 0));
        let out = s.observe(words(&["the", "quick", "brown", "fox"], 0));
        assert_eq!(join_words(&out.newly_committed), "the quick brown");
        assert_eq!(join_words(&out.volatile), "fox");
        assert_eq!(join_words(s.committed()), "the quick brown");
    }

    #[test]
    fn disagreement_keeps_tail_volatile_and_revises() {
        let mut s = Stabilizer::new();
        s.observe(words(&["the", "quick"], 0));
        // second window disagrees after "the"
        let out = s.observe(words(&["the", "slow"], 0));
        assert_eq!(join_words(&out.newly_committed), "the");
        assert_eq!(join_words(&out.volatile), "slow");
        // a third window can still revise the volatile tail
        let out3 = s.observe(words(&["slow", "loris"], 10));
        assert_eq!(join_words(&out3.newly_committed), "slow");
        assert_eq!(join_words(&out3.volatile), "loris");
    }

    #[test]
    fn case_and_punctuation_differences_still_agree() {
        let mut s = Stabilizer::new();
        s.observe(words(&["The", "quick"], 0));
        let out = s.observe(words(&["the,", "QUICK"], 0));
        assert_eq!(out.newly_committed.len(), 2);
        // display text comes from the latest hypothesis
        assert_eq!(join_words(&out.newly_committed), "the, QUICK");
    }

    #[test]
    fn committed_time_advances_to_last_word_end() {
        let mut s = Stabilizer::new();
        s.observe(words(&["a", "b", "c"], 0));
        let out = s.observe(words(&["a", "b", "c", "d"], 0));
        // committed a,b,c -> last end = 30cs
        assert_eq!(out.committed_up_to_cs, 30);
    }

    #[test]
    fn ngram_overlap_prevents_duplicate_boundary_words() {
        let mut s = Stabilizer::new();
        // commit "the quick brown fox"
        s.observe(words(&["the", "quick", "brown", "fox"], 0));
        s.observe(words(&["the", "quick", "brown", "fox"], 0));
        assert_eq!(join_words(s.committed()), "the quick brown fox");
        // next window (after a trim) re-emits "fox" as pre-roll, then continues.
        // give "fox" a timestamp that survives the time filter but duplicates text.
        let mut w = words(&["fox", "jumps", "over"], 35);
        // give "fox" a timestamp that survives the time filter (t1 > 40) so the
        // n-gram overlap dedup — not the time filter — is what removes it.
        w[0] = Word::new("fox", 38, 46);
        // feed the same continuation window twice so both windows fully agree.
        s.observe(w.clone());
        s.observe(w);
        // "fox" must appear exactly once — the boundary word was de-duplicated.
        assert_eq!(join_words(s.committed()), "the quick brown fox jumps over");
        assert_eq!(
            s.committed().iter().filter(|w| w.key() == "fox").count(),
            1,
            "boundary word 'fox' must not be duplicated"
        );
    }

    #[test]
    fn empty_hypothesis_preserves_volatile_tail() {
        let mut s = Stabilizer::new();
        s.observe(words(&["pending", "words"], 0));
        // a silent window returns nothing — the tail must survive, not be wiped
        let out = s.observe(Vec::new());
        assert_eq!(join_words(&out.volatile), "pending words");
        assert!(out.newly_committed.is_empty());
        // and it can still be flushed afterwards
        assert_eq!(join_words(&s.flush()), "pending words");
    }

    #[test]
    fn flush_commits_volatile_tail() {
        let mut s = Stabilizer::new();
        s.observe(words(&["hello", "there"], 0));
        s.observe(words(&["hello", "there", "friend"], 0)); // commit hello,there; tail=friend
        let flushed = s.flush();
        assert_eq!(join_words(&flushed), "friend");
        assert_eq!(join_words(s.committed()), "hello there friend");
    }

    #[test]
    fn already_committed_words_are_filtered_by_time() {
        let mut s = Stabilizer::new();
        s.observe(words(&["one", "two"], 0));
        s.observe(words(&["one", "two", "three"], 0)); // commit one,two (end 20)
                                                       // a stale window re-presenting only old words commits nothing new
        let out = s.observe(words(&["one", "two"], 0));
        assert!(out.newly_committed.is_empty());
    }
}

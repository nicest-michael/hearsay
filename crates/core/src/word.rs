//! A transcriber's output, in domain terms: timestamped [`Word`]s grouped into
//! [`Segment`]s, all inside a [`Hypothesis`]. Timestamps are **centiseconds**
//! (hundredths of a second). A hypothesis is *window-relative* until the app shifts
//! it to absolute time with [`Word::shifted`].

/// A single transcribed word with centisecond start/end timestamps.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub t0_cs: u32,
    pub t1_cs: u32,
}

impl Word {
    pub fn new(text: impl Into<String>, t0_cs: u32, t1_cs: u32) -> Self {
        Self {
            text: text.into(),
            t0_cs,
            t1_cs,
        }
    }

    /// Return a copy with both timestamps shifted later by `offset_cs`.
    pub fn shifted(&self, offset_cs: u32) -> Word {
        Word {
            text: self.text.clone(),
            t0_cs: self.t0_cs + offset_cs,
            t1_cs: self.t1_cs + offset_cs,
        }
    }

    /// Normalized comparison key for LocalAgreement matching: lowercased and
    /// stripped of surrounding non-alphanumerics, so `"Fox"` matches `"fox,"`.
    pub fn key(&self) -> String {
        self.text
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    }
}

/// A transcribed phrase with its time bounds and Whisper's confidence signals.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub text: String,
    pub t0_cs: u32,
    pub t1_cs: u32,
    /// Whisper's probability this segment is *not* speech. High => likely a
    /// hallucination over silence/noise (e.g. "Thank you for watching").
    pub no_speech_prob: f32,
    /// Average token log-probability. Low => low-confidence garbage.
    pub avg_logprob: f32,
}

impl Segment {
    /// Split this segment's text into [`Word`]s, distributing the segment's time
    /// span evenly across words (good-enough word timestamps without DTW).
    pub fn to_words(&self) -> Vec<Word> {
        let toks: Vec<&str> = self.text.split_whitespace().collect();
        if toks.is_empty() {
            return Vec::new();
        }
        let span = self.t1_cs.saturating_sub(self.t0_cs);
        let n = toks.len() as u32;
        toks.iter()
            .enumerate()
            .map(|(i, t)| {
                let i = i as u32;
                Word::new(
                    *t,
                    self.t0_cs + span * i / n,
                    self.t0_cs + span * (i + 1) / n,
                )
            })
            .collect()
    }
}

/// One pass of the transcriber over an audio window.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hypothesis {
    pub segments: Vec<Segment>,
}

impl Hypothesis {
    /// Build a single-segment hypothesis from plain text spanning `[t0_cs, t1_cs]`,
    /// marked fully confident. Handy for tests and simple transcribers.
    pub fn from_text(text: &str, t0_cs: u32, t1_cs: u32) -> Self {
        Hypothesis {
            segments: vec![Segment {
                text: text.to_string(),
                t0_cs,
                t1_cs,
                no_speech_prob: 0.0,
                avg_logprob: 0.0,
            }],
        }
    }

    /// Flatten to words, **dropping segments that fail the confidence gate**
    /// (the domain-owned anti-hallucination filter — see council revision R3).
    /// A segment is kept only if `no_speech_prob <= no_speech_max` AND
    /// `avg_logprob >= logprob_min`.
    pub fn confident_words(&self, no_speech_max: f32, logprob_min: f32) -> Vec<Word> {
        self.segments
            .iter()
            .filter(|s| s.no_speech_prob <= no_speech_max && s.avg_logprob >= logprob_min)
            .flat_map(|s| s.to_words())
            .collect()
    }
}

/// Join words into displayable text with single spaces.
pub fn join_words(words: &[Word]) -> String {
    words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_normalizes_case_and_punctuation() {
        assert_eq!(Word::new("Fox", 0, 1).key(), "fox");
        assert_eq!(Word::new("fox,", 0, 1).key(), "fox");
        assert_eq!(Word::new("\"Hello!\"", 0, 1).key(), "hello");
        assert_eq!(Word::new("don't", 0, 1).key(), "don't"); // inner apostrophe kept
    }

    #[test]
    fn shifted_moves_both_timestamps() {
        let w = Word::new("hi", 10, 20).shifted(100);
        assert_eq!((w.t0_cs, w.t1_cs), (110, 120));
    }

    #[test]
    fn segment_to_words_distributes_time() {
        let seg = Segment {
            text: "a b c".into(),
            t0_cs: 0,
            t1_cs: 30,
            no_speech_prob: 0.0,
            avg_logprob: 0.0,
        };
        let ws = seg.to_words();
        assert_eq!(ws.len(), 3);
        assert_eq!(ws[0], Word::new("a", 0, 10));
        assert_eq!(ws[1], Word::new("b", 10, 20));
        assert_eq!(ws[2], Word::new("c", 20, 30));
    }

    #[test]
    fn empty_segment_yields_no_words() {
        let seg = Segment {
            text: "   ".into(),
            t0_cs: 0,
            t1_cs: 10,
            no_speech_prob: 0.0,
            avg_logprob: 0.0,
        };
        assert!(seg.to_words().is_empty());
    }

    #[test]
    fn confident_words_filters_hallucinated_segment() {
        let hyp = Hypothesis {
            segments: vec![
                Segment {
                    text: "real speech".into(),
                    t0_cs: 0,
                    t1_cs: 20,
                    no_speech_prob: 0.1,
                    avg_logprob: -0.3,
                },
                Segment {
                    text: "thank you for watching".into(),
                    t0_cs: 20,
                    t1_cs: 40,
                    no_speech_prob: 0.92,
                    avg_logprob: -0.8,
                },
            ],
        };
        let words = hyp.confident_words(0.6, -1.0);
        assert_eq!(join_words(&words), "real speech");
    }

    #[test]
    fn confident_words_filters_low_logprob() {
        let hyp = Hypothesis {
            segments: vec![Segment {
                text: "garbled".into(),
                t0_cs: 0,
                t1_cs: 10,
                no_speech_prob: 0.2,
                avg_logprob: -2.5,
            }],
        };
        assert!(hyp.confident_words(0.6, -1.0).is_empty());
    }

    #[test]
    fn from_text_is_fully_confident() {
        let hyp = Hypothesis::from_text("hello world", 0, 100);
        assert_eq!(join_words(&hyp.confident_words(0.6, -1.0)), "hello world");
    }
}

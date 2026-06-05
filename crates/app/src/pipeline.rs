//! Orchestration over the core ports. The pure FSM lives in `hearsay-core`; this
//! layer wires a source → aggregator → transcriber → stabilizer → sink and carries
//! **zero concrete-tech knowledge** (generic over the port impls).
//!
//! Two reusable building blocks — [`infer_window`] and [`flush`] — are shared by
//! the single-threaded [`Pipeline`] (used in tests) and the binary's two-thread
//! worker (which calls them across the aggregator/inference split).

use hearsay_core::aggregator::{Window, WindowAggregator};
use hearsay_core::config::SessionConfig;
use hearsay_core::ports::{AudioSource, Transcriber, TranscriptSink};
use hearsay_core::stabilize::Stabilizer;
use hearsay_core::transcript::TranscriptUpdate;
use hearsay_core::vad::VadConfig;

use crate::error::PipelineError;

/// Result of transcribing one window.
#[derive(Clone, Debug, PartialEq)]
pub struct StepOutcome {
    pub update: TranscriptUpdate,
    /// Absolute centiseconds the audio buffer may now be trimmed up to.
    pub committed_up_to_cs: u32,
}

/// Transcribe one window, gate hallucinations, shift to absolute time, and fold
/// into the stabilizer. Returns the UI delta + the new committed boundary.
pub fn infer_window<T: Transcriber + ?Sized>(
    transcriber: &mut T,
    stabilizer: &mut Stabilizer,
    window: &Window,
    label: &str,
    cfg: &SessionConfig,
) -> Result<StepOutcome, PipelineError> {
    let hyp = transcriber.transcribe(&window.samples)?;
    let words: Vec<_> = hyp
        .confident_words(cfg.no_speech_max, cfg.logprob_min)
        .iter()
        .map(|w| w.shifted(window.offset_cs))
        .collect();
    let outcome = stabilizer.observe(words);
    Ok(StepOutcome {
        update: TranscriptUpdate {
            source: label.to_string(),
            newly_committed: outcome.newly_committed,
            volatile: outcome.volatile,
        },
        committed_up_to_cs: outcome.committed_up_to_cs,
    })
}

/// Commit the stabilizer's volatile tail (utterance end / stop). The returned
/// update has an empty volatile tail.
pub fn flush(stabilizer: &mut Stabilizer, label: &str) -> TranscriptUpdate {
    let flushed = stabilizer.flush();
    TranscriptUpdate {
        source: label.to_string(),
        newly_committed: flushed,
        volatile: Vec::new(),
    }
}

/// A single-source, single-threaded transcription pipeline. Drives one `tick()`
/// at a time: pull → buffer → (maybe) transcribe+stabilize+emit+trim → (maybe)
/// flush on utterance end. The binary's worker splits these steps across two
/// threads, but the logic is identical and shared via [`infer_window`]/[`flush`].
pub struct Pipeline<S: AudioSource, T: Transcriber, K: TranscriptSink> {
    source: S,
    transcriber: T,
    sink: K,
    aggregator: WindowAggregator,
    stabilizer: Stabilizer,
    cfg: SessionConfig,
    label: String,
}

impl<S: AudioSource, T: Transcriber, K: TranscriptSink> Pipeline<S, T, K> {
    pub fn new(source: S, transcriber: T, sink: K, cfg: SessionConfig, vad_cfg: VadConfig) -> Self {
        let label = source.label().to_string();
        Self {
            source,
            transcriber,
            sink,
            aggregator: WindowAggregator::new(cfg, vad_cfg),
            stabilizer: Stabilizer::new(),
            cfg,
            label,
        }
    }

    pub fn start(&mut self) -> Result<(), PipelineError> {
        self.source.start()?;
        Ok(())
    }

    /// One iteration of the pipeline.
    pub fn tick(&mut self) -> Result<(), PipelineError> {
        let audio = self.source.poll()?;
        self.aggregator.push_audio(&audio);

        if let Some(window) = self.aggregator.poll_window() {
            let out = infer_window(
                &mut self.transcriber,
                &mut self.stabilizer,
                &window,
                &self.label,
                &self.cfg,
            )?;
            self.sink.emit(&out.update);
            self.aggregator.trim_to(out.committed_up_to_cs);
        }

        if self.aggregator.utterance_ended() {
            let update = flush(&mut self.stabilizer, &self.label);
            if !update.newly_committed.is_empty() {
                self.sink.emit(&update);
            }
            self.aggregator.note_flushed();
        }
        Ok(())
    }

    /// Stop capture and flush any remaining volatile tail.
    pub fn stop(&mut self) {
        self.source.stop();
        let update = flush(&mut self.stabilizer, &self.label);
        if !update.newly_committed.is_empty() {
            self.sink.emit(&update);
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hearsay_core::audio::samples_for_cs;
    use hearsay_core::error::CaptureError;
    use hearsay_core::transcript::TranscriptView;
    use hearsay_core::word::Hypothesis;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    /// Yields one canned audio chunk per `poll`, then empty.
    struct FakeSource {
        chunks: VecDeque<Vec<f32>>,
        label: String,
        started: bool,
    }
    impl FakeSource {
        fn new(label: &str, chunks: Vec<Vec<f32>>) -> Self {
            Self {
                chunks: chunks.into(),
                label: label.into(),
                started: false,
            }
        }
    }
    impl AudioSource for FakeSource {
        fn start(&mut self) -> Result<(), CaptureError> {
            self.started = true;
            Ok(())
        }
        fn poll(&mut self) -> Result<Vec<f32>, CaptureError> {
            Ok(self.chunks.pop_front().unwrap_or_default())
        }
        fn stop(&mut self) {
            self.started = false;
        }
        fn label(&self) -> &str {
            &self.label
        }
    }

    /// Returns scripted hypotheses in order, one per `transcribe` call.
    struct FakeTranscriber {
        scripted: VecDeque<Hypothesis>,
    }
    impl FakeTranscriber {
        fn new(scripted: Vec<Hypothesis>) -> Self {
            Self {
                scripted: scripted.into(),
            }
        }
    }
    impl Transcriber for FakeTranscriber {
        fn transcribe(
            &mut self,
            _audio: &[f32],
        ) -> Result<Hypothesis, hearsay_core::error::TranscribeError> {
            Ok(self.scripted.pop_front().unwrap_or_default())
        }
    }

    /// Accumulates emitted updates into a view (Send, for the inference thread).
    #[derive(Clone, Default)]
    struct CollectingSink {
        view: Arc<Mutex<TranscriptView>>,
        count: Arc<Mutex<usize>>,
    }
    impl TranscriptSink for CollectingSink {
        fn emit(&mut self, update: &TranscriptUpdate) {
            self.view.lock().unwrap().apply(update);
            *self.count.lock().unwrap() += 1;
        }
    }

    fn speech(cs: u32) -> Vec<f32> {
        let n = samples_for_cs(cs);
        (0..n)
            .map(|i| if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect()
    }

    #[test]
    fn pipeline_converges_via_local_agreement() {
        // Three windows of growing hypotheses; LocalAgreement commits the agreed
        // prefix each step, the final tail is flushed on stop.
        let chunks = vec![speech(120), speech(120), speech(120)];
        let scripted = vec![
            Hypothesis::from_text("the quick", 0, 200),
            Hypothesis::from_text("the quick brown", 0, 300),
            Hypothesis::from_text("the quick brown fox", 0, 400),
        ];
        let sink = CollectingSink::default();
        let view = sink.view.clone();
        let mut p = Pipeline::new(
            FakeSource::new("Mic", chunks),
            FakeTranscriber::new(scripted),
            sink,
            SessionConfig::default(),
            VadConfig::default(),
        );
        p.start().unwrap();
        p.tick().unwrap(); // window 1: commit nothing
        assert!(
            view.lock().unwrap().committed.is_empty(),
            "must not commit from a single window (no agreement yet)"
        );
        p.tick().unwrap(); // window 2: commit "the quick"
        assert_eq!(view.lock().unwrap().committed_text(), "the quick");
        p.tick().unwrap(); // window 3: commit "brown"
        assert_eq!(view.lock().unwrap().committed_text(), "the quick brown");
        p.stop(); // flush "fox"
        assert_eq!(view.lock().unwrap().committed_text(), "the quick brown fox");
    }

    #[test]
    fn silence_produces_no_transcription() {
        let silence: Vec<f32> = vec![0.00001; samples_for_cs(300)];
        let sink = CollectingSink::default();
        let count = sink.count.clone();
        let mut p = Pipeline::new(
            FakeSource::new("Mic", vec![silence]),
            FakeTranscriber::new(vec![Hypothesis::from_text("ghost words", 0, 200)]),
            sink,
            SessionConfig::default(),
            VadConfig::default(),
        );
        p.start().unwrap();
        p.tick().unwrap();
        assert_eq!(
            *count.lock().unwrap(),
            0,
            "no window over silence -> no emit"
        );
    }

    #[test]
    fn utterance_end_flushes_tail_mid_session() {
        // tick 1: a speech window leaves "hello world" volatile (first window, so
        // nothing is committed yet). tick 2: 0.8 s of silence is below the 1 s step
        // (no second window emitted) but past the 700 ms utterance gap, so the
        // utterance-end flush is what commits the tail.
        let chunks = vec![speech(120), vec![0.00001; samples_for_cs(80)]];
        let scripted = vec![Hypothesis::from_text("hello world", 0, 200)];
        let sink = CollectingSink::default();
        let view = sink.view.clone();
        let mut p = Pipeline::new(
            FakeSource::new("Mic", chunks),
            FakeTranscriber::new(scripted),
            sink,
            SessionConfig::default(),
            VadConfig::default(),
        );
        p.start().unwrap();
        p.tick().unwrap();
        assert_eq!(view.lock().unwrap().committed_text(), "");
        p.tick().unwrap();
        assert_eq!(view.lock().unwrap().committed_text(), "hello world");
    }

    #[test]
    fn capture_error_propagates() {
        struct FailingSource;
        impl AudioSource for FailingSource {
            fn start(&mut self) -> Result<(), CaptureError> {
                Err(CaptureError::DeviceNotFound("nope".into()))
            }
            fn poll(&mut self) -> Result<Vec<f32>, CaptureError> {
                Ok(vec![])
            }
            fn stop(&mut self) {}
            fn label(&self) -> &str {
                "x"
            }
        }
        let mut p = Pipeline::new(
            FailingSource,
            FakeTranscriber::new(vec![]),
            CollectingSink::default(),
            SessionConfig::default(),
            VadConfig::default(),
        );
        assert!(matches!(p.start(), Err(PipelineError::Capture(_))));
    }
}

//! Microphone capture via `cpal`, exposed as an [`AudioSource`].
//!
//! The real-time capture callback does only allocation-free work — convert each
//! frame to a mono f32 and push it into a lock-free [`rtrb`] ring. [`poll`] (on the
//! aggregator thread) drains the ring and resamples to 16 kHz. The cpal `Stream` is
//! `!Send` on CoreAudio, so it lives behind [`SendStream`]; we only ever build,
//! use, and drop it on the single aggregator thread that owns this source, so the
//! `unsafe impl Send` never results in real cross-thread access.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use hearsay_core::error::CaptureError;
use hearsay_core::ports::AudioSource;
use rtrb::{Consumer, Producer, RingBuffer};

use crate::resample::Resampler16k;

/// A selectable input device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
}

/// Enumerate input devices for the UI's source picker.
// `name()` is deprecated in cpal 0.17 in favor of `description()`/`id()`, but the
// plain name is exactly what users recognize in a picker; richer device info is a
// future enhancement.
#[allow(deprecated)]
pub fn list_input_devices() -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());
    let mut out = Vec::new();
    if let Ok(devices) = host.input_devices() {
        for d in devices {
            if let Ok(name) = d.name() {
                let is_default = Some(&name) == default_name.as_ref();
                out.push(DeviceInfo { name, is_default });
            }
        }
    }
    out
}

/// Wrapper asserting `Send` for the `!Send` CoreAudio stream. Sound because the
/// stream is only ever touched on the aggregator thread (see module docs).
struct SendStream(#[allow(dead_code)] cpal::Stream);
unsafe impl Send for SendStream {}

pub struct CpalMicSource {
    /// `None` = system default input.
    device_name: Option<String>,
    label: String,
    stream: Option<SendStream>,
    consumer: Option<Consumer<f32>>,
    resampler: Resampler16k,
}

impl CpalMicSource {
    /// `device_name = None` selects the default input device.
    pub fn new(device_name: Option<String>) -> Self {
        let label = device_name
            .clone()
            .unwrap_or_else(|| "Default microphone".to_string());
        Self {
            device_name,
            label,
            stream: None,
            consumer: None,
            resampler: Resampler16k::new(hearsay_core::TARGET_RATE),
        }
    }
}

impl AudioSource for CpalMicSource {
    #[allow(deprecated)] // see note on list_input_devices re: name()
    fn start(&mut self) -> Result<(), CaptureError> {
        if self.stream.is_some() {
            return Ok(());
        }
        let host = cpal::default_host();
        let device = match &self.device_name {
            Some(name) => host
                .input_devices()
                .map_err(|e| CaptureError::Backend(e.to_string()))?
                .find(|d| d.name().ok().as_deref() == Some(name.as_str()))
                .ok_or_else(|| CaptureError::DeviceNotFound(name.clone()))?,
            None => host
                .default_input_device()
                .ok_or_else(|| CaptureError::DeviceNotFound("default input".into()))?,
        };
        if let Ok(name) = device.name() {
            self.label = name;
        }

        let supported = device
            .default_input_config()
            .map_err(|e| CaptureError::Backend(e.to_string()))?;
        let in_rate = supported.sample_rate(); // cpal 0.17: SampleRate is a u32 alias
        let channels = supported.channels() as usize;
        let sample_format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();

        // ~10 s of device-rate mono headroom so a slow inference step can't starve us.
        let capacity = (in_rate as usize * 10).max(16_000);
        let (producer, consumer) = RingBuffer::<f32>::new(capacity);

        let stream = build_input_stream(&device, &config, sample_format, channels, producer)?;
        stream
            .play()
            .map_err(|e| CaptureError::Backend(e.to_string()))?;

        self.resampler = Resampler16k::new(in_rate);
        self.consumer = Some(consumer);
        self.stream = Some(SendStream(stream));
        log::info!(
            "mic capture started: {} @ {in_rate} Hz, {channels} ch",
            self.label
        );
        Ok(())
    }

    fn poll(&mut self) -> Result<Vec<f32>, CaptureError> {
        let Some(consumer) = self.consumer.as_mut() else {
            return Ok(Vec::new());
        };
        let mut mono = Vec::with_capacity(consumer.slots());
        while let Ok(s) = consumer.pop() {
            mono.push(s);
        }
        Ok(self.resampler.process(&mono))
    }

    fn stop(&mut self) {
        self.stream = None;
        self.consumer = None;
    }

    fn label(&self) -> &str {
        &self.label
    }
}

/// Build a cpal input stream for the device's native sample format, down-mixing to
/// mono f32 in the (allocation-free) callback.
fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    format: cpal::SampleFormat,
    channels: usize,
    producer: Producer<f32>,
) -> Result<cpal::Stream, CaptureError> {
    use cpal::SampleFormat as F;
    match format {
        F::F32 => make_stream::<f32>(device, config, channels, producer),
        F::I16 => make_stream::<i16>(device, config, channels, producer),
        F::U16 => make_stream::<u16>(device, config, channels, producer),
        F::I32 => make_stream::<i32>(device, config, channels, producer),
        F::F64 => make_stream::<f64>(device, config, channels, producer),
        other => Err(CaptureError::Unsupported(format!(
            "unsupported sample format {other:?}"
        ))),
    }
}

fn make_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    mut producer: Producer<f32>,
) -> Result<cpal::Stream, CaptureError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let err_fn = |e| log::warn!("mic stream error: {e}");
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let inv = 1.0 / channels as f32;
                for frame in data.chunks_exact(channels) {
                    let mut sum = 0.0f32;
                    for &s in frame {
                        sum += f32::from_sample(s);
                    }
                    // Drop on overrun rather than block — RT-safe.
                    let _ = producer.push(sum * inv);
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| CaptureError::Backend(e.to_string()))
}

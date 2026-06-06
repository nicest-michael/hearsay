//! Spike: prove the macOS VoiceProcessingIO (hardware AEC) AudioUnit works as a single
//! duplex unit in Rust via coreaudio-rs, before building the real adapter around it.
//!
//! Confirms: the unit creates + initializes (no -10851 format error), the mic input
//! callback fires with echo-cancelled frames, and the render callback fires for output.
//! Outputs silence so it doesn't blast audio. Run: `cargo run -p hearsay-adapters --example vpio_spike`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use coreaudio::audio_unit::audio_format::LinearPcmFlags;
use coreaudio::audio_unit::render_callback::{self, data};
use coreaudio::audio_unit::{AudioUnit, Element, IOType, SampleFormat, Scope, StreamFormat};
use objc2_audio_toolbox::{kAudioOutputUnitProperty_EnableIO, kAudioUnitProperty_StreamFormat};

fn main() -> Result<(), coreaudio::Error> {
    println!("[vpio] creating VoiceProcessingIO unit…");
    let mut au = AudioUnit::new_uninitialized(IOType::VoiceProcessingIO)?;

    // Enable mic (bus 1, input scope) and speaker (bus 0, output scope).
    let enable: u32 = 1;
    au.set_property(
        kAudioOutputUnitProperty_EnableIO,
        Scope::Input,
        Element::Input,
        Some(&enable),
    )?;
    au.set_property(
        kAudioOutputUnitProperty_EnableIO,
        Scope::Output,
        Element::Output,
        Some(&enable),
    )?;

    // VPIO runs at the hardware rate; query it (fall back to 48k) and match our client format.
    let hw_rate = au
        .stream_format(Scope::Output, Element::Output)
        .map(|f| f.sample_rate)
        .unwrap_or(48_000.0);
    println!("[vpio] hardware rate: {hw_rate} Hz");

    let fmt = StreamFormat {
        sample_rate: hw_rate,
        sample_format: SampleFormat::F32,
        flags: LinearPcmFlags::IS_FLOAT | LinearPcmFlags::IS_PACKED | LinearPcmFlags::IS_NON_INTERLEAVED,
        channels: 1, // VPIO + coreaudio-rs input callback are mono
    };
    let asbd = fmt.to_asbd();
    // mic (near-end) format: Scope::Output, Element::Input (bus 1)
    au.set_property(kAudioUnitProperty_StreamFormat, Scope::Output, Element::Input, Some(&asbd))?;
    // speaker (far-end / AEC reference) format: Scope::Input, Element::Output (bus 0)
    au.set_property(kAudioUnitProperty_StreamFormat, Scope::Input, Element::Output, Some(&asbd))?;

    let mic_frames = Arc::new(AtomicU64::new(0));
    let mic_energy = Arc::new(Mutex::new(0.0f64));
    let render_calls = Arc::new(AtomicU64::new(0));

    type Args = render_callback::Args<data::NonInterleaved<f32>>;

    {
        let mic_frames = mic_frames.clone();
        let mic_energy = mic_energy.clone();
        au.set_input_callback(move |args: Args| {
            let Args {
                num_frames,
                mut data,
                ..
            } = args;
            mic_frames.fetch_add(num_frames as u64, Ordering::Relaxed);
            if let Some(ch) = data.channels_mut().next() {
                let e: f64 = ch.iter().take(num_frames).map(|&v| (v as f64) * (v as f64)).sum();
                *mic_energy.lock().unwrap() += e;
            }
            Ok(())
        })?;
    }
    {
        let render_calls = render_calls.clone();
        au.set_render_callback(move |args: Args| {
            let Args {
                num_frames,
                mut data,
                ..
            } = args;
            render_calls.fetch_add(1, Ordering::Relaxed);
            for ch in data.channels_mut() {
                ch.iter_mut().take(num_frames).for_each(|s| *s = 0.0); // silence
            }
            Ok(())
        })?;
    }

    println!("[vpio] initializing (this triggers the mic TCC prompt if not yet granted)…");
    au.initialize()?;
    au.start()?;
    println!("[vpio] running 3s…");
    std::thread::sleep(Duration::from_secs(3));
    au.stop()?;

    let frames = mic_frames.load(Ordering::Relaxed);
    let energy = *mic_energy.lock().unwrap();
    let rms = if frames > 0 {
        (energy / frames as f64).sqrt()
    } else {
        0.0
    };
    println!("[vpio] RESULT: mic_frames={frames}  render_calls={}  mic_rms={rms:.5}", render_calls.load(Ordering::Relaxed));
    if frames > 0 && render_calls.load(Ordering::Relaxed) > 0 {
        println!("[vpio] PASS — duplex VPIO runs (mic capture + render both firing)");
    } else {
        println!("[vpio] PARTIAL — unit ran but mic_frames or render_calls is 0 (check TCC mic permission)");
    }
    Ok(())
}

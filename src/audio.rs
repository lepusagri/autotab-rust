use anyhow::{Context, Result, ensure};
use std::{path::Path, process::Command};

/// Interleaved samples; FFmpeg handles decoding and band-limited resampling.
pub fn decode(ffmpeg: &Path, input: &Path, rate: u32, channels: u16) -> Result<Vec<f32>> {
    let mut command = Command::new(ffmpeg);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW for GUI-launched jobs
    }
    let output = command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(input)
        .args([
            "-vn",
            "-ac",
            &channels.to_string(),
            "-ar",
            &rate.to_string(),
            "-f",
            "f32le",
            "pipe:1",
        ])
        .output()
        .context("Cannot start FFmpeg")?;
    ensure!(
        output.status.success(),
        "FFmpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    ensure!(
        output
            .stdout
            .len()
            .is_multiple_of(4 * usize::from(channels)),
        "Invalid PCM data from FFmpeg"
    );
    let samples: Vec<_> = output
        .stdout
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    ensure!(!samples.is_empty(), "Input contains no audio samples");
    ensure!(
        samples.iter().all(|sample| sample.is_finite()),
        "Audio contains non-finite samples"
    );
    Ok(samples)
}

pub fn write_stereo(path: &Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44_100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

//! Opt-in hardware test: cargo test --release --test arc -- --ignored --nocapture
use std::{f64::consts::PI, fs, path::Path, process::Command};

fn transcribe(audio: &Path, directory: &Path, name: &str, device: &str) -> serde_json::Value {
    let prefix = directory.join(name);
    let profiles = directory.join(format!("{name}-profile"));
    let result = Command::new(env!("CARGO_BIN_EXE_autotab"))
        .args(["--input"])
        .arg(audio)
        .args(["--device", device, "--precision", "fp32", "-o"])
        .arg(prefix.with_extension("txt"))
        .arg("--guitar-output")
        .arg(prefix.with_extension("wav"))
        .arg("--timings-json")
        .arg(prefix.with_extension("json"))
        .arg("--profile-dir")
        .arg(&profiles)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    println!("{}", String::from_utf8_lossy(&result.stdout));
    if device == "openvino" {
        let profile: serde_json::Value =
            serde_json::from_slice(&fs::read(profiles.join("demucs-openvino.json")).unwrap())
                .unwrap();
        assert!(
            profile["execution_devices"]
                .as_str()
                .unwrap()
                .contains("GPU")
        );
        assert_eq!(profile["inference_precision"], "f32");
        assert!(
            profile["driver"]
                .as_str()
                .is_some_and(|driver| !driver.is_empty() && driver != "unknown")
        );
        return profile;
    }
    serde_json::Value::Null
}

#[test]
#[ignore = "Requires Intel GPU, native runtimes and Demucs model from setup-rust.ps1"]
fn arc_cache_and_fp32_audio_accuracy() {
    let work = tempfile::tempdir().unwrap();
    let audio = work.path().join("plucked-a2.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44_100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&audio, spec).unwrap();
    for frame in 0..44_100 * 9 {
        let t = f64::from(frame) / 44_100.0;
        let envelope = (-2.0 * (t % 0.75)).exp();
        let value = (1..=6)
            .map(|harmonic| {
                (2.0 * PI * 110.0 * f64::from(harmonic) * t).sin() / f64::from(harmonic)
            })
            .sum::<f64>()
            * envelope
            * 0.2;
        writer.write_sample(value as f32).unwrap();
        writer.write_sample(value as f32).unwrap();
    }
    writer.finalize().unwrap();
    transcribe(&audio, work.path(), "cpu", "cpu");
    transcribe(&audio, work.path(), "gpu", "openvino");
    let warm = transcribe(&audio, work.path(), "gpu-warm", "openvino");
    assert_eq!(
        warm["loaded_from_cache"], true,
        "GPU cache must be reusable across processes"
    );
    let read = |name: &str| {
        hound::WavReader::open(work.path().join(name))
            .unwrap()
            .into_samples::<f32>()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
    };
    let cpu = read("cpu.wav");
    let gpu = read("gpu-warm.wav");
    assert_eq!(cpu.len(), gpu.len());
    let energy = cpu.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    let error = cpu
        .iter()
        .zip(&gpu)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>();
    let max_error = cpu
        .iter()
        .zip(&gpu)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let reference_rms = (energy / cpu.len() as f64).sqrt();
    let error_rms = (error / cpu.len() as f64).sqrt();
    let snr = 10.0 * (energy / error).log10();
    println!("GPU vs CPU: max error {max_error:.8}, RMS error {error_rms:.8}, SNR {snr:.2} dB");
    assert!(
        max_error < 0.002,
        "GPU output deviates by over 0.2% of full scale"
    );
    if reference_rms > 0.0001 {
        assert!(snr > 40.0, "GPU RMS error must be below 1% of signal RMS");
    } else {
        assert!(
            error_rms < 0.00001,
            "GPU introduces excessive noise into a near-silent stem"
        );
    }
}

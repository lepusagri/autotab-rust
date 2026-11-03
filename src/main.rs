#![deny(unsafe_op_in_unsafe_fn)]

mod audio;
mod download;
mod inference;
mod instruments;
mod intel_gpu;
mod muscriptor;
mod separation;
mod tab;
mod tools;

use anyhow::{Context, Result, ensure};
use basic_pitch::{BasicPitch, TrimMode};
use clap::{ArgGroup, Parser};
use std::{fs, path::PathBuf, sync::Arc, time::Instant};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Turn audio into guitar tabs locally, without Python."
)]
#[command(group(ArgGroup::new("source").required(true).args(["input", "url"])))]
struct Args {
    /// Audio file (MP3, WAV, FLAC, etc.)
    #[arg(long)]
    input: Option<PathBuf>,
    /// Direct HTTP(S) audio URL or public YouTube video URL
    #[arg(long)]
    url: Option<String>,
    #[arg(long, value_enum, default_value = "standard")]
    tuning: tab::Tuning,
    /// Tab output path (default: output/<input>_tab.txt)
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// MIDI output path (default: alongside the tab file)
    #[arg(long)]
    midi_output: Option<PathBuf>,
    /// Transcribe a clean guitar recording without stem separation
    #[arg(long)]
    skip_separation: bool,
    /// Override the six-stem Demucs ONNX model
    #[arg(long)]
    model: Option<PathBuf>,
    /// Note-recognition model (MuScriptor uses native OpenVINO, no Python)
    #[arg(long, value_enum, default_value = "basic-pitch")]
    transcriber: muscriptor::Transcriber,
    /// Local MuScriptor model.safetensors or directory; otherwise uses HF cache/download
    #[arg(long, requires = "transcriber")]
    muscriptor_model: Option<PathBuf>,
    /// Expected MuScriptor instruments: comma/space-separated names or group IDs, or 'auto'
    #[arg(long, alias = "instrument-group", default_value = "4,5,6")]
    instruments: instruments::Instruments,
    /// Maximum CPU threads used by Demucs
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u16).range(1..))]
    threads: u16,
    /// Inference backend. Auto prefers the installed Intel OpenVINO GPU runtime
    #[arg(long, value_enum, default_value = "auto")]
    device: inference::Device,
    /// Intel OpenVINO GPU index
    #[arg(long, default_value_t = 0)]
    gpu_id: u16,
    /// OpenVINO GPU precision; FP32 preserves transcription accuracy
    #[arg(long, value_enum, default_value = "fp32")]
    precision: inference::Precision,
    /// Write ONNX node/provider execution profiles
    #[arg(long)]
    profile_dir: Option<PathBuf>,
    /// Save pipeline timings as JSON
    #[arg(long)]
    timings_json: Option<PathBuf>,
    /// Save the isolated guitar WAV (also useful for CPU/GPU quality comparisons)
    #[arg(long)]
    guitar_output: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("Error: {error:#}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    let total = Instant::now();
    ensure!(
        args.transcriber.variant().is_some()
            || args.instruments == instruments::Instruments::default(),
        "--instruments requires a muscriptor-* --transcriber"
    );
    let skip_separation = args.skip_separation
        || (args.transcriber.variant().is_some() && args.instruments.has_other_instruments());
    if skip_separation && !args.skip_separation {
        println!(
            "  MuScriptor uses the original mix for automatic or non-guitar instrument selection."
        );
    }
    ensure!(
        args.muscriptor_model.is_none() || args.transcriber.variant().is_some(),
        "--muscriptor-model requires a muscriptor-* --transcriber"
    );
    if let Some(input) = &args.input {
        ensure!(input.is_file(), "Input file not found: {}", input.display());
    }
    let home = tools::project_home()?;
    let ffmpeg = tools::find_ffmpeg(&home)?;
    let device = args.device.resolve(&home);
    ensure!(
        args.precision != inference::Precision::Fp16 || device == inference::Device::Openvino,
        "--precision fp16 is only supported with OpenVINO"
    );
    let runtime = device.runtime(&home);
    ensure!(
        runtime.is_file(),
        "ONNX Runtime for {device:?} missing at {}. Run setup-rust.ps1 first.",
        runtime.display()
    );
    let model = args
        .model
        .unwrap_or_else(|| home.join("models/htdemucs_6s.onnx"));
    if !skip_separation {
        ensure!(
            model.is_file(),
            "Demucs model missing: {}. Run setup-rust.ps1.",
            model.display()
        );
    }
    let work = tempfile::tempdir().context("Cannot create working directory")?;
    let input = match args.input {
        Some(path) => path,
        None => {
            println!("[1/4] Downloading audio...");
            download::audio(args.url.as_deref().context("Missing source")?, work.path())?
        }
    };
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let output = args
        .output
        .unwrap_or_else(|| PathBuf::from("output").join(format!("{stem}_tab.txt")));
    let midi_output = args
        .midi_output
        .unwrap_or_else(|| output.with_extension("mid"));
    ensure!(
        output != midi_output && output != input && midi_output != input,
        "Input, MIDI and tab paths must differ"
    );
    println!(
        "[2/4] Decoding audio{}...",
        if skip_separation {
            ""
        } else {
            " and isolating guitar"
        }
    );
    let mix = audio::decode(&ffmpeg, &input, 44_100, 2)?;
    let audio_seconds = mix.len() as f64 / (44_100.0 * 2.0);
    let config = inference::Config {
        device,
        gpu_id: args.gpu_id,
        precision: args.precision,
        threads: args.threads.into(),
        cache: home.join(".cache/openvino"),
        profiles: args.profile_dir,
        openvino_library: home.join("runtime/openvino-2026.4.1/openvino_c.dll"),
    };
    println!(
        "Inference: {device:?}, GPU index {}, precision {:?}; transcriber {:?}",
        args.gpu_id, args.precision, args.transcriber
    );
    let environment = ort::init_from(&runtime)?
        .with_name("autotab")
        .with_telemetry(false)
        .build()?;
    let separation_started = Instant::now();
    let guitar = if skip_separation {
        mix
    } else {
        separation::guitar(&environment, &model, &mix, &config)?
    };
    let separation_seconds = separation_started.elapsed().as_secs_f64();
    let guitar_path = work.path().join("guitar.wav");
    audio::write_stereo(&guitar_path, &guitar)?;
    if let Some(path) = args.guitar_output {
        tools::create_parent(&path)?;
        fs::copy(&guitar_path, &path)
            .with_context(|| format!("Cannot write {}", path.display()))?;
    }
    println!("[3/4] Transcribing with {:?}...", args.transcriber);
    let transcription_started = Instant::now();
    let (notes, multi_instrument) = if let Some(variant) = args.transcriber.variant() {
        let mono = audio::decode(&ffmpeg, &guitar_path, 16_000, 1)?;
        let transcription = muscriptor::transcribe(
            variant,
            &mono,
            &home,
            args.muscriptor_model.as_deref(),
            &config,
            &args.instruments,
        )?;
        (transcription.guitar_notes(), Some(transcription))
    } else {
        let mono = audio::decode(&ffmpeg, &guitar_path, 22_050, 1)?;
        let backend = Arc::new(inference::PitchBackend::new(&environment, &config)?);
        let transcriber =
            BasicPitch::with_backend(backend.clone()).with_trim_mode(TrimMode::V0_4_0);
        let transcription = transcriber.transcribe(&mono)?;
        backend.finish_profile(&config)?;
        (transcription.notes, None)
    };
    let transcription_seconds = transcription_started.elapsed().as_secs_f64();
    let midi_notes = multi_instrument
        .as_ref()
        .map_or(notes.len(), |transcription| transcription.notes.len());
    println!(
        "[4/4] Writing MIDI ({midi_notes} notes) and guitar tab ({} notes)...",
        notes.len()
    );
    let midi = if let Some(transcription) = multi_instrument {
        transcription.to_midi()?
    } else {
        basic_pitch::midi::note_events_to_midi_bytes(&notes, &Default::default())?
    };
    tools::create_parent(&output)?;
    tools::create_parent(&midi_output)?;
    fs::write(&midi_output, midi)
        .with_context(|| format!("Cannot write {}", midi_output.display()))?;
    fs::write(&output, tab::render(&notes, args.tuning))
        .with_context(|| format!("Cannot write {}", output.display()))?;
    println!("Tab: {}\nMIDI: {}", output.display(), midi_output.display());
    if notes.is_empty() && midi_notes > 0 {
        println!(
            "No guitar notes detected. Other recognized instruments are retained in the MIDI."
        );
    } else if notes.is_empty() {
        println!("No notes detected. Try a cleaner guitar recording or --skip-separation.");
    }
    let elapsed = total.elapsed().as_secs_f64();
    println!(
        "Timing: separation {separation_seconds:.2}s, transcription {transcription_seconds:.2}s, total {elapsed:.2}s for {audio_seconds:.2}s audio"
    );
    if let Some(path) = args.timings_json {
        tools::create_parent(&path)?;
        let timing = serde_json::json!({"device": format!("{device:?}"), "gpu_id": args.gpu_id,
            "precision": format!("{:?}", args.precision),
            "audio_seconds": audio_seconds, "separation_seconds": separation_seconds,
            "transcription_seconds": transcription_seconds, "total_seconds": elapsed,
            "transcriber": format!("{:?}", args.transcriber), "instruments": args.instruments.to_string(), "notes": notes.len(), "midi_notes": midi_notes});
        fs::write(path, serde_json::to_vec_pretty(&timing)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    #[test]
    fn accepts_instrument_names_ids_and_automatic_mode() {
        for (flag, value) in [
            (
                "--instruments",
                "acoustic_piano,acoustic_guitar,acoustic_bass",
            ),
            ("--instrument-group", "0 4 7"),
        ] {
            let args = Args::try_parse_from([
                "autotab",
                "--input",
                "mix.wav",
                "--transcriber",
                "muscriptor-small",
                flag,
                value,
            ])
            .unwrap();
            assert_eq!(args.instruments.to_string(), "0,4,7");
        }
        let args = Args::try_parse_from([
            "autotab",
            "--input",
            "mix.wav",
            "--transcriber",
            "muscriptor-small",
            "--instruments",
            "auto",
        ])
        .unwrap();
        assert!(args.instruments.is_automatic());
        assert!(
            Args::try_parse_from([
                "autotab",
                "--input",
                "mix.wav",
                "--instruments",
                "not_an_instrument"
            ])
            .is_err()
        );
    }
    #[test]
    fn accepts_all_muscriptor_variants_and_local_checkpoint() {
        for name in ["muscriptor-small", "muscriptor-medium", "muscriptor-large"] {
            let args = Args::try_parse_from([
                "autotab",
                "--input",
                "guitar.wav",
                "--transcriber",
                name,
                "--muscriptor-model",
                "C:\\Music\\Modell März.safetensors",
            ])
            .unwrap();
            assert!(args.transcriber.variant().is_some());
            assert_eq!(
                args.muscriptor_model.unwrap(),
                PathBuf::from("C:\\Music\\Modell März.safetensors")
            );
        }
        assert!(
            Args::try_parse_from([
                "autotab",
                "--input",
                "guitar.wav",
                "--transcriber",
                "invalid"
            ])
            .is_err()
        );
    }
}

#![deny(unsafe_op_in_unsafe_fn)]

mod audio;
mod download;
mod fingering;
mod gp5;
mod inference;
mod instruments;
mod intel_gpu;
mod midi_input;
mod muscriptor;
mod rhythm;
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
    about = "Turn audio or MIDI into guitar tabs locally, without Python."
)]
#[command(group(ArgGroup::new("source").required(true).args(["input", "url"])))]
struct Args {
    /// Audio or MIDI file (MP3, WAV, MID, MIDI, etc.)
    #[arg(long)]
    input: Option<PathBuf>,
    /// MIDI selection: auto (guitar programs, otherwise melodic notes), all, or zero-based track index
    #[arg(long, default_value = "auto")]
    midi_track: midi_input::Selection,
    /// List MIDI tracks and exit, without model inference
    #[arg(long)]
    list_midi_tracks: bool,
    /// Direct HTTP(S) audio URL or public YouTube video URL
    #[arg(long)]
    url: Option<String>,
    #[arg(long, value_enum, default_value = "standard")]
    tuning: tab::Tuning,
    /// Fingering engine: pretrained tab-labeler v3 or offline rule-based search
    #[arg(long, value_enum, default_value = "v3")]
    tab_engine: fingering::Engine,
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
    /// Score/MIDI tempo in quarter notes per minute; does not time-stretch audio
    #[arg(long, default_value_t = 120.0)]
    bpm: f64,
    /// Score/MIDI time signature (e.g. 4/4, 3/4, 6/8)
    #[arg(long, default_value = "4/4")]
    time_signature: rhythm::Meter,
    /// MIDI note grid; GP5 always uses at least a 1/32 notation grid
    #[arg(long, value_enum, default_value = "off")]
    quantize: rhythm::Grid,
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
    if let Some(input) = args
        .input
        .as_ref()
        .filter(|p| midi_input::is_midi(p))
        .cloned()
    {
        return run_midi(args, input);
    }
    ensure!(
        !args.list_midi_tracks && args.midi_track == midi_input::Selection::Auto,
        "--midi-track and --list-midi-tracks require a .mid or .midi input"
    );
    let total = Instant::now();
    let rhythm = rhythm::Rhythm {
        bpm: args.bpm,
        meter: args.time_signature,
        grid: args.quantize,
    };
    rhythm.validate()?;
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
    let gp5_output = output.with_extension("gp5");
    let fingering_output = output.with_extension("fingering.json");
    let generated = [&output, &midi_output, &gp5_output, &fingering_output];
    for (index, path) in generated.iter().enumerate() {
        ensure!(
            **path != input && generated[..index].iter().all(|other| other != path),
            "Input, tab, MIDI, GP5 and fingering paths must differ"
        );
    }
    if let Some(path) = &args.timings_json {
        ensure!(
            *path != input && !generated.contains(&path),
            "Timing report must use a separate output path"
        );
    }
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
    let arrangement = fingering::run(
        &notes,
        args.tuning,
        args.tab_engine,
        &home,
        &environment,
        &config,
    )?;
    let (gp5, gp5_report) = gp5::export(&arrangement, args.tuning, rhythm, &stem)?;
    let midi = if let Some(transcription) = multi_instrument {
        transcription.to_midi(&rhythm)?
    } else {
        muscriptor::Transcription {
            notes: notes
                .iter()
                .cloned()
                .map(|note| muscriptor::DetectedNote { program: 24, note })
                .collect(),
        }
        .to_midi(&rhythm)?
    };
    tools::create_parent(&output)?;
    tools::create_parent(&midi_output)?;
    fs::write(&midi_output, midi)
        .with_context(|| format!("Cannot write {}", midi_output.display()))?;
    fs::write(&output, fingering::render(&arrangement))
        .with_context(|| format!("Cannot write {}", output.display()))?;
    fs::write(&gp5_output, gp5)?;
    let report = serde_json::json!({
        "engine": args.tab_engine, "model": matches!(args.tab_engine, fingering::Engine::V3).then_some(fingering::MODEL),
        "tuning_low_to_high": args.tuning.pitches(),
        "model_revision": matches!(args.tab_engine, fingering::Engine::V3).then_some(fingering::REVISION),
        "model_credit": matches!(args.tab_engine, fingering::Engine::V3).then_some("cstr; CC BY 4.0"),
        "arrangement": arrangement, "gp5": gp5_report,
        "source_notes": notes.iter().enumerate().map(|(index, n)| serde_json::json!({
            "index": index, "pitch": n.pitch, "start_s": n.start_s, "end_s": n.end_s,
            "amplitude": n.amplitude })).collect::<Vec<_>>()
    });
    fs::write(fingering_output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "Tab: {}\nMIDI: {}\nGuitar Pro: {}",
        output.display(),
        midi_output.display(),
        gp5_output.display()
    );
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
            "transcriber": format!("{:?}", args.transcriber), "instruments": args.instruments.to_string(), "notes": notes.len(), "midi_notes": midi_notes,
            "midi_bpm": rhythm.bpm, "time_signature": rhythm.meter.to_string(), "quantization": rhythm.grid.as_cli()});
        fs::write(path, serde_json::to_vec_pretty(&timing)?)?;
    }
    Ok(())
}

fn run_midi(args: Args, input: PathBuf) -> Result<()> {
    let started = Instant::now();
    println!("[1/4] Reading MIDI...");
    let source = midi_input::MidiFile::read(&input)?;
    if args.list_midi_tracks {
        for track in &source.tracks {
            println!(
                "Track {}: {} - {} melodic notes, {} guitar notes, {} drum notes",
                track.index, track.name, track.melodic_notes, track.guitar_notes, track.drum_notes
            );
        }
        return Ok(());
    }
    ensure!(
        args.bpm == 120.0
            && args.time_signature == rhythm::Meter::default()
            && args.quantize == rhythm::Grid::Off,
        "MIDI tempo, meter and note times are imported from the file; --bpm, --time-signature and --quantize are audio settings"
    );
    let selected = source.select(args.midi_track)?;
    let notes: Vec<_> = selected.iter().map(|n| n.note.clone()).collect();
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let output = args
        .output
        .unwrap_or_else(|| PathBuf::from("output").join(format!("{stem}_tab.txt")));
    let midi_output = args
        .midi_output
        .unwrap_or_else(|| output.with_extension("mid"));
    let gp5_output = output.with_extension("gp5");
    let report_output = output.with_extension("fingering.json");
    validate_outputs(
        &input,
        &[&output, &midi_output, &gp5_output, &report_output],
        args.timings_json.as_deref(),
    )?;
    println!(
        "[2/4] Selected {} MIDI notes; no audio transcription or separation",
        notes.len()
    );
    if source.repaired_note_offs > 0 {
        println!(
            "  {} unterminated notes closed at track end",
            source.repaired_note_offs
        );
    }
    let home = tools::project_home()?;
    let config = inference::Config {
        device: inference::Device::Cpu,
        gpu_id: 0,
        precision: inference::Precision::Fp32,
        threads: usize::from(args.threads),
        cache: home.join(".cache"),
        profiles: args.profile_dir,
        openvino_library: home.join("runtime/openvino-2026.4.1/openvino_c.dll"),
    };
    let environment = ort::init_from(config.device.runtime(&home))?
        .with_name("autotab-midi")
        .with_telemetry(false)
        .build()?;
    println!("[3/4] Computing guitar fingerings...");
    let arrangement = fingering::run(
        &notes,
        args.tuning,
        args.tab_engine,
        &home,
        &environment,
        &config,
    )?;
    let (gp5, gp5_report) = gp5::export_midi(&arrangement, args.tuning, &source, &selected, &stem)?;
    let midi = source.preview(&selected)?;
    let report = serde_json::json!({
        "input_type": "midi", "input": input, "selection": args.midi_track.to_string(),
        "engine": args.tab_engine, "model": matches!(args.tab_engine, fingering::Engine::V3).then_some(fingering::MODEL),
        "model_revision": matches!(args.tab_engine, fingering::Engine::V3).then_some(fingering::REVISION),
        "model_credit": matches!(args.tab_engine, fingering::Engine::V3).then_some("cstr; CC BY 4.0"),
        "tuning_low_to_high": args.tuning.pitches(), "arrangement": arrangement, "gp5": gp5_report,
        "ticks_per_beat": source.ppq, "source_tracks": source.tracks, "tempos": source.tempos,
        "meters": source.meters.iter().map(|m| serde_json::json!({"tick": m.tick, "meter": m.meter.to_string()})).collect::<Vec<_>>(),
        "source_notes": selected.iter().enumerate().map(|(index,n)| serde_json::json!({
            "index":index,"pitch":n.note.pitch,"start_s":n.note.start_s,"end_s":n.note.end_s,
            "start_tick":n.start_tick,"end_tick":n.end_tick,"track":n.track,"channel":n.channel,"port":n.port,
            "velocity":n.velocity,"program":n.program})).collect::<Vec<_>>() });
    println!("[4/4] Writing tablature, Guitar Pro and selected MIDI preview...");
    for path in [&output, &midi_output, &gp5_output, &report_output] {
        tools::create_parent(path)?;
    }
    fs::write(&output, fingering::render(&arrangement))?;
    fs::write(&midi_output, midi)?;
    fs::write(&gp5_output, gp5)?;
    fs::write(&report_output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "Tab: {}\nMIDI: {}\nGuitar Pro: {}\nMIDI conversion complete in {:.2}s",
        output.display(),
        midi_output.display(),
        gp5_output.display(),
        started.elapsed().as_secs_f64()
    );
    if let Some(path) = args.timings_json {
        tools::create_parent(&path)?;
        fs::write(
            path,
            serde_json::to_vec_pretty(
                &serde_json::json!({"input_type":"midi","notes":notes.len(),"total_seconds":started.elapsed().as_secs_f64()}),
            )?,
        )?;
    }
    Ok(())
}

fn validate_outputs(
    input: &std::path::Path,
    outputs: &[&std::path::PathBuf],
    timing: Option<&std::path::Path>,
) -> Result<()> {
    let normalized = |path: &std::path::Path| -> Result<String> {
        let path = path.canonicalize().or_else(|_| std::path::absolute(path))?;
        let text = path.to_string_lossy().into_owned();
        Ok(if cfg!(windows) {
            text.to_lowercase()
        } else {
            text
        })
    };
    let input = normalized(input)?;
    let mut seen = std::collections::BTreeSet::from([input]);
    for path in outputs.iter().copied().map(|p| p.as_path()).chain(timing) {
        ensure!(
            seen.insert(normalized(path)?),
            "Input and all output paths must differ; the source MIDI must not be overwritten"
        );
    }
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    #[test]
    fn parses_muscriptor_rhythm_controls_and_rejects_invalid_meter() {
        let args = Args::try_parse_from([
            "autotab",
            "--input",
            "song.wav",
            "--transcriber",
            "muscriptor-small",
            "--bpm",
            "96",
            "--time-signature",
            "6/8",
            "--quantize",
            "16",
        ])
        .unwrap();
        assert_eq!(args.bpm, 96.0);
        assert_eq!(args.time_signature.to_string(), "6/8");
        assert_eq!(args.quantize, rhythm::Grid::Sixteenth);
        assert!(
            Args::try_parse_from(["autotab", "--input", "song.wav", "--time-signature", "4/3"])
                .is_err()
        );
    }
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

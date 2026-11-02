# AutoTab in Rust

Local audio-to-guitar-tab transcription, **without Python**. The Rust executable
runs the six-stem Demucs ONNX model for guitar separation, Basic Pitch for note
recognition, and the existing string/fret heuristic for ASCII tablature. It also
writes a MIDI file.

Intel Arc acceleration is enabled automatically when the bundled OpenVINO runtime
is installed. Demucs runs directly on the Intel GPU through OpenVINO **2026.4.1**;
the smaller Basic Pitch model stays on CPU. Everything runs locally without Python.

## Run on Windows x64

For the desktop interface, double-click **`run-gui.cmd`** or run:

```powershell
.\run-gui.cmd
```

The German interface offers an audio file picker / drag-and-drop, an optional
audio URL, tuning and Intel Arc/CPU selection, output folder, live status,
cancellation, tab preview, clipboard copy and an output-folder button. Tab and
MIDI files receive a new filename when an earlier result already exists.
The GUI invokes the same native Rust backend in the background; processing and
FFmpeg run without extra console windows. Closing the GUI cancels an active job.

Use **Abspielen**, **Pause / Fortsetzen** and **Stopp** below the preview to listen
to the recognized notes, with elapsed time and a progress indicator. Playback is
a synthesized MIDI preview with the recognized timing. The ASCII tab's spacing
does not encode rhythm, and notes outside the supported fretboard can still be
heard in the MIDI. The Windows MIDI sequencer and installed synthesizer provide
playback without Python or an additional player. Starting another transcription
or closing the GUI stops playback.

**Ergebnis öffnen …** loads a saved `.txt` tab and its matching `.mid` file in the
same folder, so it can be played without analyzing the audio again. Keep both
files with the same filename stem (for example, `song_tab.txt` and `song_tab.mid`).

The command-line interface remains available:

```powershell
.\run-rust.cmd --input "C:\Music\song.mp3"
.\run-rust.cmd --input "C:\Music\song.mp3" --tuning drop_d
.\run-rust.cmd --input "C:\Music\guitar.wav" --skip-separation
.\run-rust.cmd --help
```

Output defaults to `output/<song>_tab.txt` and `output/<song>_tab.mid`.
Use `-o` and `--midi-output` to choose other paths. Intermediate files use a unique
temporary directory and are removed automatically. Concurrent jobs should use
separate output filenames.

Tunings: `standard`, `drop_d`, `d_standard`, `half_step_down`, `eb_standard`.
The tab layout retains the Python version's greedy fret assignment; it does not
preserve rhythmic spacing or guarantee optimal fingering. Notes outside the
supported fretboard may be omitted. MIDI preserves the recognized note timing.

## Setup / rebuild

Requires Windows x64, an internet connection for setup and the Visual Studio C++
build tools with a Windows SDK. Install them before running setup.

```powershell
.\setup-rust.ps1
```

If PowerShell blocks scripts, use a process-local policy:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-rust.ps1
```

Setup installs **Rust 1.92.0** into `.tools`, builds with `Cargo.lock`, and downloads
native FFmpeg 9.0.2, ONNX Runtime 1.23.2 and the six-stem Demucs model. The runtime
archive and Demucs model have checked SHA-256 hashes. Basic Pitch's official ONNX
model is embedded through `basic-pitch-rs`. No Python installation, virtual
environment, PyTorch package or yt-dlp is used by the Rust application or its setup.

The default setup also installs version-pinned OpenVINO 2026.4.1 native GPU DLLs.
Its official binary archive is distributed as a wheel; setup extracts only the
required DLLs and never installs or executes Python. `setup-rust.ps1 -CpuOnly`
skips downloading GPU libraries. Use `--device cpu` to force CPU execution when
GPU libraries are already present.

`rust-toolchain.toml` pins the compiler. `ort` is pinned to 2.0.0-rc.14 and uses the
ONNX Runtime API 22 subset, available in the bundled 1.23.2 runtime. Upgrade these
deliberately together; release candidates are not interchangeable.

```powershell
.\cargo-local.ps1 test --locked
.\cargo-local.ps1 fmt --check
.\cargo-local.ps1 @('clippy', '--locked', '--', '-D', 'warnings')
.\cargo-local.ps1 build --release --locked
.\cargo-local.ps1 build --release --locked --features gui
.\test-rust.ps1
.\test-rust.ps1 -Device openvino
.\cargo-local.ps1 @('test', '--release', '--test', 'arc', '--', '--ignored', '--nocapture')
```

`setup-rust.ps1` builds both executables with the `gui` feature. A CLI-only build
does not need the optional desktop libraries. The GUI uses eframe/egui 0.33.3 and
native rfd 0.15.4 file dialogs, compatible with the pinned Rust 1.92.0 toolchain.

GUI checks (the ignored process tests require the installed native runtimes;
the playback test requires the Windows MIDI sequencer and uses a silent fixture):

```powershell
.\cargo-local.ps1 test --release --locked --features gui
.\cargo-local.ps1 @('test', '--release', '--locked', '--features', 'gui', '--bin', 'autotab-gui', '--', '--ignored', '--test-threads=1')
```

## Intel Arc

```powershell
# Default: use the installed OpenVINO GPU runtime
.\run-rust.cmd --input "C:\Music\song.mp3"
# Explicit Intel GPU selection (GPU.0 by default)
.\run-rust.cmd --input "C:\Music\song.mp3" --device openvino --gpu-id 0
# CPU reference
.\run-rust.cmd --input "C:\Music\song.mp3" --device cpu
# Compare CPU and Arc on the same recording
.\benchmark-arc.ps1 -InputAudio "C:\Music\song.mp3"
```

The GPU path uses one latency-oriented stream, reuses input buffers, and copies
only the guitar channels into the overlap-add buffer. FP32 is the default to retain
transcription accuracy. `--precision fp16` is an opt-in speed/accuracy tradeoff and
can change recognized notes.

The first GPU run compiles Demucs and can take a minute or more. Compiled models
are cached in `.cache/openvino/`, keyed by model contents, runtime version, GPU,
driver and precision. Later processes import that compiled model directly.
The log names the actual GPU, verifies the selected precision, and reports
`cached true` on a cache hit. An explicit GPU request fails visibly if it cannot
run on the GPU. OpenVINO device indices are independent of Windows display order;
use `--gpu-id 1` for a second Intel GPU if necessary.

`--profile-dir <directory>` writes the GPU/device/cache metadata and ONNX traces
for CPU inference. `--timings-json <file>` records pipeline timings;
`--guitar-output <file.wav>` retains the separated audio for quality checks.
With `--skip-separation`, only CPU Basic Pitch is needed.

The local FP32 benchmark processed nine seconds of MP3 in **4.49 s on Arc versus
14.40 s on CPU** (median of three runs, warm GPU cache). Hardware, methodology
and accuracy checks are recorded in [docs/intel-arc.md](docs/intel-arc.md).

## URLs

```powershell
.\run-rust.cmd --url "https://example.org/song.mp3"
.\run-rust.cmd --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

Direct HTTP(S) media URLs are downloaded in Rust. Public YouTube URLs use the native
Rust `rusty_ytdl` library. YouTube availability depends on its current access rules;
private, login-protected, age-restricted or blocked videos may fail. The program
reports a download error instead of falling back to Python. Download such audio
separately and use `--input`.

**Current limitation:** the public YouTube test returned `Video source empty`.
YouTube extraction is experimental and is not confirmed working in this environment.
Local media transcription and direct HTTP downloads have passed their tests.

## Validation

Five Rust tests cover chord grouping, tunings, overlap-add boundaries, direct HTTP
download and URL scheme validation. `cargo clippy --all-targets -- -D warnings` and
format checks pass. `test-rust.ps1` generates a nine-second MP3, exercises two Demucs
chunks and clean-guitar transcription, and checks the resulting MIDI headers and
tab files. The complete pipeline uses only the Rust executable and native FFmpeg /
ONNX Runtime binaries.

The opt-in Arc integration test additionally checks GPU execution, effective FP32,
cache reuse between processes, and CPU/GPU sample accuracy on a generated plucked
tone. `benchmark-arc.ps1` performs one GPU warm-up and alternates CPU/GPU runs;
it writes measurements, GPU metadata and a median summary to `output/arc-benchmark/`.

## Runtime layout

```text
target/release/autotab.exe
target/release/autotab-gui.exe
runtime/ffmpeg.exe
runtime/onnxruntime.dll
runtime/onnxruntime_providers_shared.dll
runtime/openvino-2026.4.1/*.dll
models/htdemucs_6s.onnx
```

Run the executable from the repository or set `AUTOTAB_HOME` to the directory
containing `runtime/` and `models/`. Processing and inference stay local after any
requested URL download. Runtime assets are ignored by Git; the model is about
246 MiB and separation can use substantial RAM. `--threads` controls CPU inference
threads; the GPU uses one stream. Audio is held in memory, so memory
usage also grows with recording length.

This repository contains the standalone Rust implementation. The original project is
[AutoTab by bmcdonald3](https://github.com/bmcdonald3/AutoTab). Python sources and
history, downloaded models, native runtime binaries and generated audio / results
are excluded. Setup downloads the required runtime assets locally.

## Sources and licenses

- [Demucs](https://github.com/facebookresearch/demucs): MIT.
- [Six-stem ONNX export and tensor specification](https://huggingface.co/StemSplitio/htdemucs-6s-onnx): MIT; pinned model revision `49df9b6989cf2150840ea65b0bef77a2e471b678`.
- [basic-pitch-rs](https://github.com/AThevon/basic-pitch-rs): Apache-2.0, includes the official Spotify Basic Pitch model and Rust post-processing.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): MIT.
- [OpenVINO](https://github.com/openvinotoolkit/openvino) and [Rust bindings](https://github.com/intel/openvino-rs): Apache-2.0.
- [FFmpeg Windows essentials build](https://www.gyan.dev/ffmpeg/builds/): see the build's bundled license; this distribution includes GPL components.
- [rusty_ytdl](https://github.com/Mithronn/rusty_ytdl): MIT.

# AutoTab in Rust

Local audio/MIDI-to-guitar-tab transcription, **without Python**. Existing MIDI
files go directly to fingering search, without audio decoding or transcription.
For audio, the Rust executable
runs the six-stem Demucs ONNX model for guitar separation, Basic Pitch for note
recognition, and a pretrained tab-labeler v3 model plus constrained fingering
search. It writes ASCII tablature, playable Guitar Pro 5 notation, MIDI, and a
fingering/verification report. No local model training is required.

Note recognition can use **Basic Pitch**, **MuScriptor Small**, **MuScriptor Medium**
or **MuScriptor Large**. All four run natively from Rust without a Python process.

Intel Arc acceleration is enabled automatically when the bundled OpenVINO runtime
is installed. Demucs runs directly on the Intel GPU through OpenVINO **2026.4.1**;
the smaller Basic Pitch model stays on CPU. Everything runs locally without Python.

## Run on Windows x64

For the desktop interface, double-click **`run-gui.cmd`** or run:

```powershell
.\run-gui.cmd
```

The German interface offers an audio/MIDI file picker / drag-and-drop, an optional
audio URL, tuning and Intel Arc/CPU selection, output folder, live status,
cancellation, tab preview, clipboard copy, a **Griffpositionen** model selector,
and **Guitar Pro öffnen** for generated GP5 files. All four output files receive
a new filename stem when an earlier result already exists.
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
ASCII columns show onset groups, not rhythmic spacing. GP5 carries actual rhythmic
notation with the assigned strings/frets; MIDI retains recognized timing unless
quantization is selected. Notes outside the fretboard are explicitly reported and
retained in MIDI and the JSON source-note list.

## Existing MIDI files

Select a `.mid` or `.midi` under **Datei** (or drag it into the window), choose
**MIDI-Spur**, keep **KI-Modell v3**, and click **Tabs erstellen**. MIDI skips
FFmpeg, Demucs, Basic Pitch and MuScriptor entirely. Only the small tab-labeler
model and native ONNX Runtime are needed. Audio-specific options are hidden or
disabled; tempo, meter and note timing come from the file.

**Automatisch** selects notes on General MIDI guitar programs 24..31 (zero-based).
If none are present, it uses all melodic notes. **Alle melodischen Spuren** and
individual named tracks can also be selected. Drum channel 10 is always excluded.
Track numbers are the zero-based physical SMF track indices; a type-0 file may
contain multiple MIDI channels inside its single track. MIDI ports and per-channel
program changes are respected when selecting guitar notes.

```powershell
# Inspect track names and note counts; no model inference or download:
.\run-rust.cmd --input "C:\Music\song.mid" --list-midi-tracks
# Automatic guitar selection:
.\run-rust.cmd --input "C:\Music\song.mid" --tab-engine v3
# Select physical track 1 (track 0 is often the conductor/tempo track):
.\run-rust.cmd --input "C:\Music\song.mid" --midi-track 1 --tab-engine v3
# Include every non-drum note:
.\run-rust.cmd --input "C:\Music\song.mid" --midi-track all
```

Outputs are the same `.txt`, `.gp5`, `.mid` and `.fingering.json` files. The MIDI
preview contains only the selected notes, with their original ticks, velocities,
tempo/meter maps and selected-channel controls. The input file is left unchanged;
an output path resolving to the input is rejected. JSON records original ticks,
tracks, channels, ports, programs and the fingering assignments.

GP5 uses original musical beat positions, including tempo changes and meter
changes at measure boundaries. As for audio, GP5 timing is quantized to 1/32 and
BPM to integers. The MIDI preview retains original timing. Pitch bends and other
controllers are retained in the MIDI preview but are not converted to GP5 playing
techniques. Guitar fingering considers note-on/off spans, not sustain-pedal tails.

Supported input: standard SMF types 0 and 1 with metrical PPQ timing. Type 2,
SMPTE timing and zero-length notes are rejected explicitly. Missing note-offs
are closed at track end and reported. The current GP5 writer requires initial
tempo 4..1000 BPM, subsequent tempo changes 1..255 BPM, and meter changes at
measure boundaries; unsupported changes fail visibly rather than changing the
score silently. `--bpm`, `--time-signature` and `--quantize` are audio-only controls
when a MIDI file is imported.

## Fingering and Guitar Pro export

The default **KI-Modell v3** / `--tab-engine v3` uses
[`cstr/tab-labeler-onnx`](https://huggingface.co/cstr/tab-labeler-onnx), file
`tab-labeler-v3-egset.onnx` (GuitarSet + EGSet12, ~1.3 MB). The first nonempty
guitar result downloads revision `434976baacaf194d99f09c3c784682766bedcf97`,
verified against SHA-256
`93114786c6f4781e7a0b69abd7e93a0d4966873f8e54908f97a1de8e8cf788af`.
Weights are cached in `models/`; subsequent inference is local on CPU through
the existing ONNX Runtime. Credit: **cstr**; weights: **CC BY 4.0**. Model credit
and license are retained in the text, JSON and GP5 metadata.

The model scores candidate positions; a 64-state beam search penalizes hand
movement and fret spread. It enforces one sounding note per string, a maximum
seven-fret span (excluding open strings), and frets 0..19. Excess chord notes or
notes blocked by sustained strings are preserved as additional guitar voices.
Grips spanning more than four frets are reported for manual review. Fingering
is approximate and does not guarantee comfortable playing or original artist
fingerings. The model was trained for standard tuning; the other tunings are
experimental. Pitches outside its 40..88 input range use rule-based costs where
the requested fretboard can represent them.

Choose **Regelbasierte Griffsuche** / `--tab-engine simple` for the same bounded
search without model scores or a weight download:

```powershell
.\run-rust.cmd --input "C:\Music\guitar.wav" --skip-separation --tab-engine v3 --bpm 120 --time-signature 4/4
.\run-rust.cmd --input "C:\Music\guitar.wav" --skip-separation --tab-engine simple
```

For `song_tab.txt`, the additional files are `song_tab.mid`, `song_tab.gp5`, and
`song_tab.fingering.json`. Double-click the `.gp5` or use **Guitar Pro öffnen**
in the GUI. GP5 preserves explicit fingerings and tied notes across bars. It
rounds timing to a 1/32 grid (or a selected coarser grid) and tempo to integer BPM.
Quantization collisions are kept on additional GP5 tracks rather than losing
attacks. The JSON reports timing displacement, additional voices, wide grips,
unassigned source indices, and every source note. GP5 is read back before saving
to verify note attacks, string/fret assignments and quantized start/end times.
MIDI retains non-guitar instruments; the GP5 contains the guitar arrangement.

Tempo/meter/grid controls are available for both Basic Pitch and MuScriptor.
Tempo and downbeats are manual settings; there is no automatic beat tracking.

Opt-in native model/export smoke test (requires the local runtime and internet
on the first model download):

```powershell
.\cargo-local.ps1 @('test', '--locked', '--features', 'gui', '--bin', 'autotab', 'native_v3_gp5_roundtrip', '--', '--ignored', '--nocapture')
```

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

## MuScriptor models

Choose **Notenerkennung** in the GUI, or select a model in the CLI:

```powershell
.\run-rust.cmd --input "C:\Music\song.mp3" --transcriber muscriptor-small
.\run-rust.cmd --input "C:\Music\song.mp3" --transcriber muscriptor-medium
.\run-rust.cmd --input "C:\Music\song.mp3" --transcriber muscriptor-large
# Existing local checkpoint (select its matching size):
.\run-rust.cmd --input "C:\Music\guitar.wav" --skip-separation --transcriber muscriptor-small --muscriptor-model "C:\Models\muscriptor-small\model.safetensors"
```

Basic Pitch remains the default (`--transcriber basic-pitch`). The selected model
replaces note recognition; Demucs guitar separation, tuning, tabs and MIDI playback
remain available. MuScriptor defaults to the acoustic / clean electric / distorted
electric guitar groups. Its notes use fixed MIDI velocity, without estimated
dynamics or pitch bends. Fret assignment still uses the tab heuristic.

### MIDI tempo, time signature and note grid

For MuScriptor, the GUI offers **Tempo (BPM)**, **Taktart** and **Notenraster**.
Tempo is measured in **quarter notes per minute**, including in meters such as
6/8. The time-signature numerator can be 1–32; the denominator can be 1, 2, 4, 8,
16, 32 or 64. Examples are 4/4, 3/4, 5/4 and 6/8.

```powershell
.\run-rust.cmd --input "C:\Music\song.wav" --transcriber muscriptor-small --bpm 96 --time-signature 6/8 --quantize sixteenth
# Eighth-note grid (aliases: 8 or 1/8):
.\run-rust.cmd --input "C:\Music\song.wav" --transcriber muscriptor-small --bpm 120 --time-signature 4/4 --quantize eighth
```

`--quantize off` is the default: the detected note times in seconds are preserved
to MIDI tick resolution when changing tempo or meter metadata. `eighth` / `8`
snaps note starts and ends to an eighth-note grid; `sixteenth` / `16` uses a
sixteenth-note grid. Longer notes can span several grid steps. Quantized notes
last at least one step; same-pitch notes that collapse to one onset within the
same instrument are merged, and overlapping repeated notes are trimmed to avoid
conflicting note-on/off pairs. All instrument tracks use the same grid.

These are manual MIDI export settings, independent of model inference. No
automatic tempo/downbeat tracking is added. The grid starts at the beginning of
the recording; changing BPM alone does not time-stretch the audio or MIDI.
The MIDI records tempo and meter explicitly. ASCII tablature keeps its existing
layout and does not display rhythmic notation. Default: 120 BPM, 4/4, grid off.

### Expected instruments and multi-instrument MIDI

In the GUI, use **Erwartete Instrumente** to select multiple instrument groups, or
enable **Automatisch erkennen** to let MuScriptor identify the instrumentation.
The **Klavier, Gitarre, Bass** preset corresponds to the published example, group
IDs `0 4 7`. The list contains all named groups from `MT3_FULL_PLUS_GROUP_NAMES`.

```powershell
.\run-rust.cmd --input "C:\Music\mix.wav" --transcriber muscriptor-small --instruments "acoustic_piano,acoustic_guitar,acoustic_bass"
# Equivalent numeric group IDs (also accepted with --instrument-group):
.\run-rust.cmd --input "C:\Music\mix.wav" --transcriber muscriptor-small --instrument-group "0 4 7"
# Infer the instrument groups without a preset:
.\run-rust.cmd --input "C:\Music\mix.wav" --transcriber muscriptor-small --instruments auto
```

Names or IDs may be separated by commas or spaces. Unknown groups and an empty
manual selection are rejected. Group IDs are model classes, not General MIDI
program numbers: `4` means acoustic guitar, whose representative GM program is 24.
The selection supplies the learned conditioning and restricts program/drum tokens
to the selected groups. Automatic mode uses null instrument conditioning and
allows all valid instrument tokens.

**MIDI retains all recognized instruments**, including drums, on separate named
tracks with their General MIDI programs. **ASCII tabs contain only guitar notes**;
piano or bass notes are not assigned to guitar strings. If a result contains only
other instruments, the MIDI still contains those notes and the tab has no guitar
notes. More than 15 melodic programs use additional MIDI-port metadata to avoid
sharing conflicting program changes on the same channel; playback of these ports
depends on the MIDI player/device.

Automatic mode or any selected non-guitar group uses the **original mix**, without
Demucs guitar separation, so the other instruments remain audible to the model.
Guitar-only selection keeps the existing optional Demucs path. A conditioning
choice guides recognition; it does not separate an instrument's audio waveform.

| Model | Weight download | Pinned Hugging Face revision |
| --- | ---: | --- |
| Small | 0.41 GB | `8c127f603b807520fa465c838e9bfee8a91ada4e` |
| Medium | 1.23 GB | `f32236969308476e01fd3aae67357de5feb05a2d` |
| Large | 5.47 GB | `8809fdfbed2affa7ade94a7059e746e3880720e7` |

Model access is gated. Accept the conditions separately on the
[Small](https://huggingface.co/MuScriptor/muscriptor-small),
[Medium](https://huggingface.co/MuScriptor/muscriptor-medium) and
[Large](https://huggingface.co/MuScriptor/muscriptor-large) model pages. Downloads
can use the **Hugging-Face-Token** field shown beneath **Notenerkennung** after
selecting a MuScriptor model. The field is masked, with optional **Anzeigen** and
**Leeren** controls and a link to create/manage tokens. The entered token is kept
only for the current GUI session and passed to the backend process through its
`HF_TOKEN` environment variable, never as a command-line argument. Output from
that process redacts the entered token. AutoTab never saves it to a file.

An empty field uses the existing authentication: `HF_TOKEN`, or an existing Hugging
Face token file (`HF_TOKEN_PATH` or `HF_HOME/token`). The supplied token must belong
to an account with access to the selected model. Entering a token does not grant
model access automatically.
The weights are **CC BY-NC 4.0**; the upstream inference code is MIT-licensed.
No weights or credentials are included in this repository.

HTTPS downloads use the operating system's trusted root certificates (the Windows
certificate store on Windows), including organization-managed roots. TLS
certificate and hostname validation remain enabled. An `UnknownIssuer` error
indicates a certificate trust problem; HTTP 401/403 instead indicates model access
or authentication. No insecure TLS bypass is used.

AutoTab first checks its `models/` folder and the standard Hugging Face snapshot
cache, respecting `HF_HOME`, `HF_HUB_CACHE` and `HUGGINGFACE_HUB_CACHE`. Missing
weights download automatically on first use. Alternatively, select a local
`.safetensors` file under **Weitere Einstellungen** in the GUI. The official F32
checkpoints are supported, including the legacy single-codebook weight names.
Quantized or fine-tuned checkpoints with different architectures are not supported.

The Rust code builds OpenVINO IR directly from the checkpoint; no export script,
Python, PyTorch or local model server is needed. Audio is resampled to 16 kHz and
processed in five-second segments. The checkpoint's own Hann window and HTK mel
filterbank are used. Autoregressive decoding keeps its KV state on the selected
device; tie prompts preserve held notes across segments. `--device openvino`
uses the selected Intel Arc GPU, while `--device cpu` uses the native OpenVINO CPU
plugin. Both MuScriptor paths require the default `setup-rust.ps1` runtime; a
previous `-CpuOnly` installation needs the normal setup once.

The first run creates `.cache/muscriptor/`, including an additional full weight
copy and compiled-device caches. Allow additional disk space and system / GPU RAM,
especially for Large. First compilation is slower than later cached starts.

The native Small smoke test covers two audio segments on Intel Arc without
Python. Medium and Large share the parameterized implementation and are selectable,
but their actual checkpoints have not been exercised here: Hugging Face returned
HTTP 403 for the available credentials. Their downloads require model access or
a local checkpoint.

```powershell
.\cargo-local.ps1 @('test', '--release', '--locked', '--bin', 'autotab', 'native_small_transcribes_without_python', '--', '--ignored', '--nocapture')
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

Rust tests cover model input/scoring, constrained fingerings, extra voices, GP5
ties and rhythmic values, multi-instrument MIDI, overlap-add boundaries, direct
HTTP download and URL scheme validation. `cargo clippy --all-targets -- -D warnings` and
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
- [MuScriptor](https://github.com/muscriptor/muscriptor): MIT code; the native implementation adapts its architecture, preprocessing and event decoder. The [MIT notice](docs/licenses/muscriptor-MIT.txt) is retained. Hugging Face model weights are separately gated and CC BY-NC 4.0.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): MIT.
- [tab-labeler v3](https://huggingface.co/cstr/tab-labeler-onnx): cstr, CC BY 4.0; trained on GuitarSet and EGSet12.
- [guitarpro Rust library](https://codeberg.org/slundi/scorelib): MIT; native GP5 encoding and reading.
- [OpenVINO](https://github.com/openvinotoolkit/openvino) and [Rust bindings](https://github.com/intel/openvino-rs): Apache-2.0.
- [FFmpeg Windows essentials build](https://www.gyan.dev/ffmpeg/builds/): see the build's bundled license; this distribution includes GPL components.
- [rusty_ytdl](https://github.com/Mithronn/rusty_ytdl): MIT.

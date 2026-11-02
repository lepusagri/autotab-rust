# Intel Arc validation — 2026-11-02

The Rust version runs Demucs directly on the Intel GPU with OpenVINO 2026.4.1
(build 22982). Basic Pitch runs on CPU. Python is not used.

Test machine: **Intel Arc Graphics (iGPU), GPU.0**, Windows driver
**32.0.101.8860**. Rust **1.92.0**, release build with thin LTO. Demucs uses one GPU
stream, FP32 inference and the accuracy execution hint. CPU inference uses four
threads. CPU ONNX Runtime remains pinned to 1.23.2.

## End-to-end benchmark

Input: a generated nine-second A2 sine tone encoded as MP3, processed through two
overlapping Demucs segments, resampling, Basic Pitch, MIDI and ASCII tab output.
There was one GPU warm-up, followed by three alternating CPU/GPU runs. GPU runs
imported a compiled model from the persistent cache. Profiling was enabled for
both backends in the measured runs.

| Backend | Median total time |
| --- | ---: |
| CPU | 14.40 s |
| Intel Arc, FP32, warm cache | 4.49 s |
| End-to-end speedup | **3.20×** |

GPU model initialization took about one second with the compiled cache. Each
7.8-second padded Demucs segment took about 1.2 seconds on the GPU. This result is
specific to this machine and input; recordings of different lengths and other Arc
cards can have different speedups.

The first compilation is substantially slower and is excluded from this warm-cache
comparison. It can take a minute or more. Cache keys include the model contents,
OpenVINO version, GPU name/index, driver and precision. Cache files are written
through a temporary file and then renamed. Failed imports rebuild the cache.

Raw measurements and hardware metadata are generated in
`output/arc-benchmark/summary.json`; per-run execution metadata is in the adjacent
`*-profile/demucs-openvino.json` files. The GPU path verifies `EXECUTION_DEVICES`
and the effective inference precision before inference.

## Accuracy and regression checks

The opt-in hardware integration test uses a separate generated nine-second
plucked harmonic A2 signal. It validates actual GPU execution, effective FP32,
cache reuse between processes, equal output lengths and sample error against CPU.
Measured maximum absolute sample error was about **1.1e-7** of full scale, with
**66.75 dB SNR** between the CPU signal and its GPU error. Both runs recognized
15 notes in this fixture. These are fixture measurements, not a general guarantee
of identical note decisions at every model threshold.

```powershell
.\cargo-local.ps1 @('test', '--release', '--locked', '--test', 'arc', '--', '--ignored', '--nocapture')
.\benchmark-arc.ps1 -InputAudio .\temp\rust-smoke\tone.mp3
```

Five ordinary Rust tests cover tablature/tunings, overlap-add boundaries and HTTP
download behavior. The hardware test is opt-in so these checks can run on machines
without an Intel GPU. `test-rust.ps1 -Device auto` additionally exercises MP3 input
and clean-guitar transcription through the normal automatic backend selection.

FP32 is the supported default. FP16 is optional and can change transcription
decisions; the accuracy results above apply to FP32.

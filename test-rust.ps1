param([ValidateSet('cpu', 'auto', 'openvino')][string]$Device = 'cpu')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$testDirectory = Join-Path $PSScriptRoot 'temp/rust-smoke'
New-Item -ItemType Directory -Force $testDirectory | Out-Null
$audio = Join-Path $testDirectory 'tone.mp3'
& ./runtime/ffmpeg.exe -hide_banner -loglevel error -y -f lavfi -i 'sine=frequency=110:duration=9' -c:a libmp3lame $audio
if ($LASTEXITCODE -ne 0) { throw 'Could not generate test audio' }

foreach ($mode in @('separated', 'clean')) {
    $tab = Join-Path $testDirectory "$mode.txt"
    $parameters = @('--input', $audio, '--device', $Device, '-o', $tab)
    if ($mode -eq 'clean') { $parameters += '--skip-separation' }
    & ./target/release/autotab.exe @parameters
    if ($LASTEXITCODE -ne 0) { throw "Transcription failed: $mode" }
    if ((Get-Item $tab).Length -eq 0) { throw "Empty tab: $mode" }
    $midi = [IO.File]::ReadAllBytes([IO.Path]::ChangeExtension($tab, 'mid'))
    if ($midi.Length -lt 14 -or [Text.Encoding]::ASCII.GetString($midi, 0, 4) -ne 'MThd') {
        throw "Invalid MIDI: $mode"
    }
    if ($mode -eq 'clean' -and (Get-Content $tab -Raw) -notmatch '(?m)^E\|.*\d') {
        throw 'Expected low guitar note missing in clean A2 transcription'
    }
}
Write-Host 'Native MP3 -> guitar -> MIDI/tab and clean-guitar tests passed.'

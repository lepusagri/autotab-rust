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
    $gp5 = [IO.File]::ReadAllBytes([IO.Path]::ChangeExtension($tab, 'gp5'))
    if ($gp5.Length -lt 100 -or [Text.Encoding]::ASCII.GetString($gp5, 0, 30) -notmatch 'FICHIER GUITAR PRO') {
        throw "Invalid Guitar Pro file: $mode"
    }
    $report = Get-Content ([IO.Path]::ChangeExtension($tab, 'fingering.json')) -Raw | ConvertFrom-Json
    if ($mode -eq 'clean' -and -not ($report.arrangement.notes | Where-Object { $_.pitch -eq 45 })) {
        throw 'Expected A2 missing in clean guitar arrangement'
    }
}
Write-Host 'Native MP3 -> guitar -> MIDI/tab/GP5 and clean-guitar tests passed.'

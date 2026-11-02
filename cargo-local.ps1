$ErrorActionPreference = 'Stop'
$env:RUSTUP_HOME = Join-Path $PSScriptRoot '.tools/rustup'
$env:CARGO_HOME = Join-Path $PSScriptRoot '.tools/cargo'
$env:PATH = "$env:CARGO_HOME/bin;$env:PATH"
Push-Location $PSScriptRoot
try {
    & "$env:CARGO_HOME/bin/cargo.exe" @args
    if ($LASTEXITCODE -ne 0) { throw "Cargo failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

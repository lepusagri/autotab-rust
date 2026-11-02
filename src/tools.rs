use anyhow::{Context, Result, bail};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

pub fn project_home() -> Result<PathBuf> {
    if let Some(home) = env::var_os("AUTOTAB_HOME") {
        return Ok(PathBuf::from(home));
    }
    let executable = env::current_exe()?;
    for directory in executable.ancestors().skip(1) {
        if directory.join("runtime/onnxruntime.dll").is_file() {
            return Ok(directory.to_path_buf());
        }
    }
    let current = env::current_dir()?;
    if current.join("Cargo.toml").is_file() {
        return Ok(current);
    }
    bail!("Cannot find AutoTab runtime. Set AUTOTAB_HOME to the project or distribution directory.")
}

pub fn find_ffmpeg(home: &Path) -> Result<PathBuf> {
    let bundled = home.join("runtime/ffmpeg.exe");
    if bundled.is_file() {
        return Ok(bundled);
    }
    for directory in env::split_paths(&env::var_os("PATH").unwrap_or_default()) {
        let path = directory.join("ffmpeg.exe");
        if path.is_file() {
            return Ok(path);
        }
    }
    bail!("FFmpeg missing. Run setup-rust.ps1 first.")
}

pub fn create_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("Cannot create {}", parent.display()))?;
    }
    Ok(())
}

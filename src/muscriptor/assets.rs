use super::{Variant, weights::Weights};
use anyhow::{Context, Result, ensure};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

fn hub_home() -> Option<PathBuf> {
    std::env::var_os("HF_HOME").map(PathBuf::from).or_else(|| {
        std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(|p| PathBuf::from(p).join(".cache/huggingface"))
    })
}

pub fn find(variant: Variant, home: &Path, override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        let path = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.to_path_buf()
        };
        ensure!(
            path.is_file(),
            "MuScriptor model not found: {}",
            path.display()
        );
        return Ok(path);
    }
    let (size, revision, expected_bytes) = variant.spec();
    let local = home.join(format!(
        "models/muscriptor-{size}/{revision}/model.safetensors"
    ));
    if local.is_file() {
        return Ok(local);
    }
    let hub = std::env::var_os("HF_HUB_CACHE")
        .or_else(|| std::env::var_os("HUGGINGFACE_HUB_CACHE"))
        .map(PathBuf::from)
        .or_else(|| hub_home().map(|p| p.join("hub")));
    if let Some(hub) = hub {
        let cached = hub.join(format!(
            "models--MuScriptor--muscriptor-{size}/snapshots/{revision}/model.safetensors"
        ));
        if cached.is_file() {
            return Ok(cached);
        }
    }
    let token = std::env::var("HF_TOKEN")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            let path = std::env::var_os("HF_TOKEN_PATH")
                .map(PathBuf::from)
                .or_else(|| hub_home().map(|p| p.join("token")))?;
            fs::read_to_string(path)
                .ok()
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
        });
    let url = format!(
        "https://huggingface.co/MuScriptor/muscriptor-{size}/resolve/{revision}/model.safetensors"
    );
    println!(
        "  Downloading MuScriptor {size} ({:.2} GB). Model license: CC BY-NC 4.0",
        expected_bytes as f64 / 1e9
    );
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(7200))
        .build()?;
    let mut request = client.get(url);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    let mut response = request.send().context("MuScriptor download failed")?;
    ensure!(
        response.status().is_success(),
        "MuScriptor {size} download: HTTP {}. Accept the model conditions at https://huggingface.co/MuScriptor/muscriptor-{size}, then log in with Hugging Face or set HF_TOKEN. A local --muscriptor-model path also works.",
        response.status().as_u16()
    );
    let parent = local.parent().context("Missing model directory")?;
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    let mut buffer = vec![0; 1024 * 1024];
    let mut bytes = 0_u64;
    let mut next_report = 256 * 1024 * 1024;
    loop {
        let count = response.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes += count as u64;
        ensure!(
            bytes <= expected_bytes,
            "Unexpected MuScriptor download size"
        );
        temporary.write_all(&buffer[..count])?;
        if bytes >= next_report {
            println!(
                "  MuScriptor download: {:.0}%",
                bytes as f64 / expected_bytes as f64 * 100.0
            );
            next_report += 256 * 1024 * 1024;
        }
    }
    ensure!(bytes == expected_bytes, "Incomplete MuScriptor download");
    temporary.flush()?;
    Weights::open(temporary.path())?;
    temporary.persist_noclobber(&local).map_err(|e| e.error)?;
    Ok(local)
}

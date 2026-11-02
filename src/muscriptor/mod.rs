//! Native MuScriptor transcription. No Python or PyTorch runtime is used.
mod assets;
mod graph;
mod mel;
mod runtime;
mod tokens;
mod weights;

use crate::inference::Config;
use anyhow::{Result, ensure};
use basic_pitch::NoteEvent;
use clap::ValueEnum;
use std::path::Path;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum Transcriber {
    #[default]
    BasicPitch,
    MuscriptorSmall,
    MuscriptorMedium,
    MuscriptorLarge,
}

#[derive(Clone, Copy, Debug)]
pub enum Variant {
    Small,
    Medium,
    Large,
}

#[derive(Clone, Copy)]
pub struct Architecture {
    pub dim: usize,
    pub heads: usize,
    pub layers: usize,
    pub card: usize,
}

impl Transcriber {
    pub fn variant(self) -> Option<Variant> {
        match self {
            Self::BasicPitch => None,
            Self::MuscriptorSmall => Some(Variant::Small),
            Self::MuscriptorMedium => Some(Variant::Medium),
            Self::MuscriptorLarge => Some(Variant::Large),
        }
    }
}

impl Variant {
    pub fn spec(self) -> (&'static str, &'static str, u64) {
        match self {
            Self::Small => (
                "small",
                "8c127f603b807520fa465c838e9bfee8a91ada4e",
                411_888_600,
            ),
            Self::Medium => (
                "medium",
                "f32236969308476e01fd3aae67357de5feb05a2d",
                1_228_144_472,
            ),
            Self::Large => (
                "large",
                "8809fdfbed2affa7ade94a7059e746e3880720e7",
                5_465_642_136,
            ),
        }
    }
    fn architecture(self) -> Architecture {
        match self {
            Self::Small => Architecture {
                dim: 768,
                heads: 12,
                layers: 14,
                card: 1393,
            },
            Self::Medium => Architecture {
                dim: 1024,
                heads: 16,
                layers: 24,
                card: 1395,
            },
            Self::Large => Architecture {
                dim: 1536,
                heads: 24,
                layers: 48,
                card: 1395,
            },
        }
    }
}

pub fn transcribe(
    variant: Variant,
    audio: &[f32],
    home: &Path,
    override_path: Option<&Path>,
    config: &Config,
) -> Result<Vec<NoteEvent>> {
    ensure!(
        !audio.is_empty() && audio.iter().all(|s| s.is_finite()),
        "Empty or invalid MuScriptor audio"
    );
    let path = assets::find(variant, home, override_path)?;
    println!("  MuScriptor checkpoint: {}", path.display());
    let weights = weights::Weights::open(&path)?;
    let architecture = variant.architecture();
    let embeddings = weights.read("emb.weight")?;
    ensure!(
        weights.entry("emb.weight")?.shape == [architecture.card + 1, architecture.dim],
        "Checkpoint does not match selected MuScriptor size"
    );
    let mel = mel::Mel::new(&weights)?;
    let files = graph::build(&weights, architecture, &home.join(".cache/muscriptor"))?;
    let runtime = runtime::Runtime::new(&files, config)?;
    let mut encoder = runtime::Request::new(&runtime.encoder)?;
    let mut decoder = tokens::Decoder::default();
    let dim = architecture.dim;
    let chunks = audio.len().div_ceil(80_000);
    let duration = audio.len() as f64 / 16_000.0;
    let mut classes = Vec::with_capacity(dim * 4);
    // ClassConditioner tokenization adds one, then its forward adds one again.
    // Guitar groups 4,5,6 therefore select rows 6,7,8; the null dataset uses row 1.
    for row in [6, 7, 8] {
        classes.extend(weights.row(
            "condition_provider.conditioners.instrument_group.embed.weight",
            row,
            dim,
        )?);
    }
    classes.extend(weights.row(
        "condition_provider.conditioners.dataset_name.embed.weight",
        1,
        dim,
    )?);
    for chunk in 0..chunks {
        let started = std::time::Instant::now();
        let offset = chunk * 80_000;
        let count = (audio.len() - offset).min(80_000);
        let mut wav = vec![0.0; 80_000];
        wav[..count].copy_from_slice(&audio[offset..offset + count]);
        let mel = mel.compute(&wav);
        let projected = encoder.run(&[(&[1, 501, 512], &mel)])?;
        ensure!(
            projected.len() == 501 * dim,
            "Invalid MuScriptor encoder output"
        );
        decoder.chunk(chunk as f64 * 5.0, ((chunk + 1) as f64 * 5.0).min(duration));
        let prompt = if chunk == 0 {
            Vec::new()
        } else {
            decoder.prompt()
        };
        for &token in &prompt {
            decoder.feed(token);
        }
        let mut features = classes.clone();
        features.extend(projected);
        features
            .extend_from_slice(&embeddings[architecture.card * dim..(architecture.card + 1) * dim]);
        for &token in &prompt {
            features.extend_from_slice(&embeddings[token * dim..(token + 1) * dim]);
        }
        let mut position = features.len() / dim;
        add_positions(&mut features, dim, 0);
        let mut mask = causal_mask(position);
        let mut request = runtime::Request::new(&runtime.decoder)?; // Fresh KV state per audio chunk.
        let mut logits = request.run(&[
            (&[1, position as i64, dim as i64], &features),
            (&[1, 1, position as i64, position as i64], &mask),
        ])?;
        let mut ended = false;
        for step in prompt.len()..2000 {
            ensure!(
                logits.len() == architecture.card && logits.iter().all(|v| v.is_finite()),
                "Invalid MuScriptor logits"
            );
            let token = logits
                .iter()
                .enumerate()
                .filter(|&(token, _)| token < 1393 && tokens::allowed(token))
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .map(|(index, _)| index)
                .unwrap();
            if token == 1 {
                ended = true;
                break;
            }
            decoder.feed(token);
            if step == 1999 {
                break;
            }
            features.clear();
            features.extend_from_slice(&embeddings[token * dim..(token + 1) * dim]);
            add_positions(&mut features, dim, position);
            position += 1;
            mask.clear();
            mask.resize(position, 0.0);
            logits = request.run(&[
                (&[1, 1, dim as i64], &features),
                (&[1, 1, 1, position as i64], &mask),
            ])?;
        }
        if !ended {
            println!(
                "  Warning: MuScriptor chunk reached the 2000-token limit; keeping decoded notes."
            );
        }
        println!(
            "  MuScriptor chunk {}/{}: {:.2}s",
            chunk + 1,
            chunks,
            started.elapsed().as_secs_f64()
        );
    }
    Ok(decoder.finish())
}

fn add_positions(features: &mut [f32], dim: usize, offset: usize) {
    let half = dim / 2;
    for (row, features) in features.chunks_exact_mut(dim).enumerate() {
        let position = (offset + row) as f32;
        for index in 0..half {
            let phase = position / 10_000_f32.powf(index as f32 / (half - 1) as f32);
            features[index] += phase.cos();
            features[index + half] += phase.sin();
        }
    }
}

fn causal_mask(length: usize) -> Vec<f32> {
    let mut mask = vec![0.0; length * length];
    for row in 0..length {
        mask[row * length + row + 1..(row + 1) * length].fill(f32::NEG_INFINITY);
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_positions_match_prefill() {
        let mut full = vec![0.0; 8 * 768];
        add_positions(&mut full, 768, 0);
        let mut incremental = vec![0.0; 768];
        add_positions(&mut incremental, 768, 7);
        assert_eq!(&full[7 * 768..], incremental);
        assert_eq!(causal_mask(2), [0.0, f32::NEG_INFINITY, 0.0, 0.0]);
    }

    #[test]
    #[ignore = "Requires cached MuScriptor small and native Intel OpenVINO runtime"]
    fn native_small_transcribes_without_python() {
        let home = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config = Config {
            device: crate::inference::Device::Openvino,
            gpu_id: 0,
            precision: crate::inference::Precision::Fp32,
            threads: 4,
            cache: home.join(".cache/openvino"),
            profiles: None,
            openvino_library: home.join("runtime/openvino-2026.4.1/openvino_c.dll"),
        };
        let audio: Vec<_> = (0..128_000)
            .map(|n| {
                let n = n % 80_000;
                if n >= 48_000 {
                    return 0.0;
                }
                let time = (n % 9600) as f32 / 16_000.0;
                let envelope = (1.0 - (-time * 1200.0).exp()) * (-time * 6.0).exp();
                let frequency = if n / 9600 % 2 == 0 { 110.0 } else { 146.8324 };
                (1..=8)
                    .map(|harmonic| {
                        (time * std::f32::consts::TAU * frequency * harmonic as f32).sin()
                            / (harmonic as f32).powf(1.5)
                    })
                    .sum::<f32>()
                    * envelope
                    * 0.3
            })
            .collect();
        let notes = transcribe(Variant::Small, &audio, &home, None, &config).unwrap();
        assert!(
            !notes.is_empty(),
            "Expected notes in the synthetic A2 fixture"
        );
        assert!(
            notes
                .iter()
                .all(|n| n.start_s >= 0.0 && n.end_s <= 8.0 && n.end_s > n.start_s)
        );
        assert!(
            notes.iter().any(|n| n.start_s >= 5.0),
            "Expected notes from the second audio chunk"
        );
    }
}

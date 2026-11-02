use crate::{
    inference::{Config, Device},
    intel_gpu,
};
use anyhow::{Result, ensure};
use ort::{environment::Environment, value::TensorRef};
use std::{path::Path, time::Instant};

// StemSplit six-stem export: drums, bass, other, vocals, guitar, piano.
const SEGMENT: usize = 343_980;
const OVERLAP: usize = SEGMENT / 4;
const STRIDE: usize = SEGMENT - OVERLAP;
const GUITAR: usize = 4;

pub fn guitar(
    environment: &Environment,
    model: &Path,
    stereo: &[f32],
    config: &Config,
) -> Result<Vec<f32>> {
    ensure!(
        stereo.len().is_multiple_of(2) && !stereo.is_empty(),
        "Expected nonempty stereo PCM"
    );
    let initialization = Instant::now();
    let mut intel = if config.device == Device::Openvino {
        Some(intel_gpu::Demucs::new(model, config, SEGMENT)?)
    } else {
        None
    };
    let mut session = if intel.is_none() {
        Some(
            config
                .builder(environment, "demucs")?
                .commit_from_file(model)?,
        )
    } else {
        None
    };
    println!(
        "  Demucs session: {:?}, load {:.2}s",
        config.device,
        initialization.elapsed().as_secs_f64()
    );
    let frames = stereo.len() / 2;
    let starts = chunk_starts(frames);
    let mut result = vec![0.0; stereo.len()];
    let mut weight = vec![0.0; frames];
    let mut chunk = vec![0.0_f32; 2 * SEGMENT];
    for (index, &start) in starts.iter().enumerate() {
        let started = Instant::now();
        let length = (frames - start).min(SEGMENT);
        chunk.fill(0.0);
        for i in 0..length {
            chunk[i] = stereo[(start + i) * 2];
            chunk[SEGMENT + i] = stereo[(start + i) * 2 + 1];
        }
        let data = if let Some(intel) = &mut intel {
            intel.infer(&chunk, SEGMENT)?
        } else {
            let input = TensorRef::from_array_view(([1, 2, SEGMENT], chunk.as_slice()))?;
            let outputs = session
                .as_mut()
                .expect("ORT session exists")
                .run(ort::inputs!["mix" => input])?;
            let output = outputs
                .get("stems")
                .ok_or_else(|| anyhow::anyhow!("Model has no 'stems' output"))?;
            let (shape, data) = output.try_extract_tensor::<f32>()?;
            ensure!(
                shape.as_ref() == [1, 6, 2, SEGMENT as i64],
                "Unexpected six-stem ONNX output shape {shape:?}"
            );
            data[GUITAR * 2 * SEGMENT..(GUITAR + 1) * 2 * SEGMENT].to_vec()
        };
        ensure!(
            data.iter().all(|v| v.is_finite()),
            "Demucs produced invalid samples"
        );
        for i in 0..length {
            let w = window(i);
            weight[start + i] += w;
            for channel in 0..2 {
                result[(start + i) * 2 + channel] += data[channel * SEGMENT + i] * w;
            }
        }
        println!(
            "  Demucs chunk {}/{}: {:.2}s",
            index + 1,
            starts.len(),
            started.elapsed().as_secs_f64()
        );
    }
    for (frame, &w) in weight.iter().enumerate() {
        ensure!(w > 0.0, "Uncovered sample in Demucs overlap-add");
        result[frame * 2] /= w;
        result[frame * 2 + 1] /= w;
    }
    if let Some(session) = &mut session {
        config.finish_profile(session)?;
    }
    Ok(result)
}

fn chunk_starts(frames: usize) -> Vec<usize> {
    if frames == 0 {
        return Vec::new();
    }
    let mut starts = vec![0];
    while starts.last().copied().unwrap() + SEGMENT < frames {
        starts.push(starts.last().copied().unwrap() + STRIDE);
    }
    starts
}

fn window(index: usize) -> f32 {
    // Positive endpoints avoid losing the first/last sample through division by zero.
    ((index + 1).min(SEGMENT - index).min(OVERLAP + 1) as f32) / (OVERLAP + 1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_add_preserves_constant_signal_at_boundaries() {
        let frames = SEGMENT * 2 + 13;
        let mut accumulated = vec![0.0_f32; frames];
        let mut weights = vec![0.0_f32; frames];
        for start in chunk_starts(frames) {
            for i in 0..(frames - start).min(SEGMENT) {
                accumulated[start + i] += 0.42 * window(i);
                weights[start + i] += window(i);
            }
        }
        for (value, weight) in accumulated.into_iter().zip(weights) {
            assert!(weight > 0.0);
            assert!((value / weight - 0.42).abs() < 1e-6);
        }
        assert_eq!(chunk_starts(SEGMENT), vec![0]);
        assert_eq!(chunk_starts(SEGMENT + 1), vec![0, STRIDE]);
    }
}

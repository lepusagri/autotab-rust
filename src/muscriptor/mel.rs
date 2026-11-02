//! The published checkpoint contains the exact HTK filterbank and Hann window.
use super::weights::Weights;
use anyhow::{Result, ensure};
use rustfft::{FftPlanner, num_complex::Complex32};

pub struct Mel {
    bank: Vec<f32>,
    window: Vec<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl Mel {
    pub fn new(weights: &Weights) -> Result<Self> {
        let bank = weights
            .read("condition_provider.conditioners.self_wav.mel_spec_transform.mel_scale.fb")?;
        let window = weights.read(
            "condition_provider.conditioners.self_wav.mel_spec_transform.spectrogram.window",
        )?;
        ensure!(
            bank.len() == 1025 * 512 && window.len() == 2048,
            "Unexpected MuScriptor spectrogram weights"
        );
        Ok(Self {
            bank,
            window,
            fft: FftPlanner::new().plan_fft_forward(2048),
        })
    }

    pub fn compute(&self, audio: &[f32]) -> Vec<f32> {
        assert_eq!(audio.len(), 80_000);
        let mut result = vec![0.0_f32; 501 * 512];
        let mut frame = vec![Complex32::default(); 2048];
        for time in 0..501 {
            for (index, sample) in frame.iter_mut().enumerate() {
                let mut position = time as isize * 160 + index as isize - 1024;
                if position < 0 {
                    position = -position;
                }
                if position >= audio.len() as isize {
                    position = 2 * audio.len() as isize - position - 2;
                }
                *sample = Complex32::new(audio[position as usize] * self.window[index], 0.0);
            }
            self.fft.process(&mut frame);
            let output = &mut result[time * 512..(time + 1) * 512];
            for (frequency, value) in frame[..1025].iter().enumerate() {
                let magnitude = value.norm();
                for (mel, weight) in output
                    .iter_mut()
                    .zip(&self.bank[frequency * 512..(frequency + 1) * 512])
                {
                    *mel += magnitude * weight;
                }
            }
            for value in output {
                *value = (*value + 1e-6).ln();
            }
        }
        result
    }
}

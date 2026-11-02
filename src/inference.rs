use anyhow::{Context, Result, ensure};
use basic_pitch::{Posteriorgram, Posteriorgrams, constants::*, model::Backend};
use clap::ValueEnum;
use ort::{
    environment::Environment,
    session::{
        Session,
        builder::{AutoDevicePolicy, SessionBuilder},
    },
    value::TensorRef,
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Device {
    Auto,
    Cpu,
    Openvino,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Precision {
    Fp32,
    Fp16,
}

pub struct Config {
    pub device: Device,
    pub gpu_id: u16,
    pub precision: Precision,
    pub threads: usize,
    pub cache: PathBuf,
    pub profiles: Option<PathBuf>,
    pub openvino_library: PathBuf,
}

impl Device {
    pub fn resolve(self, home: &Path) -> Self {
        if self != Self::Auto {
            return self;
        }
        if home
            .join("runtime/openvino-2026.4.1/openvino_c.dll")
            .is_file()
        {
            Self::Openvino
        } else {
            Self::Cpu
        }
    }

    pub fn runtime(self, home: &Path) -> PathBuf {
        home.join("runtime/onnxruntime.dll")
    }
}

impl Config {
    pub fn builder(&self, environment: &Environment, model: &str) -> Result<SessionBuilder> {
        let mut builder = Session::builder(environment)?
            .with_intra_threads(self.threads)
            .map_err(builder_error)?;
        builder = builder
            .with_auto_device(AutoDevicePolicy::PreferCPU)
            .map_err(builder_error)?;
        if let Some(directory) = &self.profiles {
            fs::create_dir_all(directory)?;
            builder = builder
                .with_profiling(directory.join(model))
                .map_err(builder_error)?;
        }
        Ok(builder)
    }

    pub fn finish_profile(&self, session: &mut Session) -> Result<()> {
        if self.profiles.is_some() {
            println!("  ONNX profile: {}", session.end_profiling()?);
        }
        Ok(())
    }
}

fn builder_error(error: ort::Error<SessionBuilder>) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

pub struct PitchBackend {
    session: Mutex<Session>,
}

impl PitchBackend {
    pub fn new(environment: &Environment, config: &Config) -> Result<Self> {
        let session = config
            .builder(environment, "basic-pitch")?
            .commit_from_memory(basic_pitch::model::MODEL_ONNX)
            .context("Cannot load Basic Pitch model")?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    pub fn finish_profile(&self, config: &Config) -> Result<()> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("Basic Pitch session lock poisoned"))?;
        config.finish_profile(&mut session)
    }
}

impl Backend for PitchBackend {
    fn predict_window(&self, window: &[f32]) -> basic_pitch::Result<Posteriorgrams> {
        let result = (|| -> Result<Posteriorgrams> {
            ensure!(
                window.len() == AUDIO_N_SAMPLES,
                "Invalid Basic Pitch window length"
            );
            let input = TensorRef::from_array_view(([1, AUDIO_N_SAMPLES, 1], window))?;
            let mut session = self
                .session
                .lock()
                .map_err(|_| anyhow::anyhow!("Basic Pitch session lock poisoned"))?;
            let outputs = session.run(ort::inputs![MODEL_INPUT_NAME => input])?;
            let extract = |name: &str, bins: usize| -> Result<Posteriorgram> {
                let value = outputs
                    .get(name)
                    .with_context(|| format!("Missing Basic Pitch output {name}"))?;
                let (shape, data) = value.try_extract_tensor::<f32>()?;
                ensure!(
                    shape.as_ref() == [1, ANNOT_N_FRAMES as i64, bins as i64],
                    "Unexpected Basic Pitch output shape {shape:?}"
                );
                Ok(Posteriorgram::from_vec(data.to_vec(), bins)?)
            };
            Ok(Posteriorgrams::new(
                extract(MODEL_OUTPUT_NOTE, 88)?,
                extract(MODEL_OUTPUT_ONSET, 88)?,
                extract(MODEL_OUTPUT_CONTOUR, 264)?,
            )?)
        })();
        result.map_err(|error| basic_pitch::Error::backend("autotab-onnx", error.to_string()))
    }
}

use crate::{
    inference::{Config, Device, Precision},
    intel_gpu::{Handle, core_property, model_property, set_property, status},
};
use anyhow::{Context, Result, ensure};
use openvino_sys as ov;
use std::{ffi::CString, path::Path, ptr, slice};

pub struct Runtime {
    pub decoder: Handle<ov::ov_compiled_model_t>,
    pub encoder: Handle<ov::ov_compiled_model_t>,
    _core: Handle<ov::ov_core_t>,
    #[cfg(windows)]
    _dependencies: libloading::os::windows::Library,
}

impl Runtime {
    pub fn new(files: &super::graph::Files, config: &Config) -> Result<Self> {
        ensure!(
            config.openvino_library.is_file(),
            "MuScriptor requires the native OpenVINO runtime. Run setup-rust.ps1 first."
        );
        #[cfg(windows)]
        // SAFETY: pinned native library; the owner outlives every OpenVINO handle.
        let dependencies = unsafe {
            libloading::os::windows::Library::load_with_flags(
                config
                    .openvino_library
                    .parent()
                    .context("Missing OpenVINO folder")?
                    .join("openvino.dll"),
                libloading::os::windows::LOAD_WITH_ALTERED_SEARCH_PATH,
            )?
        };
        ov::library::load_from(config.openvino_library.clone()).map_err(|e| anyhow::anyhow!(e))?;
        let mut core = Handle::new(ov::ov_core_free);
        // SAFETY: empty output slot receives an exclusively owned core handle.
        status(unsafe { ov::ov_core_create(&mut core.ptr) })?;
        let device_name = if config.device == Device::Cpu {
            "CPU".to_owned()
        } else {
            format!("GPU.{}", config.gpu_id)
        };
        let device = CString::new(device_name.clone())?;
        let precision = if config.precision == Precision::Fp16 {
            "f16"
        } else {
            "f32"
        };
        for (key, value) in [
            (c"PERFORMANCE_HINT", "LATENCY"),
            (c"NUM_STREAMS", "1"),
            (c"INFERENCE_PRECISION_HINT", precision),
            (
                c"EXECUTION_MODE_HINT",
                if precision == "f32" {
                    "ACCURACY"
                } else {
                    "PERFORMANCE"
                },
            ),
        ] {
            set_property(&core, &device, key, value)?;
        }
        if config.device == Device::Cpu {
            set_property(
                &core,
                &device,
                c"INFERENCE_NUM_THREADS",
                &config.threads.to_string(),
            )?;
        }
        let cache = files
            .bin
            .parent()
            .context("Missing cache directory")?
            .join("compiled");
        std::fs::create_dir_all(&cache)?;
        set_property(
            &core,
            &device,
            c"CACHE_DIR",
            cache.to_str().context("Invalid cache path")?,
        )?;
        println!(
            "  MuScriptor device: {} ({device_name}), precision {precision}",
            core_property(&core, &device, c"FULL_DEVICE_NAME")?
        );
        let encoder = compile(&core, &device, &files.encoder, &files.bin)?;
        println!("  Compiling/loading MuScriptor decoder with GPU-resident KV state...");
        let decoder = compile(&core, &device, &files.decoder, &files.bin)?;
        let actual = model_property(&decoder, c"EXECUTION_DEVICES")?;
        ensure!(
            actual.contains(&device_name),
            "Requested {device_name}, but MuScriptor executes on {actual}"
        );
        Ok(Self {
            decoder,
            encoder,
            _core: core,
            #[cfg(windows)]
            _dependencies: dependencies,
        })
    }
}

fn compile(
    core: &Handle<ov::ov_core_t>,
    device: &CString,
    xml: &Path,
    bin: &Path,
) -> Result<Handle<ov::ov_compiled_model_t>> {
    let mut model = Handle::new(ov::ov_model_free);
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let xml: Vec<u16> = xml.as_os_str().encode_wide().chain(Some(0)).collect();
        let bin: Vec<u16> = bin.as_os_str().encode_wide().chain(Some(0)).collect();
        // SAFETY: both UTF-16 paths and the output slot remain live during the call.
        status(unsafe {
            ov::ov_core_read_model_unicode(
                core.ptr,
                xml.as_ptr().cast(),
                bin.as_ptr().cast(),
                &mut model.ptr,
            )
        })?;
    }
    #[cfg(not(windows))]
    anyhow::bail!("The bundled MuScriptor runtime requires Windows x64");
    let mut compiled = Handle::new(ov::ov_compiled_model_free);
    // SAFETY: all handles are live, no variadic properties are passed.
    status(unsafe {
        ov::ov_core_compile_model(core.ptr, model.ptr, device.as_ptr(), 0, &mut compiled.ptr)
    })?;
    Ok(compiled)
}

pub struct Request {
    handle: Handle<ov::ov_infer_request_t>,
}

impl Request {
    pub fn new(model: &Handle<ov::ov_compiled_model_t>) -> Result<Self> {
        let mut handle = Handle::new(ov::ov_infer_request_free);
        // SAFETY: the compiled model outlives this request at each call site.
        status(unsafe { ov::ov_compiled_model_create_infer_request(model.ptr, &mut handle.ptr) })?;
        Ok(Self { handle })
    }

    pub fn run(&mut self, inputs: &[(&[i64], &[f32])]) -> Result<Vec<f32>> {
        let mut tensors = Vec::new();
        for (index, (shape, values)) in inputs.iter().enumerate() {
            ensure!(
                shape.iter().all(|&n| n > 0)
                    && shape.iter().map(|&n| n as usize).product::<usize>() == values.len(),
                "Invalid MuScriptor input shape"
            );
            let mut tensor = Handle::new(ov::ov_tensor_free);
            let shape = ov::ov_shape_t {
                rank: shape.len() as i64,
                dims: shape.as_ptr().cast_mut(),
            };
            // SAFETY: the API copies the shape and allocates an owned F32 tensor.
            status(unsafe {
                ov::ov_tensor_create(ov::ov_element_type_e::F32, shape, &mut tensor.ptr)
            })?;
            let mut destination = ptr::null_mut();
            // SAFETY: this live tensor has exactly values.len() F32 elements.
            status(unsafe { ov::ov_tensor_data(tensor.ptr, &mut destination) })?;
            // SAFETY: source and allocated tensor storage are valid and non-overlapping.
            unsafe {
                ptr::copy_nonoverlapping(values.as_ptr(), destination.cast::<f32>(), values.len())
            };
            // SAFETY: the request and input tensor stay alive through synchronous inference.
            status(unsafe {
                ov::ov_infer_request_set_input_tensor_by_index(self.handle.ptr, index, tensor.ptr)
            })?;
            tensors.push(tensor);
        }
        // SAFETY: all input tensors remain alive until inference finishes.
        status(unsafe { ov::ov_infer_request_infer(self.handle.ptr) })?;
        let mut output = Handle::new(ov::ov_tensor_free);
        // SAFETY: output slot receives an owned reference to result zero.
        status(unsafe {
            ov::ov_infer_request_get_output_tensor_by_index(self.handle.ptr, 0, &mut output.ptr)
        })?;
        let mut count = 0;
        let mut data = ptr::null_mut();
        // SAFETY: the runtime owns a live F32 result tensor throughout this read.
        status(unsafe { ov::ov_tensor_get_size(output.ptr, &mut count) })?;
        // SAFETY: pointer remains valid through the subsequent copy.
        status(unsafe { ov::ov_tensor_data(output.ptr, &mut data) })?;
        ensure!(!data.is_null() && count > 0, "Empty MuScriptor result");
        // SAFETY: the tensor API reports count allocated F32 elements.
        Ok(unsafe { slice::from_raw_parts(data.cast::<f32>(), count) }.to_vec())
    }
}

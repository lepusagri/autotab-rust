//! Minimal owned bindings for cached GPU inference. All pointers are confined to
//! this module. Each handle owns exactly one C API object; tensors and requests
//! are destroyed before their compiled model and core. No handle crosses threads.
use crate::inference::{Config, Precision};
use anyhow::{Context, Result, ensure};
use openvino_sys as ov;
use sha2::{Digest, Sha256};
use std::{
    ffi::{CStr, CString},
    fs,
    path::Path,
    ptr, slice,
};

struct Handle<T> {
    ptr: *mut T,
    free: unsafe fn(*mut T),
}

impl<T> Handle<T> {
    fn new(free: unsafe fn(*mut T)) -> Self {
        Self {
            ptr: ptr::null_mut(),
            free,
        }
    }
}

impl<T> Drop for Handle<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: this handle exclusively owns the corresponding API object.
            unsafe { (self.free)(self.ptr) };
        }
    }
}

fn status(code: ov::ov_status_e) -> Result<()> {
    if code == ov::ov_status_e::OK {
        return Ok(());
    }
    // SAFETY: the API returns either null or a library-owned NUL-terminated message.
    let message = unsafe {
        let message = ov::ov_get_last_err_msg();
        if message.is_null() {
            format!("{code:?}")
        } else {
            CStr::from_ptr(message).to_string_lossy().into_owned()
        }
    };
    anyhow::bail!("OpenVINO {code:?}: {message}")
}

fn owned_string(value: *mut std::ffi::c_char) -> Result<String> {
    ensure!(!value.is_null(), "OpenVINO returned a null property");
    // SAFETY: successful property queries allocate a NUL-terminated string that
    // the caller must release with ov_free after copying it.
    unsafe {
        let string = CStr::from_ptr(value).to_string_lossy().into_owned();
        ov::ov_free(value);
        Ok(string)
    }
}

fn core_property(core: &Handle<ov::ov_core_t>, device: &CStr, key: &CStr) -> Result<String> {
    let mut value = ptr::null_mut();
    // SAFETY: core is live; all strings and the output slot live through the call.
    status(unsafe {
        ov::ov_core_get_property(core.ptr, device.as_ptr(), key.as_ptr(), &mut value)
    })?;
    owned_string(value)
}

fn model_property(model: &Handle<ov::ov_compiled_model_t>, key: &CStr) -> Result<String> {
    let mut value = ptr::null_mut();
    // SAFETY: model is live and the output pointer is owned on success.
    status(unsafe { ov::ov_compiled_model_get_property(model.ptr, key.as_ptr(), &mut value) })?;
    owned_string(value)
}

fn set_property(
    core: &Handle<ov::ov_core_t>,
    device: &CStr,
    key: &CStr,
    value: &str,
) -> Result<()> {
    let value = CString::new(value)?;
    // SAFETY: all pointers designate live API objects or NUL-terminated strings.
    status(unsafe {
        ov::ov_core_set_property(core.ptr, device.as_ptr(), key.as_ptr(), value.as_ptr())
    })
}

#[cfg(windows)]
fn intel_driver_versions() -> Result<String> {
    use winreg::HKLM;
    let display = HKLM.open_subkey(
        r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}",
    )?;
    let mut versions = Vec::new();
    for name in display.enum_keys().filter_map(Result::ok) {
        let Ok(key) = display.open_subkey(name) else {
            continue;
        };
        let hardware: String = key.get_value("MatchingDeviceId").unwrap_or_default();
        if hardware.to_ascii_lowercase().contains("ven_8086") {
            let version: String = key.get_value("DriverVersion")?;
            versions.push(version);
        }
    }
    versions.sort();
    versions.dedup();
    ensure!(
        !versions.is_empty(),
        "Intel display driver version not found in Windows"
    );
    Ok(versions.join(","))
}

/// Request/tensors are dropped before their dependencies. Retaining imported bytes
/// also makes the input buffer lifetime independent of runtime import internals.
pub struct Demucs {
    request: Handle<ov::ov_infer_request_t>,
    input: Handle<ov::ov_tensor_t>,
    _compiled: Handle<ov::ov_compiled_model_t>,
    _core: Handle<ov::ov_core_t>,
    _cache_bytes: Vec<u8>,
    #[cfg(windows)]
    _dependencies: libloading::os::windows::Library,
}

impl Demucs {
    pub fn new(model: &Path, config: &Config, segment: usize) -> Result<Self> {
        #[cfg(windows)]
        // SAFETY: setup installs this pinned DLL and its sibling dependencies.
        // The handle outlives every API object; PATH is never changed.
        let dependencies = unsafe {
            libloading::os::windows::Library::load_with_flags(
                config
                    .openvino_library
                    .parent()
                    .context("Missing runtime directory")?
                    .join("openvino.dll"),
                libloading::os::windows::LOAD_WITH_ALTERED_SEARCH_PATH,
            )?
        };
        openvino_sys::library::load_from(config.openvino_library.clone())
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut core = Handle::new(ov::ov_core_free);
        // SAFETY: the output slot starts null and is owned by core on success.
        status(unsafe { ov::ov_core_create(&mut core.ptr) })?;
        let device = CString::new(format!("GPU.{}", config.gpu_id))?;
        let name = core_property(&core, &device, c"FULL_DEVICE_NAME")?;
        // GPU_DRIVER_VERSION is no longer public in OpenVINO 2026. Query the
        // installed Intel display drivers using Windows' native registry API.
        #[cfg(windows)]
        let driver = intel_driver_versions().context("Cannot query Intel GPU driver version")?;
        #[cfg(not(windows))]
        let driver = core_property(&core, &device, c"GPU_DRIVER_VERSION")?;
        println!(
            "  Intel GPU: {name} ({}), driver {driver}",
            device.to_string_lossy()
        );
        for (key, value) in [
            (c"CACHE_MODE", "OPTIMIZE_SPEED"),
            (c"PERFORMANCE_HINT", "LATENCY"),
            (c"NUM_STREAMS", "1"),
            (
                c"INFERENCE_PRECISION_HINT",
                if config.precision == Precision::Fp16 {
                    "f16"
                } else {
                    "f32"
                },
            ),
            (
                c"EXECUTION_MODE_HINT",
                if config.precision == Precision::Fp16 {
                    "PERFORMANCE"
                } else {
                    "ACCURACY"
                },
            ),
        ] {
            set_property(&core, &device, key, value)?;
        }

        fs::create_dir_all(&config.cache)?;
        let runtime_version = openvino::version().build_number;
        let mut hash = Sha256::new();
        hash.update(fs::read(model).with_context(|| format!("Cannot read {}", model.display()))?);
        hash.update(format!(
            "autotab-gpu-v1/{runtime_version}/{name}/{driver}/{}/{:?}",
            config.gpu_id, config.precision
        ));
        let cache_path = config
            .cache
            .join(format!("demucs-{:x}.blob", hash.finalize()));
        let mut compiled = Handle::new(ov::ov_compiled_model_free);
        let mut cache_bytes = Vec::new();
        let mut cached = false;
        if let Ok(bytes) = fs::read(&cache_path) {
            // SAFETY: bytes remain alive for the complete compiled-model lifetime.
            let code = unsafe {
                ov::ov_core_import_model(
                    core.ptr,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    device.as_ptr(),
                    &mut compiled.ptr,
                )
            };
            if code == ov::ov_status_e::OK {
                cache_bytes = bytes;
                cached = true;
            } else {
                eprintln!(
                    "  GPU cache rejected: {}. Rebuilding.",
                    status(code).unwrap_err()
                );
                compiled = Handle::new(ov::ov_compiled_model_free);
                fs::remove_file(&cache_path)?;
            }
        }
        if !cached {
            println!("  Compiling Demucs for the GPU (first run; saved for later runs)...");
            #[cfg(windows)]
            {
                use std::os::windows::ffi::OsStrExt;
                let model_path: Vec<u16> = model.as_os_str().encode_wide().chain(Some(0)).collect();
                // SAFETY: the core and path are live; zero property args means no
                // variadic arguments. Device properties were set above.
                status(unsafe {
                    ov::ov_core_compile_model_from_file_unicode(
                        core.ptr,
                        model_path.as_ptr().cast(),
                        device.as_ptr(),
                        0,
                        &mut compiled.ptr,
                    )
                })?;
            }
            #[cfg(not(windows))]
            anyhow::bail!("This bundled OpenVINO runtime requires Windows x64");
            let temporary = cache_path.with_extension(format!("{}.part", std::process::id()));
            let path = CString::new(temporary.to_str().context("Non-UTF8 cache path")?)?;
            // SAFETY: compiled is live; the API synchronously writes the requested file.
            status(unsafe { ov::ov_compiled_model_export_model(compiled.ptr, path.as_ptr()) })?;
            if let Err(error) = fs::rename(&temporary, &cache_path) {
                // Another process may have populated the same content-addressed cache.
                fs::remove_file(&temporary).ok();
                if !cache_path.is_file() {
                    return Err(error.into());
                }
            }
        }
        let devices = model_property(&compiled, c"EXECUTION_DEVICES")?;
        let precision = model_property(&compiled, c"INFERENCE_PRECISION_HINT")?;
        ensure!(
            devices.contains("GPU"),
            "Requested Intel GPU but runtime selected {devices}"
        );
        let expected = if config.precision == Precision::Fp16 {
            "f16"
        } else {
            "f32"
        };
        ensure!(
            precision == expected,
            "Requested {expected} but runtime uses {precision}"
        );
        println!("  OpenVINO execution: {devices}, precision {precision}, cached {cached}");
        if let Some(directory) = &config.profiles {
            fs::create_dir_all(directory)?;
            let metadata = serde_json::json!({"device_name": name, "execution_devices": devices,
                "inference_precision": precision, "loaded_from_cache": cached,
                "runtime": runtime_version, "driver": driver});
            fs::write(
                directory.join("demucs-openvino.json"),
                serde_json::to_vec_pretty(&metadata)?,
            )?;
        }
        let mut request = Handle::new(ov::ov_infer_request_free);
        let mut input = Handle::new(ov::ov_tensor_free);
        // SAFETY: compiled stays alive; the returned request/tensor references are owned.
        unsafe {
            status(ov::ov_compiled_model_create_infer_request(
                compiled.ptr,
                &mut request.ptr,
            ))?;
            status(ov::ov_infer_request_get_input_tensor(
                request.ptr,
                &mut input.ptr,
            ))?;
        }
        check_tensor(&input, &[1, 2, segment as i64])?;
        Ok(Self {
            request,
            input,
            _compiled: compiled,
            _core: core,
            _cache_bytes: cache_bytes,
            #[cfg(windows)]
            _dependencies: dependencies,
        })
    }

    pub fn infer(&mut self, planar: &[f32], segment: usize) -> Result<Vec<f32>> {
        ensure!(planar.len() == segment * 2, "Invalid Demucs input length");
        let mut data = ptr::null_mut();
        // SAFETY: input was verified to be a float32 tensor of exactly planar.len()
        // elements. This request is synchronous and never used concurrently.
        unsafe {
            status(ov::ov_tensor_data(self.input.ptr, &mut data))?;
            ensure!(!data.is_null(), "Missing input buffer");
            ptr::copy_nonoverlapping(planar.as_ptr(), data.cast::<f32>(), planar.len());
            status(ov::ov_infer_request_infer(self.request.ptr))?;
        }
        let mut output = Handle::new(ov::ov_tensor_free);
        // SAFETY: request is live; the returned tensor handle is owned.
        status(unsafe {
            ov::ov_infer_request_get_output_tensor(self.request.ptr, &mut output.ptr)
        })?;
        check_tensor(&output, &[1, 6, 2, segment as i64])?;
        // SAFETY: check_tensor established dtype and size. We copy only the two
        // guitar channels while output and request are still alive.
        unsafe {
            status(ov::ov_tensor_data(output.ptr, &mut data))?;
            ensure!(!data.is_null(), "Missing output buffer");
            Ok(slice::from_raw_parts(data.cast::<f32>().add(8 * segment), 2 * segment).to_vec())
        }
    }
}

fn check_tensor(tensor: &Handle<ov::ov_tensor_t>, expected: &[i64]) -> Result<()> {
    let mut element_type = ov::ov_element_type_e::F32;
    let mut shape = ov::ov_shape_t {
        rank: 0,
        dims: ptr::null_mut(),
    };
    // SAFETY: tensor is live; metadata is copied out and shape is then released.
    unsafe {
        status(ov::ov_tensor_get_element_type(
            tensor.ptr,
            &mut element_type,
        ))?;
        ensure!(
            element_type == ov::ov_element_type_e::F32,
            "Expected float32 tensor"
        );
        status(ov::ov_tensor_get_shape(tensor.ptr, &mut shape))?;
        let matches = shape.rank == expected.len() as i64
            && !shape.dims.is_null()
            && slice::from_raw_parts(shape.dims, expected.len()) == expected;
        status(ov::ov_shape_free(&mut shape))?;
        ensure!(matches, "Unexpected Demucs tensor shape");
    }
    Ok(())
}

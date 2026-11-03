use anyhow::{Context, Result, ensure};
use std::{
    ffi::OsString,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
pub struct Backend {
    pub home: PathBuf,
    pub executable: PathBuf,
}

impl Backend {
    pub fn discover() -> Result<Self> {
        let current = std::env::current_exe()?;
        let home = if let Some(home) = std::env::var_os("AUTOTAB_HOME") {
            PathBuf::from(home)
        } else {
            current
                .ancestors()
                .skip(1)
                .find(|path| path.join("runtime/onnxruntime.dll").is_file())
                .context("Einrichtung fehlt. Bitte setup-rust.ps1 ausführen.")?
                .to_path_buf()
        };
        let executable = current
            .parent()
            .context("Programmordner fehlt")?
            .join("autotab.exe");
        ensure!(
            executable.is_file(),
            "autotab.exe fehlt neben der GUI. Bitte setup-rust.ps1 ausführen."
        );
        Ok(Self { home, executable })
    }
}

#[derive(Clone, Debug)]
pub enum Source {
    File(PathBuf),
    Url(String),
}

/// Session-only credential. Debug output deliberately never includes its value.
#[derive(Clone, Default)]
pub struct HfToken(String);

impl std::fmt::Debug for HfToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("HfToken([redacted])")
    }
}

impl HfToken {
    pub fn new(value: &str) -> Self {
        Self(value.trim().to_owned())
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.0.bytes().all(|byte| byte.is_ascii_graphic()),
            "Der Hugging-Face-Token darf keine Leer- oder Steuerzeichen enthalten."
        );
        Ok(())
    }

    fn apply(&self, command: &mut Command) {
        if !self.0.is_empty() {
            command.env("HF_TOKEN", &self.0);
        }
    }

    fn redact(&self, text: &str) -> String {
        if self.0.is_empty() {
            text.to_owned()
        } else {
            text.replace(&self.0, "[Token ausgeblendet]")
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub source: Source,
    pub output: PathBuf,
    pub tuning: &'static str,
    pub device: &'static str,
    pub gpu_id: u16,
    pub threads: u16,
    pub clean_guitar: bool,
    pub transcriber: &'static str,
    pub muscriptor_model: Option<PathBuf>,
    pub hf_token: HfToken,
    pub instruments: crate::instruments::Instruments,
}

impl Options {
    pub fn validate(&self) -> Result<()> {
        self.hf_token.validate()?;
        if let Some(path) = &self.muscriptor_model {
            ensure!(
                path.is_file(),
                "Die lokale MuScriptor-Modelldatei wurde nicht gefunden."
            );
        }
        match &self.source {
            Source::File(path) => ensure!(
                path.is_file(),
                "Bitte eine vorhandene Audiodatei auswählen."
            ),
            Source::Url(url) => {
                let url = reqwest::Url::parse(url)
                    .context("Bitte einen gültigen Audio-Link eingeben.")?;
                ensure!(
                    matches!(url.scheme(), "http" | "https") && url.host_str().is_some(),
                    "Der Link muss mit https:// oder http:// beginnen."
                );
            }
        }
        ensure!(
            self.output
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("txt")),
            "Die Tab-Ausgabe muss eine .txt-Datei sein."
        );
        ensure!(
            !self.output.exists() && !self.output.with_extension("mid").exists(),
            "Ausgabedateien existieren bereits. Bitte einen anderen Zielordner wählen oder erneut starten."
        );
        ensure!(
            self.threads > 0,
            "Mindestens ein CPU-Thread ist erforderlich."
        );
        Ok(())
    }

    fn arguments(&self) -> Result<Vec<OsString>> {
        self.validate()?;
        let mut args = Vec::new();
        match &self.source {
            Source::File(path) => {
                args.push("--input".into());
                args.push(std::path::absolute(path)?.into_os_string());
            }
            Source::Url(url) => {
                args.push("--url".into());
                args.push(url.into());
            }
        }
        for (flag, value) in [
            ("--tuning", self.tuning.to_owned()),
            ("--device", self.device.to_owned()),
            ("--gpu-id", self.gpu_id.to_string()),
            ("--threads", self.threads.to_string()),
            ("--transcriber", self.transcriber.to_owned()),
        ] {
            args.push(flag.into());
            args.push(value.into());
        }
        args.push("--output".into());
        args.push(self.output.clone().into_os_string());
        if self.transcriber.starts_with("muscriptor-") {
            args.push("--instruments".into());
            args.push(self.instruments.to_string().into());
        }
        if let Some(path) = &self.muscriptor_model {
            args.push("--muscriptor-model".into());
            args.push(std::path::absolute(path)?.into_os_string());
        }
        if self.clean_guitar {
            args.push("--skip-separation".into());
        }
        Ok(args)
    }
}

pub fn output_path(folder: &Path, source: &Source) -> PathBuf {
    let stem = match source {
        Source::File(path) => path
            .file_stem()
            .filter(|stem| !stem.is_empty())
            .unwrap_or_else(|| std::ffi::OsStr::new("audio"))
            .to_string_lossy()
            .into_owned(),
        Source::Url(_) => "download".into(),
    };
    for suffix in 0.. {
        let name = if suffix == 0 {
            format!("{stem}_tab.txt")
        } else {
            format!("{stem}_tab_{suffix}.txt")
        };
        let candidate = folder.join(name);
        if !candidate.exists() && !candidate.with_extension("mid").exists() {
            return candidate;
        }
    }
    unreachable!()
}

#[derive(Debug)]
pub enum Event {
    Line(String),
    Finished(Outcome),
}

#[derive(Debug)]
pub enum Outcome {
    Success {
        tab: PathBuf,
        midi: PathBuf,
        elapsed: Duration,
    },
    Cancelled,
    Failed(String),
}

pub struct Job {
    pub events: Receiver<Event>,
    cancel: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Job {
    pub fn start(backend: Backend, options: Options) -> Result<Self> {
        let arguments = options.arguments()?;
        let (sender, events) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancellation = cancel.clone();
        let worker = thread::Builder::new()
            .name("autotab-worker".into())
            .spawn(move || {
                let result = run(backend, &options, arguments, &sender, &cancellation);
                let outcome = result.unwrap_or_else(|error| Outcome::Failed(format!("{error:#}")));
                sender.send(Event::Finished(outcome)).ok();
            })?;
        Ok(Self {
            events,
            cancel,
            worker: Some(worker),
        })
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        self.cancel();
        if let Some(worker) = self.worker.take() {
            worker.join().ok();
        }
    }
}

pub fn hide_console(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
}

fn read_lines(
    reader: impl Read + Send + 'static,
    sender: Sender<Event>,
    token: HfToken,
) -> JoinHandle<()> {
    thread::spawn(move || {
        for line in BufReader::new(reader).lines() {
            match line {
                Ok(line) => {
                    if sender.send(Event::Line(token.redact(&line))).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    sender
                        .send(Event::Line(format!(
                            "Protokoll konnte nicht gelesen werden: {error}"
                        )))
                        .ok();
                    break;
                }
            }
        }
    })
}

fn terminate(child: &mut Child) {
    // Cancel the owned process tree, including any active FFmpeg/download child.
    #[cfg(windows)]
    {
        let mut kill = Command::new("taskkill.exe");
        hide_console(&mut kill);
        let success = kill
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        if !success {
            child.kill().ok();
        }
    }
    #[cfg(not(windows))]
    child.kill().ok();
    child.wait().ok();
}

fn run(
    backend: Backend,
    options: &Options,
    arguments: Vec<OsString>,
    sender: &Sender<Event>,
    cancel: &AtomicBool,
) -> Result<Outcome> {
    let started = Instant::now();
    let mut command = Command::new(&backend.executable);
    command
        .args(arguments)
        .current_dir(&backend.home)
        .env("AUTOTAB_HOME", &backend.home)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    hide_console(&mut command);
    options.hf_token.apply(&mut command);
    let mut child = command
        .spawn()
        .context("Die Verarbeitung konnte nicht gestartet werden")?;
    let stdout = read_lines(
        child
            .stdout
            .take()
            .context("Kein Ausgabeprotokoll verfügbar")?,
        sender.clone(),
        options.hf_token.clone(),
    );
    let stderr = read_lines(
        child
            .stderr
            .take()
            .context("Kein Fehlerprotokoll verfügbar")?,
        sender.clone(),
        options.hf_token.clone(),
    );
    let outcome = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                break if status.success() {
                    Outcome::Success {
                        tab: options.output.clone(),
                        midi: options.output.with_extension("mid"),
                        elapsed: started.elapsed(),
                    }
                } else {
                    Outcome::Failed(
                        "Die Verarbeitung ist fehlgeschlagen. Details stehen im Protokoll.".into(),
                    )
                };
            }
            Ok(None) => {}
            Err(error) => {
                terminate(&mut child);
                break Outcome::Failed(error.to_string());
            }
        }
        if cancel.load(Ordering::Relaxed) {
            terminate(&mut child);
            break Outcome::Cancelled;
        }
        thread::sleep(Duration::from_millis(50));
    };
    stdout.join().ok();
    stderr.join().ok();
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_name_preserves_existing_tab_and_midi() {
        let directory = tempfile::tempdir().unwrap();
        let source = Source::File("a song.mp3".into());
        std::fs::write(directory.path().join("a song_tab.txt"), "existing").unwrap();
        std::fs::write(directory.path().join("a song_tab_1.mid"), "existing").unwrap();
        assert_eq!(
            output_path(directory.path(), &source),
            directory.path().join("a song_tab_2.txt")
        );
    }

    #[test]
    fn source_with_spaces_and_unicode_is_one_argument() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("Gitarre März.mp3");
        std::fs::write(&input, "fixture").unwrap();
        let model = directory
            .path()
            .join("Modell März mit Leerzeichen.safetensors");
        std::fs::write(&model, "fixture").unwrap();
        let options = Options {
            source: Source::File(input.clone()),
            output: directory.path().join("output.txt"),
            tuning: "drop_d",
            device: "openvino",
            gpu_id: 1,
            threads: 4,
            clean_guitar: true,
            transcriber: "muscriptor-small",
            muscriptor_model: Some(model.clone()),
            hf_token: HfToken::new("test-session-credential"),
            instruments: "acoustic_piano,acoustic_guitar,acoustic_bass"
                .parse()
                .unwrap(),
        };
        let arguments = options.arguments().unwrap();
        assert!(!arguments.iter().any(|argument| {
            argument
                .to_string_lossy()
                .contains("test-session-credential")
        }));
        assert!(!format!("{options:?}").contains("test-session-credential"));
        assert_eq!(arguments[1], input.as_os_str());
        assert!(arguments.contains(&OsString::from("--skip-separation")));
        assert!(
            arguments
                .windows(2)
                .any(|pair| pair == [OsString::from("--instruments"), OsString::from("0,4,7")])
        );
        assert!(arguments.windows(2).any(|pair| pair
            == [
                OsString::from("--transcriber"),
                OsString::from("muscriptor-small")
            ]));
        assert!(arguments.windows(2).any(|pair| pair
            == [
                OsString::from("--muscriptor-model"),
                model.clone().into_os_string()
            ]));
    }

    #[cfg(windows)]
    #[test]
    fn token_is_child_only_and_redacted_from_output() {
        let original = std::env::var_os("HF_TOKEN");
        let token = HfToken::new("  test-session-credential  ");
        let mut command = Command::new("cmd.exe");
        command.args(["/d", "/c", "echo %HF_TOKEN%"]);
        hide_console(&mut command);
        token.apply(&mut command);
        let output = command.output().unwrap();
        assert!(output.status.success());
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "test-session-credential"
        );
        assert_eq!(std::env::var_os("HF_TOKEN"), original);
        let (sender, receiver) = mpsc::channel();
        read_lines(std::io::Cursor::new(output.stdout), sender, token)
            .join()
            .unwrap();
        let Event::Line(line) = receiver.recv().unwrap() else {
            panic!("Expected output line")
        };
        assert_eq!(line, "[Token ausgeblendet]");
    }

    #[test]
    fn blank_token_preserves_existing_auth_and_invalid_input_is_rejected() {
        let mut command = Command::new("unused-test-program");
        command.env("HF_TOKEN", "existing-authentication");
        HfToken::new("   ").apply(&mut command);
        assert!(command.get_envs().any(|(key, value)| key == "HF_TOKEN"
            && value == Some(std::ffi::OsStr::new("existing-authentication"))));
        let token = HfToken::new("test-session\ncredential");
        let error = token.validate().unwrap_err().to_string();
        assert!(!error.contains("test-session"));
    }

    fn fixture(directory: &Path) -> PathBuf {
        let path = directory.join("Gitarre März mit Leerzeichen.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for frame in 0..44_100 * 3 {
            let time = f64::from(frame) / 44_100.0;
            let value = (1..=6)
                .map(|harmonic| {
                    (2.0 * std::f64::consts::PI * 110.0 * f64::from(harmonic) * time).sin()
                        / f64::from(harmonic)
                })
                .sum::<f64>()
                * (-2.0 * (time % 0.75)).exp()
                * 0.2;
            writer.write_sample(value as f32).unwrap();
        }
        writer.finalize().unwrap();
        path
    }

    fn native_backend() -> Backend {
        let executable = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("autotab.exe");
        assert!(
            executable.is_file(),
            "Build the CLI alongside the GUI first"
        );
        Backend {
            home: env!("CARGO_MANIFEST_DIR").into(),
            executable,
        }
    }

    #[test]
    #[ignore = "Needs native runtimes, an Intel GPU, and the built CLI"]
    fn worker_runs_arc_pipeline_and_reads_both_output_files() {
        let directory = tempfile::tempdir().unwrap();
        let options = Options {
            source: Source::File(fixture(directory.path())),
            output: directory.path().join("result.txt"),
            tuning: "standard",
            device: "openvino",
            gpu_id: 0,
            threads: 4,
            clean_guitar: false,
            transcriber: "basic-pitch",
            muscriptor_model: None,
            hf_token: HfToken::default(),
            instruments: crate::instruments::Instruments::default(),
        };
        let job = Job::start(native_backend(), options).unwrap();
        let mut saw_gpu = false;
        loop {
            match job.events.recv_timeout(Duration::from_secs(120)).unwrap() {
                Event::Line(line) => {
                    saw_gpu |= line.contains("OpenVINO execution: GPU.0");
                }
                Event::Finished(Outcome::Success { tab, midi, .. }) => {
                    assert!(saw_gpu);
                    assert!(std::fs::read_to_string(tab).unwrap().contains("e|"));
                    assert!(std::fs::read(midi).unwrap().starts_with(b"MThd"));
                    break;
                }
                Event::Finished(other) => panic!("Unexpected job result: {other:?}"),
            }
        }
    }

    #[test]
    #[ignore = "Needs the built native CLI and FFmpeg"]
    fn cancel_terminates_backend_before_it_writes_results() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("cancelled.txt");
        let options = Options {
            source: Source::File(fixture(directory.path())),
            output: output.clone(),
            tuning: "standard",
            device: "cpu",
            gpu_id: 0,
            threads: 1,
            clean_guitar: false,
            transcriber: "basic-pitch",
            muscriptor_model: None,
            hf_token: HfToken::default(),
            instruments: crate::instruments::Instruments::default(),
        };
        let job = Job::start(native_backend(), options).unwrap();
        loop {
            match job.events.recv_timeout(Duration::from_secs(15)).unwrap() {
                Event::Line(line) if line.starts_with("[2/4]") => job.cancel(),
                Event::Line(_) => {}
                Event::Finished(Outcome::Cancelled) => break,
                Event::Finished(other) => panic!("Expected cancellation, got {other:?}"),
            }
        }
        assert!(!output.exists());
        assert!(!output.with_extension("mid").exists());
    }
}

use crate::job::{self, Backend, Event, Job, Options, Outcome, Source};
use crate::playback::{Player, State as PlaybackState};
use eframe::egui::{self, Color32, RichText};
use std::{collections::VecDeque, fs, path::PathBuf, process::Command, time::Duration};

const BLUE: Color32 = Color32::from_rgb(37, 99, 235);
const MUTED: Color32 = Color32::from_rgb(91, 107, 128);
const TUNINGS: [(&str, &str); 4] = [
    ("Standard E", "standard"),
    ("Drop D", "drop_d"),
    ("D-Standard", "d_standard"),
    ("Halbton tiefer / Eb", "half_step_down"),
];
const DEVICES: [(&str, &str); 3] = [
    ("Automatisch", "auto"),
    ("Intel Arc", "openvino"),
    ("CPU", "cpu"),
];
const TRANSCRIBERS: [(&str, &str); 4] = [
    ("Basic Pitch", "basic-pitch"),
    ("MuScriptor Small · 0,41 GB", "muscriptor-small"),
    ("MuScriptor Medium · 1,23 GB", "muscriptor-medium"),
    ("MuScriptor Large · 5,47 GB", "muscriptor-large"),
];

pub struct AutoTab {
    backend: Option<Backend>,
    startup_error: Option<String>,
    from_url: bool,
    input: String,
    url: String,
    folder: String,
    tuning: usize,
    device: usize,
    gpu_id: u16,
    threads: u16,
    clean_guitar: bool,
    transcriber: usize,
    muscriptor_model: String,
    hf_token: String,
    show_hf_token: bool,
    job: Option<Job>,
    status: String,
    error: Option<String>,
    progress: Option<(usize, usize)>,
    preview: String,
    result: Option<(PathBuf, PathBuf)>,
    logs: VecDeque<String>,
    player: Player,
}

impl AutoTab {
    pub fn new(creation: &eframe::CreationContext<'_>) -> Self {
        let context = &creation.egui_ctx;
        context.set_visuals(egui::Visuals::light());
        let mut style = (*context.style()).clone();
        style.spacing.item_spacing = egui::vec2(10.0, 9.0);
        style.spacing.button_padding = egui::vec2(12.0, 7.0);
        style.visuals.panel_fill = Color32::from_rgb(242, 245, 250);
        style.visuals.selection.bg_fill = BLUE;
        style
            .text_styles
            .insert(egui::TextStyle::Body, egui::FontId::proportional(15.0));
        style
            .text_styles
            .insert(egui::TextStyle::Button, egui::FontId::proportional(15.0));
        context.set_style(style);
        let (backend, startup_error) = match Backend::discover() {
            Ok(backend) => (Some(backend), None),
            Err(error) => (None, Some(format!("{error:#}"))),
        };
        let folder = backend
            .as_ref()
            .map(|backend| backend.home.join("output"))
            .unwrap_or_else(|| PathBuf::from("output"));
        Self {
            backend,
            startup_error,
            from_url: false,
            input: String::new(),
            url: String::new(),
            folder: folder.to_string_lossy().into_owned(),
            tuning: 0,
            device: 0,
            gpu_id: 0,
            threads: 4,
            clean_guitar: false,
            transcriber: 0,
            muscriptor_model: String::new(),
            hf_token: String::new(),
            show_hf_token: false,
            job: None,
            status: "Bereit für deine Audiodatei".into(),
            error: None,
            progress: None,
            preview: String::new(),
            result: None,
            logs: VecDeque::new(),
            player: Player::default(),
        }
    }

    fn source(&self) -> Source {
        if self.from_url {
            Source::Url(self.url.trim().to_owned())
        } else {
            Source::File(PathBuf::from(self.input.trim()))
        }
    }

    fn folder_path(&self) -> PathBuf {
        let folder = PathBuf::from(self.folder.trim());
        if folder.is_absolute() {
            folder
        } else {
            self.backend
                .as_ref()
                .map(|backend| backend.home.join(&folder))
                .unwrap_or(folder)
        }
    }

    fn start(&mut self) {
        self.player.clear();
        self.error = None;
        self.result = None;
        self.preview.clear();
        self.logs.clear();
        self.progress = None;
        let Some(backend) = self.backend.clone() else {
            return;
        };
        if self.folder.trim().is_empty() {
            self.error = Some("Bitte einen Zielordner auswählen.".into());
            return;
        }
        let source = self.source();
        let options = Options {
            output: job::output_path(&self.folder_path(), &source),
            source,
            tuning: TUNINGS[self.tuning].1,
            device: DEVICES[self.device].1,
            gpu_id: self.gpu_id,
            threads: self.threads,
            clean_guitar: self.clean_guitar,
            transcriber: TRANSCRIBERS[self.transcriber].1,
            hf_token: if self.transcriber > 0 {
                job::HfToken::new(&self.hf_token)
            } else {
                job::HfToken::default()
            },
            muscriptor_model: if self.transcriber > 0 && !self.muscriptor_model.trim().is_empty() {
                Some(PathBuf::from(self.muscriptor_model.trim()))
            } else {
                None
            },
        };
        match Job::start(backend, options) {
            Ok(job) => {
                self.job = Some(job);
                self.status = "Verarbeitung wird gestartet …".into();
            }
            Err(error) => {
                self.error = Some(format!("{error:#}"));
                self.status = "Bitte Eingaben prüfen".into();
            }
        }
    }

    fn receive_events(&mut self) {
        let events: Vec<_> = self
            .job
            .as_ref()
            .map(|job| job.events.try_iter().collect())
            .unwrap_or_default();
        for event in events {
            match event {
                Event::Line(line) => {
                    if line.starts_with("[1/4]") {
                        self.status = "Audio wird heruntergeladen …".into();
                    }
                    if line.starts_with("[2/4]") {
                        self.status = "Audio und Gitarrentrennung werden vorbereitet …".into();
                    }
                    if line.contains("Compiling Demucs") {
                        self.status =
                            "Intel Arc wird vorbereitet – der erste Start kann länger dauern …"
                                .into();
                    }
                    if let Some(progress) = parse_chunks(&line) {
                        self.progress = Some(progress);
                        self.status = format!(
                            "Gitarrenspur isolieren · Abschnitt {} von {}",
                            progress.0, progress.1
                        );
                    }
                    if line.starts_with("[3/4]") {
                        self.progress = None;
                        self.status = "Noten werden erkannt …".into();
                    }
                    if line.starts_with("[4/4]") {
                        self.status = "Tabulatur und MIDI werden gespeichert …".into();
                    }
                    self.logs.push_back(line.chars().take(4_000).collect());
                    while self.logs.len() > 400 {
                        self.logs.pop_front();
                    }
                }
                Event::Finished(outcome) => {
                    self.job = None;
                    self.progress = None;
                    match outcome {
                        Outcome::Success { tab, midi, elapsed } => match fs::read_to_string(&tab) {
                            Ok(preview) => {
                                self.preview = preview;
                                self.status = format!(
                                    "Fertig – deine Tabs sind bereit ({:.1} s)",
                                    elapsed.as_secs_f64()
                                );
                                self.result = Some((tab, midi));
                            }
                            Err(error) => {
                                self.status = "Ausgabe konnte nicht geöffnet werden".into();
                                self.error = Some(error.to_string());
                            }
                        },
                        Outcome::Cancelled => self.status = "Verarbeitung abgebrochen".into(),
                        Outcome::Failed(message) => {
                            self.status = "Verarbeitung fehlgeschlagen".into();
                            self.error = Some(message);
                        }
                    }
                }
            }
        }
    }

    fn form(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).fill(Color32::WHITE).inner_margin(16.0).corner_radius(10.0).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong("1  Audio auswählen");
                ui.add_space(16.0);
                ui.selectable_value(&mut self.from_url, false, "Datei");
                ui.selectable_value(&mut self.from_url, true, "Link");
            });
            if self.from_url {
                let label = ui.label("Audio-Link");
                ui.add(egui::TextEdit::singleline(&mut self.url).desired_width(f32::INFINITY)
                    .hint_text("https://… / direkter Audio-Link")).labelled_by(label.id);
                ui.label(RichText::new("Direkte Audio-Links werden unterstützt. YouTube ist experimentell.").small().color(MUTED));
            } else {
                let label = ui.label("Audiodatei");
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.input).desired_width((ui.available_width() - 135.0).max(200.0))
                        .hint_text("MP3, WAV, FLAC … oder Datei hierher ziehen")).labelled_by(label.id);
                    if ui.button("Datei wählen …").clicked()
                        && let Some(file) = rfd::FileDialog::new().set_title("Audiodatei auswählen")
                            .add_filter("Audio", &["mp3", "wav", "flac", "ogg", "m4a", "aac", "opus", "aiff", "wma"])
                            .add_filter("Alle Dateien", &["*"]).pick_file() {
                        self.input = file.to_string_lossy().into_owned();
                        self.error = None;
                    }
                });
            }
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label("Stimmung");
                egui::ComboBox::from_id_salt("tuning").selected_text(TUNINGS[self.tuning].0).width(160.0).show_ui(ui, |ui| {
                    for (index, (label, _)) in TUNINGS.iter().enumerate() { ui.selectable_value(&mut self.tuning, index, *label); }
                });
                ui.add_space(20.0);
                ui.label("Verarbeitung");
                egui::ComboBox::from_id_salt("device").selected_text(DEVICES[self.device].0).width(150.0).show_ui(ui, |ui| {
                    for (index, (label, _)) in DEVICES.iter().enumerate() { ui.selectable_value(&mut self.device, index, *label); }
                });
            });
            ui.checkbox(&mut self.clean_guitar, "Reine Gitarrenaufnahme – Gitarrentrennung überspringen");
            ui.horizontal(|ui| {
                ui.label("Notenerkennung");
                egui::ComboBox::from_id_salt("transcriber").selected_text(TRANSCRIBERS[self.transcriber].0).width(250.0).show_ui(ui, |ui| {
                    for (index, (label, _)) in TRANSCRIBERS.iter().enumerate() {
                        if ui.selectable_value(&mut self.transcriber, index, *label).changed() { self.muscriptor_model.clear(); }
                    }
                });
            });
            if self.transcriber > 0 {
                let token_label = ui.label("Hugging-Face-Token (optional)");
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.hf_token)
                        .password(!self.show_hf_token)
                        .desired_width((ui.available_width() - 205.0).max(180.0))
                        .hint_text("hf_… · leer = vorhandene Hugging-Face-Anmeldung"))
                        .labelled_by(token_label.id);
                    ui.checkbox(&mut self.show_hf_token, "Anzeigen");
                    if ui.button("Leeren").clicked() {
                        self.hf_token.clear();
                        self.show_hf_token = false;
                    }
                });
                ui.label(RichText::new("Nur für diese Sitzung. Verwende den Token des Kontos mit Modellfreigabe.").small().color(MUTED));
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("MuScriptor: CC BY-NC 4.0 · Modellfreigabe für den Download erforderlich.").small().color(MUTED));
                    ui.hyperlink_to("Modellseite", format!("https://huggingface.co/MuScriptor/{}", TRANSCRIBERS[self.transcriber].1));
                    ui.hyperlink_to("Token erstellen / verwalten", "https://huggingface.co/settings/tokens");
                });
                ui.label(RichText::new("Vorhandene Modelle werden aus dem lokalen Hugging-Face-Cache geladen.").small().color(MUTED));
            }
            ui.separator();
            ui.strong("2  Ausgabeordner");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.folder).desired_width((ui.available_width() - 135.0).max(200.0))
                    .hint_text("Ordner für Tabulatur und MIDI"));
                if ui.button("Ordner wählen …").clicked()
                    && let Some(folder) = rfd::FileDialog::new().set_title("Ausgabeordner auswählen")
                        .set_directory(self.folder_path()).pick_folder() {
                    self.folder = folder.to_string_lossy().into_owned();
                }
            });
            ui.label(RichText::new("Tabulatur (.txt) und MIDI (.mid). Vorhandene Ergebnisse bleiben erhalten.").small().color(MUTED));
            egui::CollapsingHeader::new("Weitere Einstellungen").show(ui, |ui| {
                if self.transcriber > 0 {
                    ui.label("Lokale MuScriptor-Modelldatei (optional)");
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.muscriptor_model).desired_width((ui.available_width() - 135.0).max(200.0)).hint_text("Automatisch aus Cache / Download"));
                        if ui.button("Modell wählen …").clicked()
                            && let Some(path) = rfd::FileDialog::new().set_title("MuScriptor-Modell auswählen").add_filter("Safetensors", &["safetensors"]).pick_file() {
                            self.muscriptor_model = path.to_string_lossy().into_owned();
                        }
                    });
                }
                ui.horizontal(|ui| {
                    ui.label("Intel-GPU-Index"); ui.add(egui::DragValue::new(&mut self.gpu_id).range(0..=15));
                    ui.add_space(20.0);
                    ui.label("CPU-Threads"); ui.add(egui::DragValue::new(&mut self.threads).range(1..=64));
                });
                ui.small("GPU.0 ist normalerweise die integrierte Intel-GPU. Für eine zweite Intel-GPU: Index 1.");
            });
        });
    }

    fn results(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("Tab-Vorschau");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(self.job.is_none(), egui::Button::new("Ergebnis öffnen …"))
                    .clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .set_title("Gespeicherte AutoTab-Tabs öffnen")
                        .set_directory(self.folder_path())
                        .add_filter("AutoTab-Tabulatur", &["txt"])
                        .pick_file()
                {
                    match read_saved_result(&path) {
                        Ok((preview, midi)) => {
                            self.player.clear();
                            self.preview = preview;
                            self.result = Some((path, midi));
                            self.error = None;
                            self.status = "Ergebnis geladen – MIDI-Vorschau ist verfügbar".into();
                        }
                        Err(error) => self.error = Some(format!("{error:#}")),
                    }
                }
                if self.result.is_some() && ui.button("Tabs kopieren").clicked() {
                    ui.ctx().copy_text(self.preview.clone());
                }
                if self.result.is_some() && ui.button("Ausgabeordner öffnen").clicked() {
                    let folder = self.result.as_ref().and_then(|(tab, _)| tab.parent());
                    if let Some(folder) = folder {
                        let mut command = Command::new("explorer.exe");
                        job::hide_console(&mut command);
                        if let Err(error) = command.arg(folder).spawn() {
                            self.error = Some(error.to_string());
                        }
                    }
                }
            });
        });
        if let Some((tab, midi)) = &self.result {
            ui.label(
                RichText::new(format!("{}\n{}", tab.display(), midi.display()))
                    .small()
                    .color(MUTED),
            );
        }
        self.playback_controls(ui);
        egui::CollapsingHeader::new("Verarbeitungsprotokoll")
            .id_salt("log")
            .show(ui, |ui| {
                let mut text = self.logs.iter().cloned().collect::<Vec<_>>().join("\n");
                egui::ScrollArea::both()
                    .max_height(140.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut text)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY)
                                .interactive(false),
                        );
                    });
            });
        let height = ui.available_height().clamp(160.0, 360.0);
        egui::Frame::group(ui.style())
            .fill(Color32::WHITE)
            .inner_margin(14.0)
            .corner_radius(10.0)
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.set_min_height((height - 32.0).max(80.0));
                if self.preview.is_empty() {
                    ui.label(
                        RichText::new("Deine Gitarrentabs erscheinen hier nach der Verarbeitung.")
                            .color(MUTED),
                    );
                } else {
                    egui::ScrollArea::both()
                        .auto_shrink([false, false])
                        .max_height((height - 32.0).max(80.0))
                        .show(ui, |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut self.preview)
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(f32::INFINITY)
                                    .interactive(false),
                            );
                        });
                }
            });
    }

    fn playback_controls(&mut self, ui: &mut egui::Ui) {
        let state = self.player.state();
        ui.horizontal_wrapped(|ui| {
            let label = match state {
                PlaybackState::Playing => "Pause",
                PlaybackState::Paused => "Fortsetzen",
                PlaybackState::Finished => "Erneut abspielen",
                _ => "Abspielen",
            };
            if ui
                .add_enabled(
                    self.result.is_some() && self.job.is_none(),
                    egui::Button::new(label),
                )
                .clicked()
            {
                let result = (|| -> anyhow::Result<()> {
                    if state == PlaybackState::Playing {
                        self.player.pause()
                    } else {
                        if state == PlaybackState::Empty {
                            let (_, midi) = self.result.as_ref().ok_or_else(|| {
                                anyhow::anyhow!("Bitte zuerst ein Ergebnis erstellen oder öffnen")
                            })?;
                            self.player.load(midi)?;
                        }
                        self.player.play()
                    }
                })();
                match result {
                    Ok(()) => self.error = None,
                    Err(error) => {
                        self.player.clear();
                        self.error = Some(format!("{error:#}"));
                    }
                }
                ui.ctx().request_repaint();
            }
            if ui
                .add_enabled(state != PlaybackState::Empty, egui::Button::new("Stopp"))
                .clicked()
            {
                if let Err(error) = self.player.stop() {
                    self.player.clear();
                    self.error = Some(format!("{error:#}"));
                }
                ui.ctx().request_repaint();
            }
            if !self.player.duration().is_zero() {
                ui.label(format!(
                    "{} / {}",
                    time_label(self.player.position()),
                    time_label(self.player.duration())
                ));
            }
            ui.label(
                RichText::new("MIDI-Vorschau · Synthesizer-Klang")
                    .small()
                    .color(MUTED),
            );
        });
        if !self.player.duration().is_zero() {
            let fraction = (self.player.position().as_secs_f64()
                / self.player.duration().as_secs_f64())
            .clamp(0.0, 1.0);
            ui.add(egui::ProgressBar::new(fraction as f32).desired_height(5.0));
        }
    }
}

impl eframe::App for AutoTab {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        self.receive_events();
        if let Err(error) = self.player.poll() {
            self.player.clear();
            self.error = Some(format!("{error:#}"));
        }
        let running = self.job.is_some();
        if running || self.player.state() == PlaybackState::Playing {
            context.request_repaint_after(Duration::from_millis(100));
        }
        if !running
            && let Some(path) = context.input(|input| {
                input
                    .raw
                    .dropped_files
                    .iter()
                    .find_map(|file| file.path.clone())
            })
        {
            self.input = path.to_string_lossy().into_owned();
            self.from_url = false;
        }
        egui::CentralPanel::default()
            .frame(
                egui::Frame::NONE
                    .fill(Color32::from_rgb(242, 245, 250))
                    .inner_margin(22.0),
            )
            .show(context, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("main_scroll")
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let (rect, _) = ui
                                .allocate_exact_size(egui::vec2(42.0, 42.0), egui::Sense::hover());
                            ui.painter().rect_filled(rect, 9.0, BLUE);
                            for row in 0..6 {
                                let y = rect.top() + 11.0 + row as f32 * 4.0;
                                ui.painter().line_segment(
                                    [
                                        egui::pos2(rect.left() + 8.0, y),
                                        egui::pos2(rect.right() - 8.0, y),
                                    ],
                                    egui::Stroke::new(1.0, Color32::WHITE),
                                );
                            }
                            ui.vertical(|ui| {
                                ui.label(
                                    RichText::new("AutoTab")
                                        .size(26.0)
                                        .strong()
                                        .color(Color32::from_rgb(23, 39, 62)),
                                );
                                ui.label(
                                    RichText::new("Audio in Gitarrentabs verwandeln").color(MUTED),
                                );
                            });
                        });
                        ui.add_space(10.0);
                        if let Some(error) = &self.startup_error {
                            ui.colored_label(Color32::from_rgb(174, 38, 45), error);
                        }
                        ui.add_enabled_ui(!running, |ui| self.form(ui));
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    !running && self.backend.is_some(),
                                    egui::Button::new(
                                        RichText::new("Tabs erstellen").color(Color32::WHITE),
                                    )
                                    .fill(BLUE)
                                    .min_size(egui::vec2(150.0, 36.0)),
                                )
                                .clicked()
                            {
                                self.start();
                            }
                            if running && ui.button("Abbrechen").clicked() {
                                if let Some(job) = &self.job {
                                    job.cancel();
                                }
                                self.status = "Verarbeitung wird abgebrochen …".into();
                            }
                            if running {
                                ui.spinner();
                            }
                            ui.label(&self.status);
                        });
                        if let Some((current, total)) = self.progress {
                            ui.add(
                                egui::ProgressBar::new(current as f32 / total.max(1) as f32).text(
                                    format!("Gitarrentrennung: {current} / {total} Abschnitte"),
                                ),
                            );
                        }
                        if let Some(error) = &self.error {
                            ui.colored_label(Color32::from_rgb(174, 38, 45), error);
                        }
                        ui.add_space(2.0);
                        self.results(ui);
                    });
            });
    }
}

fn read_saved_result(path: &std::path::Path) -> anyhow::Result<(String, PathBuf)> {
    use anyhow::{Context, ensure};
    let preview = fs::read_to_string(path).context("Die Tabulatur konnte nicht geöffnet werden")?;
    let midi = path.with_extension("mid");
    ensure!(
        midi.is_file(),
        "Die passende MIDI-Datei fehlt: {}. Bitte die Tabs erneut erzeugen, um sie abspielen zu können.",
        midi.display()
    );
    Ok((preview, midi))
}

fn time_label(duration: Duration) -> String {
    let seconds = duration.as_secs();
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

fn parse_chunks(line: &str) -> Option<(usize, usize)> {
    let rest = line.trim().strip_prefix("Demucs chunk ")?;
    let fraction = rest.split(':').next()?;
    let (current, total) = fraction.split_once('/')?;
    let (current, total) = (current.parse().ok()?, total.parse().ok()?);
    if total == 0 || current > total {
        None
    } else {
        Some((current, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn progress_uses_real_segment_counts() {
        assert_eq!(parse_chunks("  Demucs chunk 1/12: 1.23s"), Some((1, 12)));
        assert_eq!(parse_chunks("  Demucs chunk 3/2: 1.23s"), None);
        assert_eq!(parse_chunks("  Intel GPU: Arc"), None);
    }

    #[test]
    fn saved_tabs_need_their_corresponding_midi() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("song_tab.txt");
        fs::write(&path, "e|--0--|\n").unwrap();
        assert!(read_saved_result(&path).is_err());
        fs::write(path.with_extension("mid"), b"MThd").unwrap();
        let (preview, midi) = read_saved_result(&path).unwrap();
        assert!(preview.contains("e|"));
        assert_eq!(midi, path.with_extension("mid"));
    }
}

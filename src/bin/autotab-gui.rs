#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

#[path = "../gui/app.rs"]
mod app;
#[path = "../instruments.rs"]
#[allow(dead_code)] // The shared catalog also provides CLI inference helpers.
mod instruments;
#[path = "../gui/job.rs"]
mod job;
#[path = "../midi_input.rs"]
#[allow(dead_code)] // Import/export helpers are shared with the CLI backend.
mod midi_input;
#[path = "../gui/playback.rs"]
mod playback;
#[path = "../rhythm.rs"]
#[allow(dead_code)] // Export timing helpers are used by the CLI backend.
mod rhythm;

fn main() {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([900.0, 780.0])
            .with_min_inner_size([780.0, 640.0]),
        ..Default::default()
    };
    if let Err(error) = eframe::run_native(
        "AutoTab",
        options,
        Box::new(|creation| Ok(Box::new(app::AutoTab::new(creation)))),
    ) {
        rfd::MessageDialog::new()
            .set_title("AutoTab konnte nicht starten")
            .set_description(error.to_string())
            .set_level(rfd::MessageLevel::Error)
            .show();
    }
}

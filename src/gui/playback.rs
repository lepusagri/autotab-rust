//! MIDI preview using the Windows sequencer and its installed synthesizer.
//! Commands use device IDs and typed parameters, never shell/command strings.
use anyhow::{Context, Result, ensure};
use std::{
    fs::File,
    io::Read,
    os::windows::ffi::OsStrExt,
    path::Path,
    time::{Duration, Instant},
};
use windows_sys::Win32::Media::Multimedia::*;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum State {
    #[default]
    Empty,
    Ready,
    Playing,
    Paused,
    Finished,
}

#[derive(Default)]
pub struct Player {
    device: Option<Device>,
    state: State,
    position: Duration,
    duration: Duration,
    started: Option<(Instant, Duration)>,
}

impl Player {
    pub fn state(&self) -> State {
        self.state
    }
    pub fn position(&self) -> Duration {
        self.position
    }
    pub fn duration(&self) -> Duration {
        self.duration
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        self.clear();
        let mut header = [0_u8; 4];
        File::open(path)
            .context("Zugehörige MIDI-Datei konnte nicht geöffnet werden")?
            .read_exact(&mut header)
            .context("Die MIDI-Datei ist leer oder beschädigt")?;
        ensure!(&header == b"MThd", "Keine unterstützte Standard-MIDI-Datei");
        let device = Device::open(path)?;
        let duration = Duration::from_millis(device.query(MCI_STATUS_LENGTH)? as u64);
        ensure!(
            !duration.is_zero(),
            "Die MIDI-Datei enthält keine abspielbare Notenfolge"
        );
        self.device = Some(device);
        self.duration = duration;
        self.state = State::Ready;
        Ok(())
    }

    pub fn play(&mut self) -> Result<()> {
        let device = self
            .device
            .as_ref()
            .context("Bitte zuerst ein Ergebnis mit MIDI-Datei laden")?;
        if self.state == State::Finished {
            self.position = Duration::ZERO;
        }
        device.play(self.position.as_millis().min(u128::from(u32::MAX)) as u32)?;
        self.started = Some((Instant::now(), self.position));
        self.state = State::Playing;
        Ok(())
    }

    pub fn pause(&mut self) -> Result<()> {
        if self.state != State::Playing {
            return Ok(());
        }
        let device = self
            .device
            .as_ref()
            .context("MIDI-Wiedergabe ist nicht geöffnet")?;
        device.simple(MCI_PAUSE)?;
        // MCISEQ reports paused playback as stopped. Keep our paused state and
        // resume explicitly from the saved position instead of inferring it.
        self.position = self.elapsed_position();
        self.started = None;
        self.state = State::Paused;
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        if let Some(device) = &self.device {
            device.simple(MCI_STOP)?;
            device.rewind()?;
            self.state = State::Ready;
        }
        self.position = Duration::ZERO;
        self.started = None;
        Ok(())
    }

    pub fn poll(&mut self) -> Result<()> {
        if self.state != State::Playing {
            return Ok(());
        }
        let device = self
            .device
            .as_ref()
            .context("MIDI-Wiedergabe ist nicht geöffnet")?;
        if device.query(MCI_STATUS_MODE)? as u32 != MCI_MODE_PLAY {
            self.state = State::Finished;
            self.position = self.duration;
            self.started = None;
        } else {
            self.position = self.elapsed_position();
        }
        Ok(())
    }

    fn elapsed_position(&self) -> Duration {
        // MCISEQ's native position can stay at the last MIDI event during a
        // rest. Use a monotonic clock for smooth progress and pause/resume,
        // while the device's mode remains authoritative for end-of-playback.
        self.started
            .map(|(started, from)| from.saturating_add(started.elapsed()))
            .unwrap_or(self.position)
            .min(self.duration)
    }
}

/// Sole owner of an MCI device. Closing the device also silences its playback.
struct Device {
    id: u32,
}

impl Device {
    fn open(path: &Path) -> Result<Self> {
        let path = std::path::absolute(path)?;
        let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let device_type: Vec<u16> = "sequencer".encode_utf16().chain(Some(0)).collect();
        let mut parameters = MCI_OPEN_PARMSW {
            dwCallback: 0,
            wDeviceID: 0,
            lpstrDeviceType: device_type.as_ptr(),
            lpstrElementName: path.as_ptr(),
            lpstrAlias: std::ptr::null(),
        };
        // SAFETY: all pointers refer to live, NUL-terminated UTF-16 buffers; MCI
        // consumes the parameter block synchronously and returns the owned ID.
        check(unsafe {
            mciSendCommandW(
                0,
                MCI_OPEN,
                (MCI_OPEN_TYPE | MCI_OPEN_ELEMENT | MCI_WAIT) as usize,
                &mut parameters as *mut _ as usize,
            )
        })?;
        let device = Self {
            id: parameters.wDeviceID,
        };
        let mut format = MCI_SET_PARMS {
            dwCallback: 0,
            dwTimeFormat: MCI_FORMAT_MILLISECONDS,
            dwAudio: 0,
        };
        // SAFETY: device is open; the typed parameter block stays alive through the call.
        check(unsafe {
            mciSendCommandW(
                device.id,
                MCI_SET,
                (MCI_SET_TIME_FORMAT | MCI_WAIT) as usize,
                &mut format as *mut _ as usize,
            )
        })?;
        Ok(device)
    }

    fn simple(&self, command: u32) -> Result<()> {
        // SAFETY: callers pass only parameterless PAUSE/STOP commands for this live ID.
        check(unsafe { mciSendCommandW(self.id, command, MCI_WAIT as usize, 0) })
    }

    fn query(&self, item: i32) -> Result<usize> {
        let mut status = MCI_STATUS_PARMS {
            dwCallback: 0,
            dwReturn: 0,
            dwItem: item as u32,
            dwTrack: 0,
        };
        // SAFETY: device is open and the API writes into a valid status block.
        check(unsafe {
            mciSendCommandW(
                self.id,
                MCI_STATUS,
                (MCI_STATUS_ITEM | MCI_WAIT) as usize,
                &mut status as *mut _ as usize,
            )
        })?;
        Ok(status.dwReturn)
    }

    fn play(&self, position: u32) -> Result<()> {
        let mut play = MCI_PLAY_PARMS {
            dwCallback: 0,
            dwFrom: position,
            dwTo: 0,
        };
        // SAFETY: the live device consumes the start position synchronously; no
        // WAIT flag is passed, so playback does not block the UI thread.
        check(unsafe {
            mciSendCommandW(
                self.id,
                MCI_PLAY,
                MCI_FROM as usize,
                &mut play as *mut _ as usize,
            )
        })
    }

    fn rewind(&self) -> Result<()> {
        let mut seek = MCI_SEEK_PARMS {
            dwCallback: 0,
            dwTo: 0,
        };
        // SAFETY: the live device receives its matching seek parameter structure.
        check(unsafe {
            mciSendCommandW(
                self.id,
                MCI_SEEK,
                (MCI_SEEK_TO_START | MCI_WAIT) as usize,
                &mut seek as *mut _ as usize,
            )
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: this object uniquely owns the device ID returned by OPEN.
        unsafe {
            mciSendCommandW(self.id, MCI_CLOSE, MCI_WAIT as usize, 0);
        }
    }
}

fn check(error: u32) -> Result<()> {
    if error == 0 {
        return Ok(());
    }
    let mut text = [0_u16; 512];
    // SAFETY: the output buffer is writable for the explicitly supplied length.
    let found = unsafe { mciGetErrorStringW(error, text.as_mut_ptr(), text.len() as u32) };
    let end = text
        .iter()
        .position(|&value| value == 0)
        .unwrap_or(text.len());
    let message = if found != 0 {
        String::from_utf16_lossy(&text[..end])
    } else {
        format!("Windows-MIDI-Fehler {error}")
    };
    anyhow::bail!("MIDI-Vorschau: {message}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silent_midi(path: &Path) {
        // Format 0, 480 PPQN, 120 BPM. Only note-off messages, never an audible
        // note-on. The final event at tick 960 makes the fixture two beats long.
        let track: &[u8] = &[
            0, 0xff, 0x51, 3, 7, 0xa1, 0x20, 0, 0x80, 69, 0, 0x87, 0x40, 0x80, 69, 0, 0, 0xff,
            0x2f, 0,
        ];
        let mut data = b"MThd\0\0\0\x06\0\0\0\x01\x01\xe0MTrk".to_vec();
        data.extend_from_slice(&(track.len() as u32).to_be_bytes());
        data.extend_from_slice(track);
        std::fs::write(path, data).unwrap();
    }

    #[test]
    fn invalid_file_does_not_open_playback() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("invalid.mid");
        std::fs::write(&path, "not MIDI").unwrap();
        let mut player = Player::default();
        assert!(player.load(&path).is_err());
        assert_eq!(player.state(), State::Empty);
        assert_eq!(player.position(), Duration::ZERO);
    }

    #[test]
    #[ignore = "Requires the Windows MIDI sequencer; uses a silent MIDI fixture"]
    fn native_silent_play_pause_stop_and_replay() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("Vorschau März mit Leerzeichen.mid");
        silent_midi(&path);
        let mut player = Player::default();
        player.load(&path).unwrap();
        // The Windows sequencer rounds tick-to-millisecond conversions (999 ms
        // for this exact one-second sequence on some drivers).
        assert!((990..=1_010).contains(&player.duration().as_millis()));
        player.play().unwrap();
        std::thread::sleep(Duration::from_millis(200));
        player.poll().unwrap();
        assert_eq!(player.state(), State::Playing);
        assert!(player.position() > Duration::ZERO);
        player.pause().unwrap();
        let paused_at = player.position();
        std::thread::sleep(Duration::from_millis(100));
        player.poll().unwrap();
        assert_eq!(player.state(), State::Paused);
        assert_eq!(player.position(), paused_at);
        player.play().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        player.poll().unwrap();
        assert_eq!(player.state(), State::Playing);
        assert!(player.position() > paused_at);
        player.stop().unwrap();
        assert_eq!(player.state(), State::Ready);
        assert_eq!(player.position(), Duration::ZERO);
        player.play().unwrap();
        std::thread::sleep(Duration::from_millis(1_200));
        player.poll().unwrap();
        assert_eq!(player.state(), State::Finished);
        player.play().unwrap();
        assert_eq!(player.state(), State::Playing);
        player.clear();
        assert_eq!(player.state(), State::Empty);

        // Exercise AutoTab's actual type-1 exporter as well. Zero amplitude
        // writes note-on velocity zero (equivalent to note-off), so it is silent.
        let notes = [basic_pitch::notes::NoteEvent {
            start_frame: 0,
            end_frame: 129,
            start_s: 0.0,
            end_s: 1.5,
            pitch: 69,
            amplitude: 0.0,
            pitch_bends: None,
        }];
        let data = basic_pitch::midi::note_events_to_midi_bytes(
            &notes,
            &basic_pitch::midi::MidiParams::default(),
        )
        .unwrap();
        std::fs::write(&path, data).unwrap();
        player.load(&path).unwrap();
        assert!(
            (1_490..=1_510).contains(&player.duration().as_millis()),
            "AutoTab MIDI duration: {:?}",
            player.duration()
        );
        assert_eq!(player.state(), State::Ready);
    }
}

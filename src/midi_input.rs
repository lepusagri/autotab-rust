//! Standard MIDI file import with a global tempo map and explicit track selection.
use crate::rhythm::Meter;
use anyhow::{Context, Result, ensure};
use basic_pitch::{
    NoteEvent,
    midi::midly::{Format, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind},
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt, fs,
    path::Path,
    str::FromStr,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Selection {
    #[default]
    Auto,
    All,
    Track(usize),
}
impl FromStr for Selection {
    type Err = String;
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "all" => Ok(Self::All),
            _ => value
                .parse()
                .map(Self::Track)
                .map_err(|_| "MIDI track must be auto, all, or a zero-based track index".into()),
        }
    }
}
impl fmt::Display for Selection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::All => f.write_str("all"),
            Self::Track(index) => write!(f, "{index}"),
        }
    }
}

pub fn is_midi(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("mid") || e.eq_ignore_ascii_case("midi"))
}

#[derive(Clone, Debug, Serialize)]
pub struct TrackInfo {
    pub index: usize,
    pub name: String,
    pub melodic_notes: usize,
    pub drum_notes: usize,
    pub guitar_notes: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct Tempo {
    pub tick: u64,
    pub micros: u32,
}
#[derive(Clone, Debug)]
pub struct MeterChange {
    pub tick: u64,
    pub meter: Meter,
}
#[derive(Clone)]
pub struct ImportedNote {
    pub note: NoteEvent,
    pub start_tick: u64,
    pub end_tick: u64,
    pub track: usize,
    pub channel: u8,
    pub port: u8,
    pub program: u8,
    pub velocity: u8,
    on: (usize, usize),
    off: Option<(usize, usize)>,
}
pub struct MidiFile {
    pub ppq: u16,
    pub tracks: Vec<TrackInfo>,
    pub tempos: Vec<Tempo>,
    pub meters: Vec<MeterChange>,
    pub notes: Vec<ImportedNote>,
    pub repaired_note_offs: usize,
    bytes: Vec<u8>,
}

impl MidiFile {
    pub fn read(path: &Path) -> Result<Self> {
        Self::parse(fs::read(path).with_context(|| format!("Cannot read MIDI {}", path.display()))?)
    }
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let smf =
            Smf::parse(&bytes).map_err(|e| anyhow::anyhow!("Invalid Standard MIDI file: {e}"))?;
        ensure!(
            smf.header.format != Format::Sequential,
            "MIDI format 2 has independent timelines and is not supported"
        );
        let Timing::Metrical(ppq) = smf.header.timing else {
            anyhow::bail!("SMPTE MIDI timing is not supported for tablature notation");
        };
        let ppq = ppq.as_int();
        ensure!(ppq > 0, "MIDI ticks per quarter must be positive");
        let mut tracks = Vec::new();
        let mut events = Vec::new();
        let mut ends = Vec::new();
        for (track, source) in smf.tracks.iter().enumerate() {
            let mut tick = 0u64;
            let mut name = format!("Track {track}");
            for (index, event) in source.iter().enumerate() {
                tick += u64::from(event.delta.as_int());
                if let TrackEventKind::Meta(MetaMessage::TrackName(value)) = event.kind {
                    name = String::from_utf8_lossy(value).trim().to_owned();
                }
                events.push((tick, track, index, event.kind));
            }
            tracks.push(TrackInfo {
                index: track,
                name,
                melodic_notes: 0,
                drum_notes: 0,
                guitar_notes: 0,
            });
            ends.push(tick);
        }
        events.sort_by_key(|e| (e.0, e.1, e.2));
        let mut tempos = BTreeMap::from([(0, 500_000u32)]);
        let mut meters = BTreeMap::from([(0, Meter::default())]);
        let mut programs = BTreeMap::<(u8, u8), u8>::new();
        let mut ports = vec![0u8; smf.tracks.len()];
        type Pending = (u64, u8, u8, (usize, usize));
        let mut pending = BTreeMap::<(usize, u8, u8, u8), VecDeque<Pending>>::new();
        let mut raw = Vec::new();
        for (tick, track, index, kind) in events {
            match kind {
                TrackEventKind::Meta(MetaMessage::MidiPort(value)) => ports[track] = value.as_int(),
                TrackEventKind::Meta(MetaMessage::Tempo(value)) => {
                    ensure!(value.as_int() > 0, "MIDI contains a zero tempo");
                    tempos.insert(tick, value.as_int());
                }
                TrackEventKind::Meta(MetaMessage::TimeSignature(n, d, _, _)) => {
                    let denominator = 1u32
                        .checked_shl(u32::from(d))
                        .context("Invalid MIDI meter")?;
                    let meter: Meter = format!("{n}/{denominator}")
                        .parse()
                        .map_err(anyhow::Error::msg)?;
                    meters.insert(tick, meter);
                }
                TrackEventKind::Midi { channel, message } => {
                    let channel = channel.as_int();
                    let port = ports[track];
                    match message {
                        MidiMessage::ProgramChange { program } => {
                            programs.insert((port, channel), program.as_int());
                        }
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            pending
                                .entry((track, port, channel, key.as_int()))
                                .or_default()
                                .push_back((
                                    tick,
                                    vel.as_int(),
                                    programs.get(&(port, channel)).copied().unwrap_or(0),
                                    (track, index),
                                ));
                        }
                        MidiMessage::NoteOff { key, .. } | MidiMessage::NoteOn { key, .. } => {
                            if let Some(start) = pending
                                .get_mut(&(track, port, channel, key.as_int()))
                                .and_then(VecDeque::pop_front)
                            {
                                raw.push((
                                    track,
                                    port,
                                    channel,
                                    key.as_int(),
                                    start,
                                    tick,
                                    Some((track, index)),
                                ));
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        let mut repaired = 0;
        for ((track, port, channel, pitch), starts) in pending {
            for start in starts {
                raw.push((
                    track,
                    port,
                    channel,
                    pitch,
                    start,
                    ends[track].max(start.0 + 1),
                    None,
                ));
                repaired += 1;
            }
        }
        // Collapse repeated metadata, including files repeating the same tempo in every track.
        let mut previous = None;
        let tempos: Vec<_> = tempos
            .into_iter()
            .filter_map(|(tick, micros)| {
                let changed = previous != Some(micros);
                previous = Some(micros);
                changed.then_some(Tempo { tick, micros })
            })
            .collect();
        let mut previous = None;
        let meters: Vec<_> = meters
            .into_iter()
            .filter_map(|(tick, meter)| {
                let changed = previous != Some(meter);
                previous = Some(meter);
                changed.then_some(MeterChange { tick, meter })
            })
            .collect();
        let mut result = Self {
            ppq,
            tracks,
            tempos,
            meters,
            notes: vec![],
            repaired_note_offs: repaired,
            bytes: bytes.clone(),
        };
        for (track, port, channel, pitch, (start, velocity, program, on), end, off) in raw {
            ensure!(
                end > start,
                "MIDI contains a zero-length note in track {track}, pitch {pitch}"
            );
            let info = &mut result.tracks[track];
            if channel == 9 {
                info.drum_notes += 1;
                continue;
            }
            info.melodic_notes += 1;
            info.guitar_notes += usize::from((24..=31).contains(&program));
            result.notes.push(ImportedNote {
                note: NoteEvent {
                    pitch,
                    start_s: result.tick_to_seconds(start as f64),
                    end_s: result.tick_to_seconds(end as f64),
                    amplitude: f32::from(velocity) / 127.0,
                    start_frame: 0,
                    end_frame: 0,
                    pitch_bends: None,
                },
                start_tick: start,
                end_tick: end,
                track,
                channel,
                port,
                program,
                velocity,
                on,
                off,
            });
        }
        result.notes.sort_by(|a, b| {
            a.start_tick
                .cmp(&b.start_tick)
                .then(a.note.pitch.cmp(&b.note.pitch))
                .then(a.track.cmp(&b.track))
        });
        Ok(result)
    }

    pub fn tick_to_seconds(&self, tick: f64) -> f64 {
        let mut seconds = 0.0;
        let mut previous = 0.0;
        let mut tempo = 500_000;
        for change in &self.tempos {
            if change.tick as f64 > tick {
                break;
            }
            seconds += (change.tick as f64 - previous) / f64::from(self.ppq) * f64::from(tempo)
                / 1_000_000.0;
            previous = change.tick as f64;
            tempo = change.micros;
        }
        seconds + (tick - previous) / f64::from(self.ppq) * f64::from(tempo) / 1_000_000.0
    }

    pub fn select(&self, selection: Selection) -> Result<Vec<ImportedNote>> {
        if let Selection::Track(index) = selection {
            ensure!(
                index < self.tracks.len(),
                "MIDI track {index} does not exist"
            );
        }
        let guitar = self.notes.iter().any(|n| (24..=31).contains(&n.program));
        let result: Vec<_> = self
            .notes
            .iter()
            .filter(|n| match selection {
                Selection::Auto => !guitar || (24..=31).contains(&n.program),
                Selection::All => true,
                Selection::Track(track) => n.track == track,
            })
            .cloned()
            .collect();
        ensure!(
            !result.is_empty(),
            "The selected MIDI track contains no melodic notes (drums are excluded)"
        );
        Ok(result)
    }

    pub fn preview(&self, notes: &[ImportedNote]) -> Result<Vec<u8>> {
        let smf = Smf::parse(&self.bytes).map_err(|e| anyhow::anyhow!("Invalid MIDI: {e}"))?;
        let selected: BTreeSet<_> = notes
            .iter()
            .flat_map(|n| std::iter::once(n.on).chain(n.off))
            .collect();
        let channels: BTreeSet<_> = notes.iter().map(|n| (n.port, n.channel)).collect();
        let mut tracks = Vec::new();
        for (track, source) in smf.tracks.iter().enumerate() {
            let mut tick = 0u64;
            let mut port = 0;
            let mut events = Vec::new();
            for (index, event) in source.iter().enumerate() {
                tick += u64::from(event.delta.as_int());
                if let TrackEventKind::Meta(MetaMessage::MidiPort(value)) = event.kind {
                    port = value.as_int();
                }
                let keep = match event.kind {
                    TrackEventKind::Midi {
                        message: MidiMessage::NoteOn { .. } | MidiMessage::NoteOff { .. },
                        ..
                    } => selected.contains(&(track, index)),
                    TrackEventKind::Midi { channel, .. } => {
                        channels.contains(&(port, channel.as_int()))
                    }
                    TrackEventKind::Meta(MetaMessage::EndOfTrack) => false,
                    _ => true,
                };
                if keep {
                    events.push((tick, index, event.kind));
                }
            }
            for note in notes.iter().filter(|n| n.track == track && n.off.is_none()) {
                events.push((
                    note.end_tick,
                    usize::MAX - 1,
                    TrackEventKind::Meta(MetaMessage::MidiPort(note.port.into())),
                ));
                events.push((
                    note.end_tick,
                    usize::MAX,
                    TrackEventKind::Midi {
                        channel: note.channel.into(),
                        message: MidiMessage::NoteOff {
                            key: note.note.pitch.into(),
                            vel: 0.into(),
                        },
                    },
                ));
            }
            events.sort_by_key(|e| (e.0, e.1));
            let mut previous = 0;
            let mut output = Vec::new();
            for (tick, _, kind) in events {
                let delta = tick - previous;
                ensure!(delta <= 0x0fff_ffff, "MIDI event gap is too large");
                output.push(TrackEvent {
                    delta: (delta as u32).into(),
                    kind,
                });
                previous = tick;
            }
            output.push(TrackEvent {
                delta: 0.into(),
                kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
            });
            tracks.push(output);
        }
        let output = Smf {
            header: smf.header,
            tracks,
        };
        let mut bytes = Vec::new();
        output
            .write(&mut bytes)
            .map_err(|e| anyhow::anyhow!("MIDI write failed: {e:?}"))?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use basic_pitch::midi::midly::{Header, num::u15};
    fn event(delta: u32, kind: TrackEventKind<'static>) -> TrackEvent<'static> {
        TrackEvent {
            delta: delta.into(),
            kind,
        }
    }
    #[test]
    fn global_tempo_track_selection_velocity_zero_and_channels_are_preserved() {
        let midi = Smf {
            header: Header::new(Format::Parallel, Timing::Metrical(u15::new(480))),
            tracks: vec![
                vec![
                    event(0, TrackEventKind::Meta(MetaMessage::Tempo(500_000.into()))),
                    event(
                        480,
                        TrackEventKind::Meta(MetaMessage::Tempo(1_000_000.into())),
                    ),
                ],
                vec![
                    event(
                        0,
                        TrackEventKind::Midi {
                            channel: 0.into(),
                            message: MidiMessage::ProgramChange { program: 29.into() },
                        },
                    ),
                    event(
                        0,
                        TrackEventKind::Midi {
                            channel: 0.into(),
                            message: MidiMessage::NoteOn {
                                key: 64.into(),
                                vel: 100.into(),
                            },
                        },
                    ),
                    event(
                        960,
                        TrackEventKind::Midi {
                            channel: 0.into(),
                            message: MidiMessage::NoteOn {
                                key: 64.into(),
                                vel: 0.into(),
                            },
                        },
                    ),
                ],
                vec![
                    event(
                        0,
                        TrackEventKind::Midi {
                            channel: 1.into(),
                            message: MidiMessage::NoteOn {
                                key: 40.into(),
                                vel: 100.into(),
                            },
                        },
                    ),
                    event(
                        480,
                        TrackEventKind::Midi {
                            channel: 1.into(),
                            message: MidiMessage::NoteOff {
                                key: 40.into(),
                                vel: 0.into(),
                            },
                        },
                    ),
                ],
                vec![
                    event(
                        0,
                        TrackEventKind::Midi {
                            channel: 9.into(),
                            message: MidiMessage::NoteOn {
                                key: 36.into(),
                                vel: 90.into(),
                            },
                        },
                    ),
                    event(
                        480,
                        TrackEventKind::Midi {
                            channel: 9.into(),
                            message: MidiMessage::NoteOff {
                                key: 36.into(),
                                vel: 0.into(),
                            },
                        },
                    ),
                ],
            ],
        };
        let mut bytes = Vec::new();
        midi.write(&mut bytes).unwrap();
        let parsed = MidiFile::parse(bytes).unwrap();
        let notes = parsed.select(Selection::Auto).unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].track, 1);
        assert_eq!(notes[0].end_tick, 960);
        assert!((notes[0].note.end_s - 1.5).abs() < 1e-8);
        assert_eq!(parsed.select(Selection::All).unwrap().len(), 2);
        assert!(parsed.select(Selection::Track(3)).is_err());
        let preview = MidiFile::parse(parsed.preview(&notes).unwrap()).unwrap();
        assert_eq!(preview.notes.len(), 1);
        assert_eq!(preview.notes[0].velocity, 100);
        assert_eq!(preview.tempos.len(), 2);
        assert_eq!(preview.notes[0].end_tick, 960);
    }
    #[test]
    fn program_selection_is_independent_across_midi_ports() {
        let on = |key, delta| {
            event(
                delta,
                TrackEventKind::Midi {
                    channel: 0.into(),
                    message: MidiMessage::NoteOn {
                        key,
                        vel: 100.into(),
                    },
                },
            )
        };
        let off = |key, delta| {
            event(
                delta,
                TrackEventKind::Midi {
                    channel: 0.into(),
                    message: MidiMessage::NoteOff { key, vel: 0.into() },
                },
            )
        };
        let source = Smf {
            header: Header::new(Format::Parallel, Timing::Metrical(480.into())),
            tracks: vec![
                vec![
                    on(60.into(), 0),
                    off(60.into(), 240),
                    on(62.into(), 240),
                    off(62.into(), 240),
                ],
                vec![
                    event(0, TrackEventKind::Meta(MetaMessage::MidiPort(1.into()))),
                    event(
                        0,
                        TrackEventKind::Midi {
                            channel: 0.into(),
                            message: MidiMessage::ProgramChange { program: 29.into() },
                        },
                    ),
                    on(64.into(), 0),
                    off(64.into(), 960),
                ],
            ],
        };
        let mut bytes = Vec::new();
        source.write(&mut bytes).unwrap();
        let source = MidiFile::parse(bytes).unwrap();
        let selected = source.select(Selection::Auto).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].port, 1);
        let preview = MidiFile::parse(source.preview(&selected).unwrap()).unwrap();
        assert_eq!(preview.notes.len(), 1);
        assert_eq!(preview.notes[0].port, 1);
    }
    #[test]
    fn invalid_files_empty_tracks_and_unclosed_notes_are_handled_explicitly() {
        assert!(MidiFile::parse(b"not MIDI".to_vec()).is_err());
        let source = Smf {
            header: Header::new(Format::SingleTrack, Timing::Metrical(480.into())),
            tracks: vec![vec![
                event(
                    0,
                    TrackEventKind::Midi {
                        channel: 0.into(),
                        message: MidiMessage::NoteOn {
                            key: 64.into(),
                            vel: 90.into(),
                        },
                    },
                ),
                event(480, TrackEventKind::Meta(MetaMessage::EndOfTrack)),
            ]],
        };
        let mut bytes = Vec::new();
        source.write(&mut bytes).unwrap();
        let source = MidiFile::parse(bytes).unwrap();
        assert_eq!(source.repaired_note_offs, 1);
        assert!(source.select(Selection::Track(99)).is_err());
        let selected = source.select(Selection::Auto).unwrap();
        let preview = MidiFile::parse(source.preview(&selected).unwrap()).unwrap();
        assert_eq!(preview.repaired_note_offs, 0);
        assert_eq!(preview.notes[0].end_tick, 480);
    }
}

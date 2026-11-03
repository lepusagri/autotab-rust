use super::tokens::DetectedNote;
use crate::rhythm::{Grid, PPQ, Rhythm};
use anyhow::{Result, ensure};
use basic_pitch::{
    NoteEvent,
    midi::midly::{
        Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind,
        num::{u4, u7, u15, u24, u28},
    },
};
use std::collections::BTreeMap;

pub struct Transcription {
    pub notes: Vec<DetectedNote>,
}

impl Transcription {
    pub fn guitar_notes(&self) -> Vec<NoteEvent> {
        self.notes
            .iter()
            .filter(|n| (24..=31).contains(&n.program))
            .map(|n| n.note.clone())
            .collect()
    }

    pub fn to_midi(&self, rhythm: &Rhythm) -> Result<Vec<u8>> {
        let tempo = rhythm.tempo_micros()?;
        let signature = (
            rhythm.meter.numerator,
            rhythm.meter.midi_denominator(),
            rhythm.meter.metronome(),
        );
        let mut groups: BTreeMap<u8, Vec<&NoteEvent>> = BTreeMap::new();
        for note in &self.notes {
            groups.entry(note.program).or_default().push(&note.note);
        }
        let mut tracks = vec![vec![
            event(0, TrackEventKind::Meta(MetaMessage::Tempo(u24::new(tempo))))?,
            event(
                0,
                TrackEventKind::Meta(MetaMessage::TimeSignature(
                    signature.0,
                    signature.1,
                    signature.2,
                    8,
                )),
            )?,
            event(0, TrackEventKind::Meta(MetaMessage::EndOfTrack))?,
        ]];
        let channels: Vec<_> = (0..16).filter(|&channel| channel != 9).collect();
        let mut melodic_index = 0;
        let names: Vec<_> = groups
            .keys()
            .map(|program| {
                crate::instruments::GROUPS
                    .iter()
                    .find(|group| {
                        group.program == Some(*program) || (*program == 128 && group.id == 36)
                    })
                    .map(|group| group.name.to_owned())
                    .unwrap_or_else(|| format!("program_{program}"))
            })
            .collect();
        for ((program, notes), name) in groups.into_iter().zip(&names) {
            let (port, channel) = if program == 128 {
                (0, 9)
            } else {
                let allocation = (melodic_index / 15, channels[melodic_index % 15]);
                melodic_index += 1;
                allocation
            };
            let channel = u4::new(channel);
            let mut track = vec![
                event(
                    0,
                    TrackEventKind::Meta(MetaMessage::MidiPort(u7::new(port as u8))),
                )?,
                event(
                    0,
                    TrackEventKind::Meta(MetaMessage::TrackName(name.as_bytes())),
                )?,
            ];
            // Some notation software ignores a conductor tempo without notes.
            track.push(event(
                0,
                TrackEventKind::Meta(MetaMessage::Tempo(u24::new(tempo))),
            )?);
            track.push(event(
                0,
                TrackEventKind::Meta(MetaMessage::TimeSignature(
                    signature.0,
                    signature.1,
                    signature.2,
                    8,
                )),
            )?);
            if program != 128 {
                track.push(event(
                    0,
                    TrackEventKind::Midi {
                        channel,
                        message: MidiMessage::ProgramChange {
                            program: u7::new(if program <= 127 { program } else { 0 }),
                        },
                    },
                )?);
            }
            let mut by_pitch: BTreeMap<u8, Vec<(u64, u64)>> = BTreeMap::new();
            for note in notes {
                ensure!(
                    note.start_s.is_finite()
                        && note.end_s.is_finite()
                        && note.start_s >= 0.0
                        && note.end_s > note.start_s
                        && note.pitch <= 127,
                    "Invalid MuScriptor MIDI note"
                );
                let span = rhythm.note_ticks(note.start_s, note.end_s)?;
                by_pitch.entry(note.pitch).or_default().push(span);
            }
            let mut timed = Vec::new();
            for (pitch, mut spans) in by_pitch {
                spans.sort_unstable();
                if rhythm.grid != Grid::Off {
                    let mut cleaned: Vec<(u64, u64)> = Vec::new();
                    for (start, end) in spans {
                        if let Some(previous) = cleaned.last_mut() {
                            if previous.0 == start {
                                previous.1 = previous.1.max(end);
                                continue;
                            }
                            previous.1 = previous.1.min(start);
                        }
                        cleaned.push((start, end));
                    }
                    spans = cleaned;
                }
                for (start, end) in spans {
                    timed.push((start, 1_u8, pitch));
                    timed.push((end, 0_u8, pitch));
                }
            }
            timed.sort_unstable(); // Note-off precedes note-on at the same tick.
            let mut previous = 0;
            for (tick, on, pitch) in timed {
                let message = if on == 1 {
                    MidiMessage::NoteOn {
                        key: u7::new(pitch),
                        vel: u7::new(100),
                    }
                } else {
                    MidiMessage::NoteOff {
                        key: u7::new(pitch),
                        vel: u7::new(0),
                    }
                };
                track.push(event(
                    tick - previous,
                    TrackEventKind::Midi { channel, message },
                )?);
                previous = tick;
            }
            track.push(event(0, TrackEventKind::Meta(MetaMessage::EndOfTrack))?);
            tracks.push(track);
        }
        let smf = Smf {
            header: Header::new(Format::Parallel, Timing::Metrical(u15::new(PPQ))),
            tracks,
        };
        let mut bytes = Vec::new();
        smf.write(&mut bytes).map_err(anyhow::Error::msg)?;
        Ok(bytes)
    }
}

fn event(delta: u64, kind: TrackEventKind<'_>) -> Result<TrackEvent<'_>> {
    ensure!(
        delta <= 0x0fff_ffff,
        "MIDI event gap exceeds the Standard MIDI File limit"
    );
    Ok(TrackEvent {
        delta: u28::new(delta as u32),
        kind,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(program: u8, pitch: u8) -> DetectedNote {
        DetectedNote {
            program,
            note: NoteEvent {
                start_frame: 0,
                end_frame: 100,
                start_s: 0.0,
                end_s: 1.0,
                pitch,
                amplitude: 100.0 / 127.0,
                pitch_bends: None,
            },
        }
    }
    #[test]
    fn exports_requested_tempo_meter_and_quantized_ticks_on_every_instrument() {
        let mut note = fixture(0, 60);
        note.note.start_s = 0.14;
        note.note.end_s = 0.18;
        let mut guitar = note.clone();
        guitar.program = 24;
        let transcription = Transcription {
            notes: vec![note, guitar],
        };
        let rhythm = Rhythm {
            bpm: 96.0,
            meter: "6/8".parse().unwrap(),
            grid: Grid::Sixteenth,
        };
        let bytes = transcription.to_midi(&rhythm).unwrap();
        let midi = Smf::parse(&bytes).unwrap();
        assert!(midi.tracks[0].iter().any(|event| matches!(event.kind,
            TrackEventKind::Meta(MetaMessage::Tempo(value)) if value.as_int() == 625_000)));
        assert!(midi.tracks[0].iter().any(|event| matches!(
            event.kind,
            TrackEventKind::Meta(MetaMessage::TimeSignature(6, 3, 36, 8))
        )));
        for track in &midi.tracks[1..] {
            let mut tick = 0;
            let mut notes = Vec::new();
            for event in track {
                tick += event.delta.as_int();
                match event.kind {
                    TrackEventKind::Midi {
                        message: MidiMessage::NoteOn { .. },
                        ..
                    } => notes.push((tick, true)),
                    TrackEventKind::Midi {
                        message: MidiMessage::NoteOff { .. },
                        ..
                    } => notes.push((tick, false)),
                    _ => (),
                }
            }
            assert_eq!(notes, [(120, true), (240, false)]);
        }
    }

    #[test]
    fn quantization_merges_same_pitch_collisions_without_duplicate_note_on_events() {
        let mut first = fixture(24, 60);
        first.note.start_s = 0.01;
        first.note.end_s = 0.02;
        let mut second = first.clone();
        second.note.start_s = 0.03;
        second.note.end_s = 0.04;
        let transcription = Transcription {
            notes: vec![first, second],
        };
        let rhythm = Rhythm {
            grid: Grid::Sixteenth,
            ..Rhythm::default()
        };
        let bytes = transcription.to_midi(&rhythm).unwrap();
        let midi = Smf::parse(&bytes).unwrap();
        let ons = midi.tracks[1]
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    TrackEventKind::Midi {
                        message: MidiMessage::NoteOn { .. },
                        ..
                    }
                )
            })
            .count();
        let offs = midi.tracks[1]
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    TrackEventKind::Midi {
                        message: MidiMessage::NoteOff { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!((ons, offs), (1, 1));
        assert_eq!(
            midi.tracks[1]
                .iter()
                .map(|event| event.delta.as_int())
                .sum::<u32>(),
            120
        );
    }

    #[test]
    fn keeps_piano_guitar_bass_and_drums_but_tabs_use_only_guitar() {
        let transcription = Transcription {
            notes: vec![
                fixture(0, 60),
                fixture(24, 64),
                fixture(32, 40),
                fixture(128, 36),
            ],
        };
        assert_eq!(
            transcription
                .guitar_notes()
                .iter()
                .map(|n| n.pitch)
                .collect::<Vec<_>>(),
            [64]
        );
        let bytes = transcription.to_midi(&Rhythm::default()).unwrap();
        let midi = Smf::parse(&bytes).unwrap();
        assert_eq!(midi.tracks.len(), 5);
        let programs: Vec<_> = midi
            .tracks
            .iter()
            .flatten()
            .filter_map(|event| match event.kind {
                TrackEventKind::Midi {
                    message: MidiMessage::ProgramChange { program },
                    ..
                } => Some(program.as_int()),
                _ => None,
            })
            .collect();
        assert_eq!(programs, [0, 24, 32]);
        assert_eq!(midi.tracks.iter().flatten().filter(|event| matches!(event.kind,
            TrackEventKind::Midi { message: MidiMessage::NoteOn { vel, .. }, .. } if vel.as_int() > 0)).count(), 4);
        assert!(midi.tracks.iter().flatten().any(|event| matches!(event.kind,
            TrackEventKind::Midi { channel, message: MidiMessage::NoteOn { key, .. } } if channel.as_int() == 9 && key.as_int() == 36)));
    }

    #[test]
    fn more_than_fifteen_melodic_programs_use_extra_midi_ports_without_losing_notes() {
        let transcription = Transcription {
            notes: (0..17).map(|program| fixture(program, 60)).collect(),
        };
        let bytes = transcription.to_midi(&Rhythm::default()).unwrap();
        let midi = Smf::parse(&bytes).unwrap();
        assert_eq!(midi.tracks.len(), 18);
        assert!(
            midi.tracks
                .iter()
                .flatten()
                .any(|event| matches!(event.kind,
            TrackEventKind::Meta(MetaMessage::MidiPort(port)) if port.as_int() == 1))
        );
    }
}

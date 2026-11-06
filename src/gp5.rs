//! Native Guitar Pro 5 export with explicit fingerings and tied rhythmic notation.
use crate::midi_input::{ImportedNote, MidiFile};
use crate::{fingering::Arrangement, rhythm::Rhythm, tab::Tuning};
use anyhow::{Context, Result, ensure};
use guitarpro::{
    Beat, BeatStatus, Measure, MeasureHeader, Note, NoteType, Song, Track, Voice,
    audio::midi::MidiChannel,
    model::legacy::{
        key_signature::{Duration, TimeSignature},
        mix_table::{MixTableChange, MixTableItem},
        track::TrackSettings,
    },
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

const QUARTER: i64 = 960;
const GRID: i64 = 120;

#[derive(Clone, Debug)]
struct QNote {
    guitar: usize,
    string: u8,
    fret: u8,
    start: i64,
    end: i64,
    velocity: i16,
}

#[derive(Serialize)]
pub struct Report {
    pub attacks: usize,
    pub tracks: usize,
    pub measures: usize,
    pub bpm: i16,
    pub grid: &'static str,
    pub max_boundary_shift_ms: f64,
    pub quantization_extra_guitars: usize,
    pub tempo_changes: usize,
    pub meter_changes: usize,
}

fn length(duration: &Duration) -> i64 {
    let base = QUARTER * 4 / i64::from(duration.value);
    if duration.dotted {
        base + base / 2
    } else {
        base
    }
}

fn durations(mut start: i64, end: i64) -> Result<Vec<(i64, Duration)>> {
    let mut result = Vec::new();
    while start < end {
        let mut chosen: Option<(i64, Duration)> = None;
        for value in [1, 2, 4, 8, 16, 32, 64] {
            for dotted in [true, false] {
                let duration = Duration {
                    value,
                    dotted,
                    ..Default::default()
                };
                let ticks = length(&duration);
                let base = QUARTER * 4 / i64::from(value);
                if ticks <= end - start
                    && ticks % 60 == 0
                    && start % base == 0
                    && chosen.as_ref().is_none_or(|c| ticks > c.0)
                {
                    chosen = Some((ticks, duration));
                }
            }
        }
        let (ticks, duration) = chosen.context("Cannot represent GP5 rhythmic interval")?;
        result.push((start, duration));
        start += ticks;
    }
    Ok(result)
}

pub fn export(
    result: &Arrangement,
    tuning: Tuning,
    rhythm: Rhythm,
    title: &str,
) -> Result<(Vec<u8>, Report)> {
    export_score(result, tuning, rhythm, title, None)
}

pub fn export_midi(
    result: &Arrangement,
    tuning: Tuning,
    midi: &MidiFile,
    selected: &[ImportedNote],
    title: &str,
) -> Result<(Vec<u8>, Report)> {
    let rhythm = Rhythm {
        bpm: 60_000_000.0 / f64::from(midi.tempos[0].micros),
        meter: midi.meters[0].meter,
        ..Default::default()
    };
    export_score(result, tuning, rhythm, title, Some((midi, selected)))
}

struct Bar {
    start: i64,
    end: i64,
    meter: crate::rhythm::Meter,
}
fn signature(meter: crate::rhythm::Meter) -> TimeSignature {
    TimeSignature {
        numerator: meter.numerator as i8,
        denominator: Duration {
            value: u16::from(meter.denominator),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn export_score(
    result: &Arrangement,
    tuning: Tuning,
    rhythm: Rhythm,
    title: &str,
    midi_source: Option<(&MidiFile, &[ImportedNote])>,
) -> Result<(Vec<u8>, Report)> {
    rhythm.validate()?;
    let mut notes = Vec::new();
    let mut shift: f64 = 0.0;
    for note in &result.notes {
        let (start_tick, end_tick, scale) = if let Some((midi, selected)) = midi_source {
            let source = selected
                .get(note.index)
                .context("Missing imported note for GP5")?;
            (
                source.start_tick,
                source.end_tick,
                QUARTER as f64 / f64::from(midi.ppq),
            )
        } else {
            let (start, end) = rhythm.note_ticks(note.start_s, note.end_s)?;
            (start, end, 2.0)
        };
        let start = (start_tick as f64 * scale / GRID as f64).round() as i64 * GRID;
        let end = ((end_tick as f64 * scale / GRID as f64).round() as i64 * GRID).max(start + GRID);
        for (ticks, seconds) in [(start, note.start_s), (end, note.end_s)] {
            let quantized_seconds = midi_source.map_or(
                ticks as f64 / QUARTER as f64 * 60.0 / rhythm.bpm,
                |(midi, _)| {
                    midi.tick_to_seconds(ticks as f64 * f64::from(midi.ppq) / QUARTER as f64)
                },
            );
            shift = shift.max((quantized_seconds - seconds).abs() * 1000.0);
        }
        notes.push(QNote {
            guitar: note.guitar,
            string: note.string,
            fret: note.fret,
            start,
            end,
            velocity: (note.amplitude.clamp(0.0, 1.0) * 127.0).round() as i16,
        });
    }
    notes.sort_by_key(|n| (n.start, n.guitar, n.string));
    let original_tracks = notes.iter().map(|n| n.guitar).max().unwrap_or(1);
    let mut tracks = original_tracks;
    let mut busy: BTreeMap<(usize, u8), i64> = BTreeMap::new();
    for note in &mut notes {
        if busy
            .get(&(note.guitar, note.string))
            .is_some_and(|&end| end > note.start)
        {
            let guitar = (original_tracks + 1..=tracks).find(|&g| {
                busy.get(&(g, note.string))
                    .is_none_or(|&end| end <= note.start)
            });
            note.guitar = guitar.unwrap_or_else(|| {
                tracks += 1;
                tracks
            });
        }
        busy.insert((note.guitar, note.string), note.end);
    }
    let finish = notes.iter().map(|n| n.end).max().unwrap_or(1);
    let mut meters = BTreeMap::from([(0i64, rhythm.meter)]);
    let mut tempos = BTreeMap::from([(0i64, rhythm.bpm.round() as i16)]);
    if let Some((midi, _)) = midi_source {
        for change in &midi.meters {
            let tick = (change.tick as f64 * QUARTER as f64 / f64::from(midi.ppq)).round() as i64;
            if tick < finish {
                meters.insert(tick, change.meter);
            }
        }
        for change in &midi.tempos {
            let tick = (change.tick as f64 * QUARTER as f64 / f64::from(midi.ppq) / GRID as f64)
                .round() as i64
                * GRID;
            let bpm = (60_000_000.0 / f64::from(change.micros)).round() as i16;
            if tick >= finish && tick != 0 {
                continue;
            }
            ensure!(
                bpm > 0 && (tick == 0 || bpm <= 255),
                "GP5 tempo changes must be between 1 and 255 BPM"
            );
            tempos.insert(tick, bpm);
        }
    }
    let mut bars = Vec::new();
    let mut start = 0;
    while start < finish {
        ensure!(
            bars.len() < i16::MAX as usize && tracks <= 60,
            "GP5 export exceeds track/measure limits"
        );
        let meter = *meters.range(..=start).next_back().unwrap().1;
        let end = start + i64::from(meter.numerator) * QUARTER * 4 / i64::from(meter.denominator);
        ensure!(
            meters.range(start + 1..end).next().is_none(),
            "MIDI time-signature changes inside a measure cannot be represented in GP5"
        );
        bars.push(Bar { start, end, meter });
        start = end;
    }
    let mut song = Song {
        name: title.into(),
        tempo: tempos[&0],
        writer: "AutoTab Rust".into(),
        subtitle: "Fixed string/fret assignments".into(),
        notice: vec![
            "Timing rounded to 1/32; integer BPM; additional voices preserve excess notes.".into(),
            "v3 model option: cstr/tab-labeler-onnx, tab-labeler-v3-egset.onnx, CC BY 4.0.".into(),
        ],
        ..Default::default()
    };
    song.channels = (0..64)
        .map(|channel| MidiChannel {
            channel,
            effect_channel: channel,
            instrument: 29,
            ..Default::default()
        })
        .collect();
    song.measure_headers = bars
        .iter()
        .enumerate()
        .map(|(m, bar)| MeasureHeader {
            number: m as u16 + 1,
            start: QUARTER + bar.start,
            time_signature: signature(bar.meter),
            ..Default::default()
        })
        .collect();
    let mut pitches = tuning.pitches();
    pitches.reverse();
    let channels: Vec<_> = (0..64).filter(|channel| channel % 16 != 9).collect();
    for guitar in 1..=tracks {
        let channel = channels[guitar - 1];
        let mut track = Track {
            number: guitar as i32,
            name: if guitar == 1 {
                "Guitar".into()
            } else {
                format!("Extra guitar {guitar}")
            },
            channel_index: channel,
            port: (channel / 16 + 1) as u8,
            strings: pitches
                .iter()
                .enumerate()
                .map(|(s, &p)| (s as i8 + 1, p as i8))
                .collect(),
            settings: TrackSettings {
                tablature: true,
                notation: false,
                show_rhythm: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let source: Vec<_> = notes.iter().filter(|n| n.guitar == guitar).collect();
        for (m, bar) in bars.iter().enumerate() {
            let start = bar.start;
            let end = bar.end;
            let relevant: Vec<_> = source
                .iter()
                .filter(|n| n.start < end && n.end > start)
                .collect();
            let mut boundaries = BTreeSet::from([start, end]);
            boundaries.extend(tempos.range(start..end).map(|(&tick, _)| tick));
            for n in &relevant {
                boundaries.insert(n.start.max(start));
                boundaries.insert(n.end.min(end));
            }
            let boundaries: Vec<_> = boundaries.into_iter().collect();
            let mut voice = Voice {
                measure_index: m as i16,
                ..Default::default()
            };
            for interval in boundaries.windows(2) {
                // Alignment is relative to the measure, including non-4/4 meters.
                for (local, duration) in durations(interval[0] - start, interval[1] - start)? {
                    let position = local + start;
                    let active: Vec<_> = relevant
                        .iter()
                        .filter(|n| n.start <= position && n.end > position)
                        .collect();
                    let mut beat = Beat {
                        start: Some(QUARTER + position),
                        duration,
                        status: if active.is_empty() {
                            BeatStatus::Rest
                        } else {
                            BeatStatus::Normal
                        },
                        ..Default::default()
                    };
                    if guitar == 1
                        && position > 0
                        && let Some(&bpm) = tempos.get(&position)
                    {
                        beat.effect.mix_table_change = Some(MixTableChange {
                            tempo: Some(MixTableItem {
                                value: bpm as u8,
                                all_tracks: true,
                                ..Default::default()
                            }),
                            hide_tempo: false,
                            ..Default::default()
                        });
                    }
                    for n in active {
                        beat.notes.push(Note {
                            string: n.string as i8,
                            value: i16::from(n.fret),
                            velocity: n.velocity,
                            kind: if position == n.start {
                                NoteType::Normal
                            } else {
                                NoteType::Tie
                            },
                            ..Default::default()
                        });
                    }
                    voice.beats.push(beat);
                }
            }
            track.measures.push(Measure {
                number: m + 1,
                start: QUARTER + start,
                track_index: guitar - 1,
                header_index: m,
                time_signature: signature(bar.meter),
                voices: vec![voice],
                ..Default::default()
            });
        }
        song.tracks.push(track);
    }
    let bytes = song
        .write((5, 1, 0), None)
        .context("Cannot encode Guitar Pro file")?;
    verify(&bytes, &notes, &bars)?;
    Ok((
        bytes,
        Report {
            attacks: notes.len(),
            tracks,
            measures: bars.len(),
            bpm: song.tempo,
            grid: "1/32",
            max_boundary_shift_ms: shift,
            quantization_extra_guitars: tracks - original_tracks,
            tempo_changes: tempos.len() - 1,
            meter_changes: meters.len() - 1,
        },
    ))
}

fn verify(bytes: &[u8], expected: &[QNote], bars: &[Bar]) -> Result<()> {
    let mut song = Song::default();
    song.read_gp5(bytes)
        .context("Cannot reload generated GP5")?;
    let mut actual: Vec<(usize, u8, u8, i64, i64)> = Vec::new();
    for (track, guitar) in song.tracks.iter().enumerate() {
        let mut active = BTreeMap::<i8, usize>::new();
        for (measure_index, measure) in guitar.measures.iter().enumerate() {
            // The dependency's parser does not populate absolute measure starts.
            let bar = &bars[measure_index];
            let header = &song.measure_headers[measure_index];
            ensure!(
                header.time_signature.numerator == bar.meter.numerator as i8
                    && header.time_signature.denominator.value == u16::from(bar.meter.denominator),
                "GP5 meter changed"
            );
            let mut ticks = bar.start;
            let first = ticks;
            for beat in &measure.voices[0].beats {
                let end = ticks + length(&beat.duration);
                for n in &beat.notes {
                    match n.kind {
                        NoteType::Normal => {
                            active.insert(n.string, actual.len());
                            actual.push((track + 1, n.string as u8, n.value as u8, ticks, end));
                        }
                        NoteType::Tie => {
                            let index = active
                                .get(&n.string)
                                .context("GP5 tie has no preceding note")?;
                            // GP5 ties carry no independent fret. The dependency's reader
                            // resolves them from older measures, so resolve from our own
                            // immediately preceding note on this string instead.
                            ensure!(
                                actual[*index].4 == ticks,
                                "GP5 tie has a gap or overlap: previous {:?}, next {ticks}",
                                actual[*index]
                            );
                            actual[*index].4 = end;
                        }
                        _ => {}
                    }
                }
                ticks = end;
            }
            ensure!(
                ticks - first == bar.end - bar.start,
                "GP5 measure length changed"
            );
        }
    }
    actual.sort_unstable();
    let mut wanted: Vec<_> = expected
        .iter()
        .map(|n| (n.guitar, n.string, n.fret, n.start, n.end))
        .collect();
    wanted.sort_unstable();
    ensure!(
        actual == wanted,
        "GP5 round-trip changed note attacks, fingerings or quantized durations"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingering::Assigned;
    #[test]
    fn native_gp5_roundtrip_preserves_polyphony_ties_and_extra_guitar() {
        let notes = vec![
            Assigned {
                index: 0,
                pitch: 67,
                start_s: 0.5,
                end_s: 3.0,
                amplitude: 0.8,
                string: 1,
                fret: 3,
                guitar: 1,
            },
            Assigned {
                index: 1,
                pitch: 62,
                start_s: 0.5,
                end_s: 1.0,
                amplitude: 0.8,
                string: 2,
                fret: 3,
                guitar: 1,
            },
            Assigned {
                index: 2,
                pitch: 58,
                start_s: 1.5,
                end_s: 1.75,
                amplitude: 0.8,
                string: 3,
                fret: 3,
                guitar: 2,
            },
        ];
        let result = Arrangement {
            notes,
            extra_guitars: 1,
            ..Default::default()
        };
        let (_, report) = export(&result, Tuning::Standard, Rhythm::default(), "Test").unwrap();
        assert_eq!((report.attacks, report.tracks, report.measures), (3, 2, 2));
    }
    #[test]
    fn every_grid_interval_has_valid_rhythmic_values() {
        for start in (0..3840).step_by(GRID as usize) {
            for end in (start + GRID..=3840).step_by(GRID as usize) {
                let beats = durations(start, end).unwrap();
                assert_eq!(
                    beats.iter().map(|(_, d)| length(d)).sum::<i64>(),
                    end - start
                );
            }
        }
    }
    #[test]
    fn short_note_collisions_are_preserved_on_another_gp5_track() {
        let notes = [(0.033, 0.05), (0.084, 0.10)]
            .into_iter()
            .enumerate()
            .map(|(index, (start_s, end_s))| Assigned {
                index,
                pitch: 67,
                start_s,
                end_s,
                amplitude: 0.8,
                string: 1,
                fret: 3,
                guitar: 1,
            })
            .collect();
        let result = Arrangement {
            notes,
            ..Default::default()
        };
        let (_, report) =
            export(&result, Tuning::Standard, Rhythm::default(), "Short notes").unwrap();
        assert_eq!(report.attacks, 2);
        assert_eq!(report.quantization_extra_guitars, 1);
    }
    #[test]
    fn empty_and_non_four_four_scores_have_valid_measures() {
        for meter in ["6/8", "3/4", "1/64"] {
            let rhythm = Rhythm {
                meter: meter.parse().unwrap(),
                ..Default::default()
            };
            let (_, report) =
                export(&Arrangement::default(), Tuning::Standard, rhythm, "Rest").unwrap();
            assert_eq!((report.attacks, report.measures), (0, 1));
        }
    }
    #[test]
    fn midi_tempo_and_meter_changes_keep_original_beat_positions() {
        use basic_pitch::midi::midly::{
            Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind,
        };
        let event = |delta: u32, kind| TrackEvent {
            delta: delta.into(),
            kind,
        };
        let midi = Smf {
            header: Header::new(Format::Parallel, Timing::Metrical(480.into())),
            tracks: vec![
                vec![
                    event(0, TrackEventKind::Meta(MetaMessage::Tempo(500_000.into()))),
                    event(
                        480,
                        TrackEventKind::Meta(MetaMessage::Tempo(666_667.into())),
                    ),
                    event(
                        1440,
                        TrackEventKind::Meta(MetaMessage::TimeSignature(3, 2, 24, 8)),
                    ),
                ],
                vec![
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
                        3360,
                        TrackEventKind::Midi {
                            channel: 0.into(),
                            message: MidiMessage::NoteOff {
                                key: 64.into(),
                                vel: 0.into(),
                            },
                        },
                    ),
                ],
            ],
        };
        let mut bytes = Vec::new();
        midi.write(&mut bytes).unwrap();
        let mut source = MidiFile::parse(bytes).unwrap();
        let selected = source.select(crate::midi_input::Selection::All).unwrap();
        let n = &selected[0];
        let result = Arrangement {
            notes: vec![Assigned {
                index: 0,
                pitch: 64,
                start_s: n.note.start_s,
                end_s: n.note.end_s,
                amplitude: 0.8,
                string: 1,
                fret: 0,
                guitar: 1,
            }],
            ..Default::default()
        };
        let (bytes, report) = export_midi(
            &result,
            Tuning::Standard,
            &source,
            &selected,
            "Tempo changes",
        )
        .unwrap();
        assert_eq!(
            (report.measures, report.tempo_changes, report.meter_changes),
            (2, 1, 1)
        );
        let mut parsed = Song::default();
        parsed.read_gp5(&bytes).unwrap();
        assert_eq!(parsed.measure_headers[1].time_signature.numerator, 3);
        let changes: Vec<_> = parsed.tracks[0]
            .measures
            .iter()
            .flat_map(|m| &m.voices[0].beats)
            .filter_map(|b| {
                b.effect
                    .mix_table_change
                    .as_ref()?
                    .tempo
                    .as_ref()
                    .map(|t| t.value)
            })
            .collect();
        assert_eq!(changes, [90]);
        source.tempos.push(crate::midi_input::Tempo {
            tick: 100_000,
            micros: 200_000,
        });
        source.meters.push(crate::midi_input::MeterChange {
            tick: 100_001,
            meter: "7/8".parse().unwrap(),
        });
        let (with_trailing_metadata, _) = export_midi(
            &result,
            Tuning::Standard,
            &source,
            &selected,
            "Tempo changes",
        )
        .unwrap();
        assert_eq!(bytes, with_trailing_metadata);
    }
}

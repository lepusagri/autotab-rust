//! MT3 event decoding, including absolute shifts and sustained notes across chunks.
//! Adapted from MuScriptor (MIT); see docs/licenses/muscriptor-MIT.txt.
use basic_pitch::NoteEvent;
use std::collections::{BTreeMap, BTreeSet};

pub const PITCH: usize = 1004;
pub const VELOCITY: usize = 1132;
pub const TIE: usize = 1134;
pub const PROGRAM: usize = 1135;

#[derive(Clone, Debug)]
pub struct DetectedNote {
    pub program: u8,
    pub note: NoteEvent,
}

#[derive(Default)]
pub struct Decoder {
    open: BTreeMap<(u8, u8), f64>,
    pub notes: Vec<DetectedNote>,
    boundary: f64,
    end: f64,
    tick: usize,
    program: Option<u8>,
    velocity: Option<bool>,
    prologue: bool,
    skip: bool,
    ties: BTreeSet<(u8, u8)>,
}

impl Decoder {
    pub fn chunk(&mut self, start: f64, end: f64) {
        if self.prologue {
            self.close_all(self.boundary);
        }
        self.boundary = start;
        self.end = end;
        self.tick = 0;
        self.program = None;
        self.velocity = None;
        self.prologue = true;
        self.skip = false;
        self.ties.clear();
    }

    pub fn prompt(&self) -> Vec<usize> {
        let mut prompt = Vec::new();
        let mut previous = None;
        for &(program, pitch) in self.open.keys() {
            if previous != Some(program) {
                prompt.push(PROGRAM + program as usize);
                previous = Some(program);
            }
            prompt.push(PITCH + pitch as usize);
        }
        prompt.push(TIE);
        prompt
    }

    pub fn feed(&mut self, token: usize) {
        if self.prologue {
            match token {
                TIE => {
                    self.prologue = false;
                    let ended: Vec<_> = self
                        .open
                        .keys()
                        .filter(|key| !self.ties.contains(key))
                        .copied()
                        .collect();
                    for key in ended {
                        self.close(key, self.boundary);
                    }
                    self.velocity = None;
                }
                PROGRAM..=1264 => self.program = Some((token - PROGRAM) as u8),
                PITCH..=1131 => {
                    if let Some(program) = self.program {
                        self.ties.insert((program, (token - PITCH) as u8));
                    }
                }
                3..=1003 => {
                    self.prologue = false;
                    self.skip = true;
                    self.close_all(self.boundary);
                }
                _ => (),
            }
            return;
        }
        if self.skip {
            return;
        }
        match token {
            4..=1003 => self.tick = token - 3,
            PROGRAM..=1264 => self.program = Some((token - PROGRAM) as u8),
            VELOCITY..=1133 => self.velocity = Some(token == 1133),
            1265..=1392 => {
                let start = self.boundary + self.tick as f64 / 100.0;
                if start < self.end {
                    self.push(
                        128,
                        (token - 1265) as u8,
                        start,
                        (start + 0.01).min(self.end),
                    );
                }
            }
            PITCH..=1131 => {
                let time = self.boundary + self.tick as f64 / 100.0;
                if time >= self.end {
                    return;
                }
                if let (Some(program), Some(on)) = (self.program, self.velocity) {
                    let key = (program, (token - PITCH) as u8);
                    self.close(key, time);
                    if on {
                        self.open.insert(key, time);
                    }
                }
            }
            _ => (),
        }
    }

    fn close(&mut self, key: (u8, u8), time: f64) {
        if let Some(start) = self.open.remove(&key)
            && start < self.end
        {
            let end = time.max(start + 0.01).min(self.end);
            if end <= start {
                return;
            }
            self.push(key.0, key.1, start, end);
        }
    }

    fn push(&mut self, program: u8, pitch: u8, start: f64, end: f64) {
        self.notes.push(DetectedNote {
            program,
            note: NoteEvent {
                pitch,
                start_s: start,
                end_s: end,
                start_frame: (start * 100.0).round() as usize,
                end_frame: (end * 100.0).round() as usize,
                amplitude: 100.0 / 127.0,
                pitch_bends: None,
            },
        });
    }

    fn close_all(&mut self, time: f64) {
        let keys: Vec<_> = self.open.keys().copied().collect();
        for key in keys {
            self.close(key, time);
        }
    }

    pub fn finish(mut self) -> Vec<DetectedNote> {
        let pending: Vec<_> = self
            .open
            .iter()
            .map(|(&key, &start)| (key, start))
            .collect();
        for (key, start) in pending {
            self.close(
                key,
                if self.prologue {
                    self.boundary
                } else {
                    start + 0.01
                },
            );
        }
        self.notes.sort_by(|a, b| {
            a.note
                .start_s
                .total_cmp(&b.note.start_s)
                .then(a.program.cmp(&b.program))
                .then(a.note.pitch.cmp(&b.note.pitch))
        });
        self.notes
    }
}

pub fn allowed(token: usize, instruments: &crate::instruments::Instruments) -> bool {
    match token {
        0..PROGRAM => true,
        PROGRAM..=1264 => instruments.allows_program((token - PROGRAM) as u8),
        1265..=1392 => instruments.allows_drums(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decodes_multiple_programs_and_drum_hits_without_discarding_them() {
        let mut decoder = Decoder::default();
        decoder.chunk(0.0, 2.0);
        for token in [
            TIE,
            PROGRAM,
            VELOCITY + 1,
            PITCH + 60,
            PROGRAM + 24,
            PITCH + 64,
            PROGRAM + 32,
            PITCH + 40,
            1265 + 36,
            3 + 100,
            VELOCITY,
            PROGRAM,
            PITCH + 60,
            PROGRAM + 24,
            PITCH + 64,
            PROGRAM + 32,
            PITCH + 40,
        ] {
            decoder.feed(token);
        }
        let notes = decoder.finish();
        assert_eq!(
            notes.iter().map(|note| note.program).collect::<Vec<_>>(),
            [0, 24, 32, 128]
        );
        assert_eq!(notes[3].note.pitch, 36);
        assert_eq!(notes[3].note.end_s, 0.01);
        assert_eq!(notes[0].note.end_s, 1.0);
        let selection: crate::instruments::Instruments = "0 4 7".parse().unwrap();
        assert!(
            allowed(PROGRAM, &selection)
                && allowed(PROGRAM + 24, &selection)
                && allowed(PROGRAM + 32, &selection)
        );
        assert!(!allowed(PROGRAM + 26, &selection) && !allowed(1265 + 36, &selection));
    }
    #[test]
    fn absolute_shift_and_ties_preserve_note_across_chunks() {
        let mut decoder = Decoder::default();
        decoder.chunk(0.0, 5.0);
        for token in [TIE, PROGRAM + 26, VELOCITY + 1, 3 + 400, PITCH + 64] {
            decoder.feed(token);
        }
        decoder.chunk(5.0, 10.0);
        assert_eq!(decoder.prompt(), [PROGRAM + 26, PITCH + 64, TIE]);
        for token in decoder.prompt() {
            decoder.feed(token);
        }
        for token in [3 + 100, PROGRAM + 26, VELOCITY, PITCH + 64] {
            decoder.feed(token);
        }
        let notes = decoder.finish();
        assert_eq!(notes.len(), 1);
        assert_eq!((notes[0].note.start_s, notes[0].note.end_s), (4.0, 6.0));
        assert!(!allowed(
            PROGRAM,
            &crate::instruments::Instruments::default()
        ));
        assert!(!allowed(1265, &crate::instruments::Instruments::default()));
    }
}

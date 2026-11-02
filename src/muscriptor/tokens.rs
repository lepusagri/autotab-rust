//! MT3 event decoding, including absolute shifts and sustained notes across chunks.
//! Adapted from MuScriptor (MIT); see docs/licenses/muscriptor-MIT.txt.
use basic_pitch::NoteEvent;
use std::collections::{BTreeMap, BTreeSet};

pub const PITCH: usize = 1004;
pub const VELOCITY: usize = 1132;
pub const TIE: usize = 1134;
pub const PROGRAM: usize = 1135;

#[derive(Default)]
pub struct Decoder {
    open: BTreeMap<(u8, u8), f64>,
    pub notes: Vec<NoteEvent>,
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
            && (24..=31).contains(&key.0)
            && start < self.end
        {
            let end = time.max(start + 0.01).min(self.end);
            if end <= start {
                return;
            }
            self.notes.push(NoteEvent {
                pitch: key.1,
                start_s: start,
                end_s: end,
                start_frame: (start * 100.0).round() as usize,
                end_frame: (end * 100.0).round() as usize,
                amplitude: 100.0 / 127.0,
                pitch_bends: None,
            });
        }
    }

    fn close_all(&mut self, time: f64) {
        let keys: Vec<_> = self.open.keys().copied().collect();
        for key in keys {
            self.close(key, time);
        }
    }

    pub fn finish(mut self) -> Vec<NoteEvent> {
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
        self.notes
            .sort_by(|a, b| a.start_s.total_cmp(&b.start_s).then(a.pitch.cmp(&b.pitch)));
        self.notes
    }
}

pub fn allowed(token: usize) -> bool {
    token < PROGRAM || matches!(token - PROGRAM, 24 | 26 | 29)
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!((notes[0].start_s, notes[0].end_s), (4.0, 6.0));
        assert!(!allowed(PROGRAM));
        assert!(!allowed(1265));
    }
}

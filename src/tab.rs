use basic_pitch::NoteEvent;
use clap::ValueEnum;
use std::fmt::Write;

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum Tuning {
    Standard,
    DropD,
    DStandard,
    HalfStepDown,
    EbStandard,
}

impl Tuning {
    fn pitches(self) -> [u8; 6] {
        match self {
            Self::Standard => [40, 45, 50, 55, 59, 64],
            Self::DropD => [38, 45, 50, 55, 59, 64],
            Self::DStandard => [38, 43, 48, 53, 57, 62],
            Self::HalfStepDown | Self::EbStandard => [39, 44, 49, 54, 58, 63],
        }
    }
}

pub fn render(notes: &[NoteEvent], tuning: Tuning) -> String {
    let mut notes: Vec<_> = notes.iter().collect();
    notes.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
    let mut chords: Vec<Vec<&NoteEvent>> = Vec::new();
    for note in notes {
        if let Some(chord) = chords
            .last_mut()
            .filter(|chord| note.start_s - chord[0].start_s < 0.05)
        {
            chord.push(note);
        } else {
            chords.push(vec![note]);
        }
    }
    let strings = tuning.pitches();
    let columns: Vec<_> = chords
        .into_iter()
        .map(|mut chord| {
            chord.sort_by_key(|note| std::cmp::Reverse(note.pitch));
            let mut used = [false; 6];
            let mut column: [String; 6] = std::array::from_fn(|_| "-".into());
            for note in chord {
                for (index, open) in strings.iter().enumerate() {
                    if !used[index]
                        && let Some(fret) = note.pitch.checked_sub(*open).filter(|fret| *fret <= 22)
                    {
                        column[5 - index] = fret.to_string();
                        used[index] = true;
                        break;
                    }
                }
            }
            column
        })
        .collect();
    let mut output = String::new();
    for chunk in columns.chunks(10) {
        for (index, name) in ['e', 'B', 'G', 'D', 'A', 'E'].iter().enumerate() {
            write!(output, "{name}|").unwrap();
            for column in chunk {
                write!(output, "-{:-<9}", column[index]).unwrap();
            }
            output.push_str("|\n");
        }
        output.push('\n');
    }
    if output.is_empty() {
        output.push_str("No playable guitar notes detected.\n");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    fn note(pitch: u8, start_s: f64) -> NoteEvent {
        NoteEvent {
            pitch,
            start_s,
            end_s: start_s + 0.3,
            start_frame: 0,
            end_frame: 26,
            amplitude: 0.8,
            pitch_bends: None,
        }
    }

    #[test]
    fn matches_python_layout_and_groups_chord() {
        let result = render(
            &[note(52, 0.0), note(45, 0.02), note(52, 0.2)],
            Tuning::Standard,
        );
        let lines: Vec<_> = result.lines().collect();
        assert_eq!(lines[4], "A|-0------------------|");
        assert_eq!(lines[5], "E|-12--------12-------|");
        assert!(lines[..6].iter().all(|line| line.len() == 23));
    }

    #[test]
    fn drop_d_reaches_low_d_and_standard_does_not() {
        assert!(render(&[note(38, 0.0)], Tuning::DropD).contains("E|-0--------|"));
        assert!(!render(&[note(38, 0.0)], Tuning::Standard).contains('0'));
        assert_eq!(
            render(&[note(50, 0.0)], Tuning::HalfStepDown),
            render(&[note(50, 0.0)], Tuning::EbStandard)
        );
    }
}

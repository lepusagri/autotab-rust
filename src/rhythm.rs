//! Manual MIDI tempo/meter and quantization. The model's inference is unchanged.
use anyhow::{Result, ensure};
use clap::ValueEnum;
use std::{fmt, str::FromStr};

pub const PPQ: u16 = 480;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Meter {
    pub numerator: u8,
    pub denominator: u8,
}

impl Default for Meter {
    fn default() -> Self {
        Self {
            numerator: 4,
            denominator: 4,
        }
    }
}

impl Meter {
    fn valid(self) -> bool {
        (1..=32).contains(&self.numerator)
            && matches!(self.denominator, 1 | 2 | 4 | 8 | 16 | 32 | 64)
    }
    pub fn midi_denominator(self) -> u8 {
        self.denominator.trailing_zeros() as u8
    }
    pub fn metronome(self) -> u8 {
        if self.denominator == 8 && self.numerator >= 6 && self.numerator.is_multiple_of(3) {
            36
        } else {
            24
        }
    }
}

impl FromStr for Meter {
    type Err = String;
    fn from_str(text: &str) -> std::result::Result<Self, String> {
        let parse = || {
            let (numerator, denominator) = text.split_once('/')?;
            let meter = Self {
                numerator: numerator.parse().ok()?,
                denominator: denominator.parse().ok()?,
            };
            meter.valid().then_some(meter)
        };
        parse().ok_or_else(|| "Invalid time signature. Use N/D (e.g. 4/4, 3/4, 6/8); N=1..32 and D=1,2,4,8,16,32,64.".into())
    }
}

impl fmt::Display for Meter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.numerator, self.denominator)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum Grid {
    #[default]
    Off,
    #[value(alias = "8", alias = "1/8")]
    Eighth,
    #[value(alias = "16", alias = "1/16")]
    Sixteenth,
}

impl Grid {
    pub fn as_cli(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Eighth => "eighth",
            Self::Sixteenth => "sixteenth",
        }
    }
    fn ticks(self) -> Option<u64> {
        match self {
            Self::Off => None,
            Self::Eighth => Some(u64::from(PPQ) / 2),
            Self::Sixteenth => Some(u64::from(PPQ) / 4),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rhythm {
    pub bpm: f64,
    pub meter: Meter,
    pub grid: Grid,
}

impl Default for Rhythm {
    fn default() -> Self {
        Self {
            bpm: 120.0,
            meter: Meter::default(),
            grid: Grid::Off,
        }
    }
}

impl Rhythm {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.bpm.is_finite() && (4.0..=1000.0).contains(&self.bpm),
            "Das Tempo muss zwischen 4 und 1000 BPM liegen."
        );
        ensure!(
            self.meter.valid(),
            "Ungültige Taktart: Zähler 1–32, Nenner 1, 2, 4, 8, 16, 32 oder 64."
        );
        Ok(())
    }

    pub fn tempo_micros(self) -> Result<u32> {
        self.validate()?;
        Ok((60_000_000.0 / self.bpm).round() as u32)
    }

    pub fn note_ticks(self, start_s: f64, end_s: f64) -> Result<(u64, u64)> {
        let tempo = self.tempo_micros()?;
        ensure!(
            start_s.is_finite() && end_s.is_finite() && start_s >= 0.0 && end_s > start_s,
            "Invalid MIDI note times"
        );
        let ticks_per_second = f64::from(PPQ) * 1_000_000.0 / f64::from(tempo);
        ensure!(
            end_s * ticks_per_second < u32::MAX as f64,
            "MIDI note time is too large"
        );
        let start = start_s * ticks_per_second;
        let end = end_s * ticks_per_second;
        Ok(if let Some(step) = self.grid.ticks() {
            let start = (start / step as f64).round() as u64 * step;
            let end = (end / step as f64).round() as u64 * step;
            (start, end.max(start + step))
        } else {
            let start = start.round() as u64;
            (start, (end.round() as u64).max(start + 1))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn changing_tempo_without_quantization_preserves_playback_seconds() {
        for bpm in [60.0, 90.0, 120.0, 137.5, 180.0] {
            let rhythm = Rhythm {
                bpm,
                ..Rhythm::default()
            };
            let (start, end) = rhythm.note_ticks(0.31, 1.27).unwrap();
            let seconds_per_tick =
                f64::from(rhythm.tempo_micros().unwrap()) / 1_000_000.0 / f64::from(PPQ);
            assert!((start as f64 * seconds_per_tick - 0.31).abs() < 0.0011);
            assert!((end as f64 * seconds_per_tick - 1.27).abs() < 0.0011);
        }
    }

    #[test]
    fn eighths_and_sixteenths_snap_to_different_grids_and_keep_short_notes_positive() {
        let eighth = Rhythm {
            grid: Grid::Eighth,
            ..Rhythm::default()
        };
        let sixteenth = Rhythm {
            grid: Grid::Sixteenth,
            ..Rhythm::default()
        };
        assert_eq!(eighth.note_ticks(0.14, 0.18).unwrap(), (240, 480));
        assert_eq!(sixteenth.note_ticks(0.14, 0.18).unwrap(), (120, 240));
        assert_eq!("6/8".parse::<Meter>().unwrap().midi_denominator(), 3);
        assert!("4/3".parse::<Meter>().is_err());
        assert!("0/4".parse::<Meter>().is_err());
        assert!(
            Rhythm {
                bpm: f64::NAN,
                ..Rhythm::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Rhythm {
                bpm: 0.0,
                ..Rhythm::default()
            }
            .validate()
            .is_err()
        );
    }
}

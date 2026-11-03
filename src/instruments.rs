//! MuScriptor's published MT3_FULL_PLUS names, group IDs and representative GM programs.
//! Group IDs are model conditioning classes, not General MIDI program numbers.
use std::{fmt, str::FromStr};

pub struct Group {
    pub id: u8,
    pub name: &'static str,
    #[allow(dead_code)] // Used by the desktop UI; the CLI accepts stable names/IDs.
    pub label: &'static str,
    pub program: Option<u8>,
}

pub const GROUPS: &[Group] = &[
    Group {
        id: 0,
        name: "acoustic_piano",
        label: "Akustisches Klavier",
        program: Some(0),
    },
    Group {
        id: 1,
        name: "electric_piano",
        label: "E-Piano",
        program: Some(2),
    },
    Group {
        id: 2,
        name: "chromatic_percussion",
        label: "Chromatische Perkussion",
        program: Some(8),
    },
    Group {
        id: 3,
        name: "organ",
        label: "Orgel",
        program: Some(16),
    },
    Group {
        id: 4,
        name: "acoustic_guitar",
        label: "Akustische Gitarre",
        program: Some(24),
    },
    Group {
        id: 5,
        name: "clean_electric_guitar",
        label: "E-Gitarre clean",
        program: Some(26),
    },
    Group {
        id: 6,
        name: "distorted_electric_guitar",
        label: "E-Gitarre verzerrt",
        program: Some(29),
    },
    Group {
        id: 7,
        name: "acoustic_bass",
        label: "Akustischer Bass",
        program: Some(32),
    },
    Group {
        id: 8,
        name: "electric_bass",
        label: "E-Bass",
        program: Some(33),
    },
    Group {
        id: 9,
        name: "violin",
        label: "Violine",
        program: Some(40),
    },
    Group {
        id: 10,
        name: "viola",
        label: "Viola",
        program: Some(41),
    },
    Group {
        id: 11,
        name: "cello",
        label: "Cello",
        program: Some(42),
    },
    Group {
        id: 12,
        name: "contrabass",
        label: "Kontrabass",
        program: Some(43),
    },
    Group {
        id: 13,
        name: "orchestral_harp",
        label: "Harfe",
        program: Some(46),
    },
    Group {
        id: 14,
        name: "timpani",
        label: "Pauken",
        program: Some(47),
    },
    Group {
        id: 15,
        name: "string_ensemble",
        label: "Streicherensemble",
        program: Some(48),
    },
    Group {
        id: 16,
        name: "synth_strings",
        label: "Synth-Streicher",
        program: Some(50),
    },
    Group {
        id: 17,
        name: "voice",
        label: "Stimme / Chor",
        program: Some(52),
    },
    Group {
        id: 18,
        name: "orchestra_hit",
        label: "Orchestra Hit",
        program: Some(55),
    },
    Group {
        id: 19,
        name: "trumpet",
        label: "Trompete",
        program: Some(56),
    },
    Group {
        id: 20,
        name: "trombone",
        label: "Posaune",
        program: Some(57),
    },
    Group {
        id: 21,
        name: "tuba",
        label: "Tuba",
        program: Some(58),
    },
    Group {
        id: 22,
        name: "french_horn",
        label: "Horn",
        program: Some(60),
    },
    Group {
        id: 23,
        name: "brass_section",
        label: "Blechbläserensemble",
        program: Some(61),
    },
    Group {
        id: 24,
        name: "soprano_and_alto_sax",
        label: "Sopran- / Altsaxofon",
        program: Some(64),
    },
    Group {
        id: 25,
        name: "tenor_sax",
        label: "Tenorsaxofon",
        program: Some(66),
    },
    Group {
        id: 26,
        name: "baritone_sax",
        label: "Baritonsaxofon",
        program: Some(67),
    },
    Group {
        id: 27,
        name: "oboe",
        label: "Oboe",
        program: Some(68),
    },
    Group {
        id: 28,
        name: "english_horn",
        label: "Englischhorn",
        program: Some(69),
    },
    Group {
        id: 29,
        name: "bassoon",
        label: "Fagott",
        program: Some(70),
    },
    Group {
        id: 30,
        name: "clarinet",
        label: "Klarinette",
        program: Some(71),
    },
    Group {
        id: 31,
        name: "flutes",
        label: "Flöten",
        program: Some(72),
    },
    Group {
        id: 32,
        name: "synth_lead",
        label: "Synth Lead",
        program: Some(80),
    },
    Group {
        id: 33,
        name: "synth_pad",
        label: "Synth Pad",
        program: Some(88),
    },
    Group {
        id: 36,
        name: "drums",
        label: "Schlagzeug",
        program: None,
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruments {
    ids: Vec<u8>,
}

impl Default for Instruments {
    fn default() -> Self {
        Self { ids: vec![4, 5, 6] }
    }
}

impl Instruments {
    pub fn from_ids(ids: impl IntoIterator<Item = u8>) -> Result<Self, String> {
        let mut selected = Vec::new();
        for id in ids {
            if !GROUPS.iter().any(|group| group.id == id) {
                return Err(format!("Unknown MuScriptor group ID: {id}"));
            }
            if !selected.contains(&id) {
                selected.push(id);
            }
        }
        if selected.is_empty() {
            return Err("Select at least one instrument, or use 'auto'.".into());
        }
        Ok(Self { ids: selected })
    }

    pub fn automatic() -> Self {
        Self { ids: Vec::new() }
    }

    pub fn is_automatic(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn has_other_instruments(&self) -> bool {
        self.is_automatic() || self.ids.iter().any(|id| !(4..=6).contains(id))
    }

    pub fn embedding_rows(&self) -> Vec<usize> {
        // Tokenization adds one and ClassConditioner.forward adds one again.
        if self.is_automatic() {
            vec![1]
        } else {
            self.ids.iter().map(|&id| id as usize + 2).collect()
        }
    }

    pub fn allows_program(&self, program: u8) -> bool {
        self.is_automatic()
            || GROUPS
                .iter()
                .any(|group| self.ids.contains(&group.id) && group.program == Some(program))
    }

    pub fn allows_drums(&self) -> bool {
        self.is_automatic() || self.ids.contains(&36)
    }
}

impl FromStr for Instruments {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.trim().eq_ignore_ascii_case("auto") {
            return Ok(Self::automatic());
        }
        let mut ids = Vec::new();
        for word in text
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
        {
            let group = GROUPS.iter().find(|group| word.eq_ignore_ascii_case(group.name) || word == group.id.to_string())
                .ok_or_else(|| format!("Unknown MuScriptor instrument '{word}'. Use published MT3_FULL_PLUS names, group IDs or 'auto'."))?;
            if !ids.contains(&group.id) {
                ids.push(group.id);
            }
        }
        if ids.is_empty() {
            return Err("Select at least one instrument, or use 'auto'.".into());
        }
        Self::from_ids(ids)
    }
}

impl fmt::Display for Instruments {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_automatic() {
            return formatter.write_str("auto");
        }
        formatter.write_str(
            &self
                .ids
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn published_piano_guitar_bass_example_maps_classes_and_programs() {
        let named: Instruments = "acoustic_piano,acoustic_guitar,acoustic_bass"
            .parse()
            .unwrap();
        assert_eq!(named, "0 4 7".parse().unwrap());
        assert_eq!(named.embedding_rows(), [2, 6, 9]);
        assert!(named.allows_program(0) && named.allows_program(24) && named.allows_program(32));
        assert!(!named.allows_program(4)); // A group ID must not become a GM program.
        assert!(!named.allows_drums());
        assert_eq!(
            "4,4,acoustic_guitar"
                .parse::<Instruments>()
                .unwrap()
                .to_string(),
            "4"
        );
        assert!("acoustic_guitar,typo".parse::<Instruments>().is_err());
        assert!(" ".parse::<Instruments>().is_err());
    }

    #[test]
    fn automatic_uses_null_condition_and_allows_drums_and_other_programs() {
        let automatic: Instruments = "auto".parse().unwrap();
        assert_eq!(automatic.embedding_rows(), [1]);
        assert!(automatic.allows_program(101) && automatic.allows_drums());
        assert!("drums".parse::<Instruments>().unwrap().allows_drums());
    }
}

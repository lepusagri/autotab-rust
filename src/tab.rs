use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum Tuning {
    Standard,
    DropD,
    DStandard,
    HalfStepDown,
    EbStandard,
}

impl Tuning {
    pub fn pitches(self) -> [u8; 6] {
        match self {
            Self::Standard => [40, 45, 50, 55, 59, 64],
            Self::DropD => [38, 45, 50, 55, 59, 64],
            Self::DStandard => [38, 43, 48, 53, 57, 62],
            Self::HalfStepDown | Self::EbStandard => [39, 44, 49, 54, 58, 63],
        }
    }
}

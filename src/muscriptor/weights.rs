//! Safetensors metadata and selective reads; no framework or Python dependency.
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct Entry {
    pub shape: Vec<usize>,
    pub offset: u64,
    pub bytes: u64,
}

pub struct Weights {
    pub path: PathBuf,
    pub entries: BTreeMap<String, Entry>,
    pub header: Vec<u8>,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file =
            File::open(path).with_context(|| format!("Cannot open {}", path.display()))?;
        let size = file.metadata()?.len();
        let mut length = [0; 8];
        file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        ensure!(
            length <= 16 * 1024 * 1024 && length + 8 <= size,
            "Invalid Safetensors header"
        );
        let mut header = vec![0; length as usize];
        file.read_exact(&mut header)?;
        let json: BTreeMap<String, Value> = serde_json::from_slice(&header)?;
        let mut entries = BTreeMap::new();
        for (name, item) in json {
            if name == "__metadata__" {
                continue;
            }
            ensure!(
                item["dtype"] == "F32",
                "Unsupported dtype for {name}: expected official F32 checkpoint"
            );
            let shape: Vec<usize> = item["shape"]
                .as_array()
                .context("Missing tensor shape")?
                .iter()
                .map(|n| {
                    n.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .context("Invalid tensor dimension")
                })
                .collect::<Result<_>>()?;
            let start = item["data_offsets"][0]
                .as_u64()
                .context("Missing tensor offset")?;
            let end = item["data_offsets"][1]
                .as_u64()
                .context("Missing tensor offset")?;
            let elements = shape
                .iter()
                .try_fold(1_u64, |n, &d| n.checked_mul(d as u64))
                .context("Tensor size overflow")?;
            ensure!(
                end >= start
                    && end - start == elements.checked_mul(4).context("Tensor byte overflow")?
                    && end <= size - length - 8,
                "Invalid tensor bounds for {name}"
            );
            entries.insert(
                name,
                Entry {
                    shape,
                    offset: 8 + length + start,
                    bytes: end - start,
                },
            );
        }
        Ok(Self {
            path: path.into(),
            entries,
            header,
        })
    }

    pub fn entry(&self, name: &str) -> Result<&Entry> {
        let legacy = match name {
            "emb.weight" => "emb.0.weight",
            "linear.weight" => "linears.0.weight",
            _ => name,
        };
        self.entries
            .get(name)
            .or_else(|| self.entries.get(legacy))
            .with_context(|| format!("Missing MuScriptor weight: {name}"))
    }

    pub fn read(&self, name: &str) -> Result<Vec<f32>> {
        let entry = self.entry(name)?;
        self.read_region(entry.offset, entry.bytes as usize)
    }

    pub fn row(&self, name: &str, row: usize, dim: usize) -> Result<Vec<f32>> {
        let entry = self.entry(name)?;
        ensure!(
            entry.shape.len() == 2 && entry.shape[1] == dim && row < entry.shape[0],
            "Invalid embedding index"
        );
        self.read_region(entry.offset + (row * dim * 4) as u64, dim * 4)
    }

    fn read_region(&self, offset: u64, bytes: usize) -> Result<Vec<f32>> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut data = vec![0; bytes];
        file.read_exact(&mut data)?;
        let values: Vec<_> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(
            values.iter().all(|n| n.is_finite()),
            "Non-finite model weights"
        );
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn rejects_truncated_header_and_out_of_bounds_tensor() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&u64::MAX.to_le_bytes()).unwrap();
        assert!(Weights::open(file.path()).is_err());
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let header = br#"{"emb.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0; 4]).unwrap();
        assert!(Weights::open(file.path()).is_err());
    }
}

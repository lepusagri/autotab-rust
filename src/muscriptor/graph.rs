//! Builds OpenVINO IR directly from the official Safetensors weights in Rust.
//! KV state remains inside the infer request between decoding steps.
use super::{Architecture, weights::Weights};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    fmt::Write as _,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Clone)]
struct Port {
    node: usize,
    port: usize,
    shape: Vec<i64>,
    dtype: &'static str,
}

struct Graph<'a> {
    weights: &'a Weights,
    layers: String,
    edges: String,
    count: usize,
    extra: Vec<u8>,
    base: u64,
}

impl<'a> Graph<'a> {
    fn new(weights: &'a Weights) -> Result<Self> {
        Ok(Self {
            weights,
            layers: String::new(),
            edges: String::new(),
            count: 0,
            extra: Vec::new(),
            base: fs::metadata(&weights.path)?.len(),
        })
    }

    fn node(
        &mut self,
        kind: &str,
        version: u8,
        name: &str,
        data: &str,
        inputs: &[Port],
        outputs: &[(&'static str, Vec<i64>)],
    ) -> Vec<Port> {
        let id = self.count;
        self.count += 1;
        write!(self.layers, "<layer id=\"{id}\" name=\"{name}\" type=\"{kind}\" version=\"opset{version}\"><data {data}/>").unwrap();
        if !inputs.is_empty() {
            self.layers.push_str("<input>");
            for (index, input) in inputs.iter().enumerate() {
                port_xml(&mut self.layers, index, input.dtype, &input.shape, None);
                write!(self.edges, "<edge from-layer=\"{}\" from-port=\"{}\" to-layer=\"{id}\" to-port=\"{index}\"/>", input.node, input.port).unwrap();
            }
            self.layers.push_str("</input>");
        }
        let mut ports = Vec::new();
        if !outputs.is_empty() {
            self.layers.push_str("<output>");
            for (index, (dtype, shape)) in outputs.iter().enumerate() {
                let port = index + inputs.len();
                port_xml(
                    &mut self.layers,
                    port,
                    dtype,
                    shape,
                    Some(&format!("n{id}_{port}")),
                );
                ports.push(Port {
                    node: id,
                    port,
                    shape: shape.clone(),
                    dtype,
                });
            }
            self.layers.push_str("</output>");
        }
        self.layers.push_str("</layer>");
        ports
    }

    fn op(&mut self, kind: &str, version: u8, data: &str, inputs: &[Port], shape: &[i64]) -> Port {
        let name = format!("{kind}_{}", self.count);
        self.node(kind, version, &name, data, inputs, &[("f32", shape.into())])
            .remove(0)
    }

    fn parameter(&mut self, name: &str, shape: &[i64]) -> Port {
        let shape_text = dims(shape);
        self.node(
            "Parameter",
            1,
            name,
            &format!("shape=\"{shape_text}\" element_type=\"f32\""),
            &[],
            &[("f32", shape.into())],
        )
        .remove(0)
    }

    fn weight(&mut self, name: &str, expected: &[usize]) -> Result<Port> {
        let entry = self.weights.entry(name)?;
        ensure!(
            entry.shape == expected,
            "MuScriptor weight shape mismatch: {name}: {:?}, expected {expected:?}",
            entry.shape
        );
        let shape: Vec<_> = expected.iter().map(|&n| n as i64).collect();
        let data = format!(
            "element_type=\"f32\" shape=\"{}\" offset=\"{}\" size=\"{}\"",
            dims(&shape),
            entry.offset,
            entry.bytes
        );
        Ok(self
            .node(
                "Const",
                1,
                &format!("weight_{}", self.count),
                &data,
                &[],
                &[("f32", shape)],
            )
            .remove(0))
    }

    fn integers(&mut self, values: &[i64]) -> Port {
        let offset = self.base + self.extra.len() as u64;
        for value in values {
            self.extra.extend(value.to_le_bytes());
        }
        let data = format!(
            "element_type=\"i64\" shape=\"{}\" offset=\"{offset}\" size=\"{}\"",
            values.len(),
            values.len() * 8
        );
        self.node(
            "Const",
            1,
            &format!("int_{}", self.count),
            &data,
            &[],
            &[("i64", vec![values.len() as i64])],
        )
        .remove(0)
    }

    fn linear(
        &mut self,
        input: Port,
        name: &str,
        width: usize,
        input_width: usize,
        shape: &[i64],
    ) -> Result<Port> {
        let weight = self.weight(name, &[width, input_width])?;
        Ok(self.op(
            "MatMul",
            1,
            "transpose_a=\"false\" transpose_b=\"true\"",
            &[input, weight],
            shape,
        ))
    }

    fn scalar(&mut self, value: i64) -> Port {
        let offset = self.base + self.extra.len() as u64;
        self.extra.extend(value.to_le_bytes());
        let data = format!("element_type=\"i64\" shape=\"\" offset=\"{offset}\" size=\"8\"");
        self.node(
            "Const",
            1,
            &format!("scalar_{}", self.count),
            &data,
            &[],
            &[("i64", vec![])],
        )
        .remove(0)
    }

    fn norm(&mut self, input: Port, name: &str, dim: usize) -> Result<Port> {
        let shape = input.shape.clone();
        let axes = self.integers(&[-1]);
        let norm = self.op(
            "MVN",
            6,
            "normalize_variance=\"true\" eps=\"0.00001\" eps_mode=\"inside_sqrt\"",
            &[input, axes],
            &shape,
        );
        let weight = self.weight(&format!("{name}.weight"), &[dim])?;
        let scaled = self.op(
            "Multiply",
            1,
            "auto_broadcast=\"numpy\"",
            &[norm, weight],
            &shape,
        );
        let bias = self.weight(&format!("{name}.bias"), &[dim])?;
        Ok(self.op(
            "Add",
            1,
            "auto_broadcast=\"numpy\"",
            &[scaled, bias],
            &shape,
        ))
    }

    fn reshape(&mut self, input: Port, pattern: &[i64], shape: &[i64]) -> Port {
        let pattern = self.integers(pattern);
        self.op(
            "Reshape",
            1,
            "special_zero=\"true\"",
            &[input, pattern],
            shape,
        )
    }

    fn transpose(&mut self, input: Port, pattern: &[i64], shape: &[i64]) -> Port {
        let pattern = self.integers(pattern);
        self.op("Transpose", 1, "", &[input, pattern], shape)
    }

    fn state(&mut self, value: Port, name: &str, heads: i64, head_dim: i64) -> Port {
        let shape = vec![1, heads, -1, head_dim];
        let data = format!(
            "element_type=\"f32\" shape=\"1,{heads},0,{head_dim}\" offset=\"0\" size=\"0\""
        );
        let initial = self
            .node(
                "Const",
                1,
                &format!("empty_{name}"),
                &data,
                &[],
                &[("f32", vec![1, heads, 0, head_dim])],
            )
            .remove(0);
        let data = format!(
            "variable_id=\"{name}\" variable_type=\"f32\" variable_shape=\"1,{heads},-1,{head_dim}\""
        );
        let previous = self
            .node(
                "ReadValue",
                6,
                &format!("read_{name}"),
                &data,
                &[initial],
                &[("f32", shape.clone())],
            )
            .remove(0);
        let updated = self.op("Concat", 1, "axis=\"2\"", &[previous, value], &shape);
        self.node(
            "Assign",
            6,
            &format!("save_{name}"),
            &format!("variable_id=\"{name}\""),
            std::slice::from_ref(&updated),
            &[],
        );
        updated
    }

    fn result(&mut self, value: Port) {
        self.node("Result", 1, "result", "", &[value], &[]);
    }

    fn xml(&self) -> String {
        format!(
            "<?xml version=\"1.0\"?><net name=\"muscriptor\" version=\"11\"><layers>{}</layers><edges>{}</edges></net>",
            self.layers, self.edges
        )
    }
}

fn dims(shape: &[i64]) -> String {
    shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}
fn port_xml(xml: &mut String, id: usize, dtype: &str, shape: &[i64], name: Option<&str>) {
    let precision = if dtype == "i64" { "I64" } else { "FP32" };
    write!(xml, "<port id=\"{id}\" precision=\"{precision}\"").unwrap();
    if let Some(name) = name {
        write!(xml, " names=\"{name}\"").unwrap();
    }
    xml.push('>');
    for dim in shape {
        write!(xml, "<dim>{dim}</dim>").unwrap();
    }
    xml.push_str("</port>");
}

pub struct Files {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub bin: PathBuf,
}

pub fn build(weights: &Weights, architecture: Architecture, cache: &Path) -> Result<Files> {
    let Architecture {
        dim,
        heads,
        layers,
        card: _,
    } = architecture;
    let d = dim as i64;
    let h = heads as i64;
    let hd = (dim / heads) as i64;
    let shape = [1, -1, d];
    let mut encoder = Graph::new(weights)?;
    let mel = encoder.parameter("mel", &[1, 501, 512]);
    let projected = encoder.linear(
        mel,
        "condition_provider.conditioners.self_wav.output_proj.weight",
        dim,
        512,
        &[1, 501, d],
    )?;
    let bias = encoder.weight(
        "condition_provider.conditioners.self_wav.output_proj.bias",
        &[dim],
    )?;
    let projected = encoder.op(
        "Add",
        1,
        "auto_broadcast=\"numpy\"",
        &[projected, bias],
        &[1, 501, d],
    );
    encoder.result(projected);

    let mut graph = Graph::new(weights)?;
    let mut x = graph.parameter("features", &shape);
    let mask = graph.parameter("mask", &[1, 1, -1, -1]);
    for layer in 0..layers {
        let prefix = format!("transformer.layers.{layer}");
        let normalized = graph.norm(x.clone(), &format!("{prefix}.norm1"), dim)?;
        let qkv = graph.linear(
            normalized,
            &format!("{prefix}.self_attn.in_proj_weight"),
            dim * 3,
            dim,
            &[1, -1, d * 3],
        )?;
        let axis = graph.scalar(2);
        let qkv = graph.node(
            "Split",
            1,
            &format!("split_{layer}"),
            "num_splits=\"3\"",
            &[qkv, axis],
            &[
                ("f32", shape.into()),
                ("f32", shape.into()),
                ("f32", shape.into()),
            ],
        );
        let mut pieces = Vec::new();
        for piece in qkv {
            let reshaped = graph.reshape(piece, &[0, 0, h, hd], &[1, -1, h, hd]);
            pieces.push(graph.transpose(reshaped, &[0, 2, 1, 3], &[1, h, -1, hd]));
        }
        let key = graph.state(pieces[1].clone(), &format!("key_{layer}"), h, hd);
        let value = graph.state(pieces[2].clone(), &format!("value_{layer}"), h, hd);
        let attention = graph.op(
            "ScaledDotProductAttention",
            13,
            "causal=\"false\"",
            &[pieces[0].clone(), key, value, mask.clone()],
            &[1, h, -1, hd],
        );
        let attention = graph.transpose(attention, &[0, 2, 1, 3], &[1, -1, h, hd]);
        let attention = graph.reshape(attention, &[0, 0, d], &shape);
        let attention = graph.linear(
            attention,
            &format!("{prefix}.self_attn.out_proj.weight"),
            dim,
            dim,
            &shape,
        )?;
        x = graph.op(
            "Add",
            1,
            "auto_broadcast=\"numpy\"",
            &[x, attention],
            &shape,
        );
        let normalized = graph.norm(x.clone(), &format!("{prefix}.norm2"), dim)?;
        let hidden = graph.linear(
            normalized,
            &format!("{prefix}.linear1.weight"),
            dim * 4,
            dim,
            &[1, -1, d * 4],
        )?;
        let hidden = graph.op(
            "Gelu",
            7,
            "approximation_mode=\"erf\"",
            &[hidden],
            &[1, -1, d * 4],
        );
        let hidden = graph.linear(
            hidden,
            &format!("{prefix}.linear2.weight"),
            dim,
            dim * 4,
            &shape,
        )?;
        x = graph.op("Add", 1, "auto_broadcast=\"numpy\"", &[x, hidden], &shape);
    }
    let index = graph.integers(&[-1]);
    let axis = graph.scalar(1);
    let last = graph.op(
        "Gather",
        8,
        "batch_dims=\"0\"",
        &[x, index, axis],
        &[1, 1, d],
    );
    let last = graph.norm(last, "out_norm", dim)?;
    let logits = graph.linear(
        last,
        "linear.weight",
        architecture.card,
        dim,
        &[1, 1, architecture.card as i64],
    )?;
    graph.result(logits);
    let encoder_xml = encoder.xml();
    let decoder_xml = graph.xml();
    let mut hash = Sha256::new();
    hash.update(&weights.header);
    hash.update(&encoder_xml);
    hash.update(&decoder_xml);
    hash.update(&graph.extra);
    hash.update(weights.path.to_string_lossy().as_bytes());
    hash.update(format!("{:?}", fs::metadata(&weights.path)?.modified()?));
    let folder = cache.join(format!("{:x}", hash.finalize()));
    fs::create_dir_all(&folder)?;
    let files = Files {
        encoder: folder.join("encoder.xml"),
        decoder: folder.join("decoder.xml"),
        bin: folder.join("weights.bin"),
    };
    if !files.bin.is_file() {
        println!("  Preparing native MuScriptor graph (one-time local weight cache)...");
        let mut temporary = tempfile::NamedTempFile::new_in(&folder)?;
        std::io::copy(&mut File::open(&weights.path)?, &mut temporary)?;
        temporary.write_all(&graph.extra)?;
        temporary.flush()?;
        match temporary.persist_noclobber(&files.bin) {
            Ok(_) => (),
            Err(e) if files.bin.is_file() => drop(e),
            Err(e) => return Err(e.error).context("Cannot save native weight cache"),
        }
    }
    ensure!(
        fs::metadata(&files.bin)?.len() == graph.base + graph.extra.len() as u64,
        "Incomplete native weight cache"
    );
    write_once(&files.encoder, encoder_xml.as_bytes())?;
    write_once(&files.decoder, decoder_xml.as_bytes())?;
    Ok(files)
}

fn write_once(path: &Path, bytes: &[u8]) -> Result<()> {
    // The content-addressed XML is immutable. Publish atomically so a second
    // process never opens a truncated graph while the first one writes it.
    if path.is_file() {
        return Ok(());
    }
    let mut temporary =
        tempfile::NamedTempFile::new_in(path.parent().context("Missing graph directory")?)?;
    temporary.write_all(bytes)?;
    temporary.flush()?;
    match temporary.persist_noclobber(path) {
        Ok(_) => Ok(()),
        Err(e) if path.is_file() => {
            drop(e);
            Ok(())
        }
        Err(e) => Err(e.error).context("Cannot publish MuScriptor graph"),
    }
}

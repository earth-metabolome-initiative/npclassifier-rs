use std::{
    fs::File,
    io::{Cursor, Read, Seek},
    path::Path,
    str::FromStr,
};

use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use serde::Deserialize;

use crate::{
    NpClassifierError, RawPredictions, classifier::InferenceEngine, fingerprint::FingerprintInput,
    model::ModelHead,
};

/// On-disk weight variant for packed classifier archives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedModelVariant {
    /// Full-precision `f32` kernels.
    F32,
    /// Per-kernel `i8` quantized kernels with `f32` scale factors.
    Q8Kernel,
    /// Packed 4-bit kernels with `f32` scale factors.
    Q4Kernel,
}

impl PackedModelVariant {
    /// Returns the filename suffix used by the packed archive format.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8Kernel => "q8-kernel",
            Self::Q4Kernel => "q4-kernel",
        }
    }
}

impl core::fmt::Display for PackedModelVariant {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.suffix())
    }
}

impl FromStr for PackedModelVariant {
    type Err = NpClassifierError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "f32" => Ok(Self::F32),
            "q8" | "q8-kernel" => Ok(Self::Q8Kernel),
            "q4" | "q4-kernel" => Ok(Self::Q4Kernel),
            other => Err(NpClassifierError::Model(format!(
                "unsupported packed model variant: {other}"
            ))),
        }
    }
}

/// Runtime bundle containing the pathway, superclass, and class packed heads.
#[derive(Debug, Clone)]
pub struct PackedModelSet {
    shared: Option<PackedSequenceModel>,
    pathway: PackedHeadModel,
    superclass: PackedHeadModel,
    class: PackedHeadModel,
}

impl PackedModelSet {
    /// Loads all three classifier heads from a packed model directory.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if any archive is missing or cannot be
    /// decoded.
    pub fn from_dir(
        base_dir: &Path,
        variant: PackedModelVariant,
    ) -> Result<Self, NpClassifierError> {
        let shared_path = base_dir
            .join("shared")
            .join(format!("shared.{}.npz", variant.suffix()));
        Ok(Self {
            shared: shared_path
                .exists()
                .then(|| PackedSequenceModel::from_path(&shared_path, variant))
                .transpose()?,
            pathway: PackedHeadModel::from_path(
                ModelHead::Pathway,
                &base_dir.join(ModelHead::Pathway.as_str()).join(format!(
                    "{}.{}.npz",
                    ModelHead::Pathway.as_str(),
                    variant.suffix()
                )),
                variant,
            )?,
            superclass: PackedHeadModel::from_path(
                ModelHead::Superclass,
                &base_dir.join(ModelHead::Superclass.as_str()).join(format!(
                    "{}.{}.npz",
                    ModelHead::Superclass.as_str(),
                    variant.suffix()
                )),
                variant,
            )?,
            class: PackedHeadModel::from_path(
                ModelHead::Class,
                &base_dir.join(ModelHead::Class.as_str()).join(format!(
                    "{}.{}.npz",
                    ModelHead::Class.as_str(),
                    variant.suffix()
                )),
                variant,
            )?,
        })
    }

    /// Loads all three classifier heads from in-memory packed archives.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if any archive cannot be decoded.
    pub fn from_archives(
        pathway_archive: &[u8],
        superclass_archive: &[u8],
        class_archive: &[u8],
        variant: PackedModelVariant,
    ) -> Result<Self, NpClassifierError> {
        Self::from_archives_with_shared(
            None,
            pathway_archive,
            superclass_archive,
            class_archive,
            variant,
        )
    }

    /// Loads all classifier heads and an optional shared stem from in-memory packed archives.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if any archive cannot be decoded.
    pub fn from_archives_with_shared(
        shared_archive: Option<&[u8]>,
        pathway_archive: &[u8],
        superclass_archive: &[u8],
        class_archive: &[u8],
        variant: PackedModelVariant,
    ) -> Result<Self, NpClassifierError> {
        Ok(Self {
            shared: shared_archive
                .map(|archive| PackedSequenceModel::from_bytes(archive, variant))
                .transpose()?,
            pathway: PackedHeadModel::from_bytes(ModelHead::Pathway, pathway_archive, variant)?,
            superclass: PackedHeadModel::from_bytes(
                ModelHead::Superclass,
                superclass_archive,
                variant,
            )?,
            class: PackedHeadModel::from_bytes(ModelHead::Class, class_archive, variant)?,
        })
    }
}

impl InferenceEngine for PackedModelSet {
    fn predict(&self, fingerprint: &FingerprintInput) -> Result<RawPredictions, NpClassifierError> {
        let input = fingerprint.concatenated();
        let shared = match &self.shared {
            Some(model) => model.forward(&input)?,
            None => input,
        };

        Ok(RawPredictions {
            pathway: self.pathway.forward(&shared)?,
            superclass: self.superclass.forward(&shared)?,
            class: self.class.forward(&shared)?,
        })
    }
}

/// Runtime for one packed dense stack.
#[derive(Debug, Clone)]
struct PackedSequenceModel {
    layers: Vec<PackedLayer>,
}

impl PackedSequenceModel {
    /// Loads one packed dense stack from a packed archive on disk.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if the archive cannot be read or does
    /// not match the expected packed format.
    fn from_path(
        archive_path: &Path,
        variant: PackedModelVariant,
    ) -> Result<Self, NpClassifierError> {
        let file = File::open(archive_path)?;
        Self::from_reader(file, variant)
    }

    /// Loads one packed dense stack from an in-memory packed archive.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if the archive cannot be decoded.
    fn from_bytes(
        archive_bytes: &[u8],
        variant: PackedModelVariant,
    ) -> Result<Self, NpClassifierError> {
        Self::from_reader(Cursor::new(archive_bytes), variant)
    }

    /// Loads one packed dense stack from any seekable reader.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if the archive cannot be decoded or if
    /// it contains unsupported layer metadata.
    fn from_reader<R: Read + Seek>(
        reader: R,
        variant: PackedModelVariant,
    ) -> Result<Self, NpClassifierError> {
        let mut archive =
            NpzReader::new(reader).map_err(|error| NpClassifierError::Model(error.to_string()))?;
        let metadata = read_metadata(&mut archive)?;
        let mut layers = Vec::with_capacity(metadata.layers.len());

        for layer in metadata.layers {
            match layer.op.as_str() {
                "concat" => layers.push(PackedLayer::Concat),
                "dropout" => layers.push(PackedLayer::Dropout),
                "dense" => layers.push(PackedLayer::Dense(load_dense_layer(
                    &mut archive,
                    &layer,
                    variant,
                )?)),
                "batch_norm" => layers.push(PackedLayer::BatchNorm(load_batch_norm_layer(
                    &mut archive,
                    &layer,
                )?)),
                other => {
                    return Err(NpClassifierError::Model(format!(
                        "unsupported packed layer op {other}"
                    )));
                }
            }
        }

        Ok(Self { layers })
    }

    /// Runs the packed dense stack for one input vector.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if any intermediate layer shape is
    /// inconsistent.
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, NpClassifierError> {
        let mut x = input.to_vec();

        for layer in &self.layers {
            match layer {
                PackedLayer::Concat | PackedLayer::Dropout => {}
                PackedLayer::Dense(dense) => {
                    x = dense.forward(&x)?;
                }
                PackedLayer::BatchNorm(batch_norm) => {
                    x = batch_norm.forward(&x)?;
                }
            }
        }
        Ok(x)
    }
}

/// Runtime for one packed dense classifier head.
#[derive(Debug, Clone)]
pub struct PackedHeadModel {
    head: ModelHead,
    stack: PackedSequenceModel,
}

impl PackedHeadModel {
    /// Loads one classifier head from a packed archive on disk.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if the archive cannot be read or does
    /// not match the expected packed format.
    pub fn from_path(
        head: ModelHead,
        archive_path: &Path,
        variant: PackedModelVariant,
    ) -> Result<Self, NpClassifierError> {
        Ok(Self {
            head,
            stack: PackedSequenceModel::from_path(archive_path, variant)?,
        })
    }

    /// Loads one classifier head from an in-memory packed archive.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if the archive cannot be decoded.
    pub fn from_bytes(
        head: ModelHead,
        archive_bytes: &[u8],
        variant: PackedModelVariant,
    ) -> Result<Self, NpClassifierError> {
        Ok(Self {
            head,
            stack: PackedSequenceModel::from_bytes(archive_bytes, variant)?,
        })
    }

    /// Runs the packed dense stack for one input vector and validates the final
    /// width against the expected head contract.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if the input width does not match the
    /// head contract or if any intermediate layer shape is inconsistent.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, NpClassifierError> {
        let x = self.stack.forward(input)?;
        if x.len() == self.head.output_width() {
            Ok(x)
        } else {
            Err(NpClassifierError::InvalidPredictionWidth {
                head: self.head,
                expected: self.head.output_width(),
                actual: x.len(),
            })
        }
    }
}

#[derive(Debug, Clone)]
enum PackedLayer {
    Concat,
    Dense(DenseRuntime),
    BatchNorm(BatchNormRuntime),
    Dropout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Activation {
    Relu,
    Sigmoid,
    Linear,
}

impl Activation {
    fn parse(name: &str) -> Result<Self, NpClassifierError> {
        match name {
            "relu" => Ok(Self::Relu),
            "sigmoid" => Ok(Self::Sigmoid),
            "linear" => Ok(Self::Linear),
            other => Err(NpClassifierError::Model(format!(
                "unsupported activation {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
enum PackedKernel {
    F32 {
        values: Vec<f32>,
        input: usize,
        output: usize,
    },
    Q8 {
        values: Vec<i8>,
        scale: f32,
        input: usize,
        output: usize,
    },
    Q4 {
        packed_values: Vec<u8>,
        scales: Vec<f32>,
        block_size: usize,
        input: usize,
        output: usize,
    },
}

#[derive(Debug, Clone)]
struct DenseRuntime {
    name: String,
    bias: Vec<f32>,
    activation: Activation,
    kernel: PackedKernel,
}

impl DenseRuntime {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, NpClassifierError> {
        let (input_width, output_width) = self.kernel.dims();
        if input.len() != input_width {
            return Err(NpClassifierError::Model(format!(
                "dense layer {} expected {} inputs, got {}",
                self.name,
                input_width,
                input.len()
            )));
        }
        if self.bias.len() != output_width {
            return Err(NpClassifierError::Model(format!(
                "dense layer {} has mismatched bias width {} for output {}",
                self.name,
                self.bias.len(),
                output_width
            )));
        }

        let mut output = self.bias.clone();
        match &self.kernel {
            PackedKernel::F32 { values, .. } => {
                Self::accumulate_f32(&mut output, input, values, output_width);
            }
            PackedKernel::Q8 { values, scale, .. } => {
                Self::accumulate_q8(&mut output, input, values, *scale, output_width);
            }
            PackedKernel::Q4 {
                packed_values,
                scales,
                block_size,
                ..
            } => {
                Self::accumulate_q4(
                    &self.name,
                    &mut output,
                    input,
                    packed_values,
                    scales,
                    *block_size,
                )?;
            }
        }

        apply_activation(&mut output, self.activation);
        Ok(output)
    }

    fn accumulate_f32(output: &mut [f32], input: &[f32], values: &[f32], output_width: usize) {
        let mut offset = 0;
        for &input_value in input {
            for output_index in 0..output_width {
                output[output_index] += input_value * values[offset + output_index];
            }
            offset += output_width;
        }
    }

    fn accumulate_q8(
        output: &mut [f32],
        input: &[f32],
        values: &[i8],
        scale: f32,
        output_width: usize,
    ) {
        let mut offset = 0;
        for &input_value in input {
            for output_index in 0..output_width {
                output[output_index] +=
                    input_value * (f32::from(values[offset + output_index]) * scale);
            }
            offset += output_width;
        }
    }

    fn accumulate_q4(
        layer_name: &str,
        output: &mut [f32],
        input: &[f32],
        packed_values: &[u8],
        scales: &[f32],
        block_size: usize,
    ) -> Result<(), NpClassifierError> {
        let input_width = input.len();
        let output_width = output.len();
        let output_blocks = output_width.div_ceil(block_size);
        let expected_scales = input_width * output_blocks;
        if !(scales.len() == 1 || scales.len() == expected_scales) {
            return Err(NpClassifierError::Model(format!(
                "dense layer {} has invalid q4 scale count {} for {}x{} with block size {}",
                layer_name,
                scales.len(),
                input_width,
                output_width,
                block_size
            )));
        }

        let total = input_width * output_width;
        let expected_packed_len = total.div_ceil(2);
        if packed_values.len() != expected_packed_len {
            return Err(NpClassifierError::Model(format!(
                "dense layer {} has invalid q4 payload length {} for {} values",
                layer_name,
                packed_values.len(),
                total
            )));
        }

        for index in 0..total {
            let byte = packed_values[index / 2];
            let nibble = if index % 2 == 0 {
                byte & 0x0F
            } else {
                byte >> 4
            };
            let quantized = f32::from(sign_extend_q4(nibble));
            let input_index = index / output_width;
            let output_index = index % output_width;
            let scale = if scales.len() == 1 {
                scales[0]
            } else {
                scales[input_index * output_blocks + (output_index / block_size)]
            };
            output[output_index] += input[input_index] * (quantized * scale);
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct BatchNormRuntime {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    mean: Vec<f32>,
    variance: Vec<f32>,
    epsilon: f32,
}

impl BatchNormRuntime {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, NpClassifierError> {
        let width = input.len();
        if self.gamma.len() != width
            || self.beta.len() != width
            || self.mean.len() != width
            || self.variance.len() != width
        {
            return Err(NpClassifierError::Model(format!(
                "batch norm width mismatch: input {width}, gamma {}, beta {}, mean {}, variance {}",
                self.gamma.len(),
                self.beta.len(),
                self.mean.len(),
                self.variance.len()
            )));
        }

        let mut output = Vec::with_capacity(width);
        for (((input_value, gamma), mean), (variance, beta)) in input
            .iter()
            .zip(&self.gamma)
            .zip(&self.mean)
            .zip(self.variance.iter().zip(&self.beta))
        {
            let normalized = gamma * (input_value - mean) / (variance + self.epsilon).sqrt();
            output.push(normalized + beta);
        }
        Ok(output)
    }
}

impl PackedKernel {
    const fn dims(&self) -> (usize, usize) {
        match self {
            Self::F32 { input, output, .. }
            | Self::Q8 { input, output, .. }
            | Self::Q4 { input, output, .. } => (*input, *output),
        }
    }
}

#[derive(Debug, Deserialize)]
struct PackedArchiveMetadata {
    layers: Vec<PackedArchiveLayer>,
}

#[derive(Debug, Deserialize)]
struct PackedArchiveLayer {
    op: String,
    name: String,
    #[serde(default)]
    config: serde_json::Value,
}

fn read_metadata<R: Read + Seek>(
    archive: &mut NpzReader<R>,
) -> Result<PackedArchiveMetadata, NpClassifierError> {
    let bytes = read_array1_u8(archive, "__metadata__/json")?;
    serde_json::from_slice::<PackedArchiveMetadata>(&bytes).map_err(Into::into)
}

fn load_dense_layer<R: Read + Seek>(
    archive: &mut NpzReader<R>,
    layer: &PackedArchiveLayer,
    variant: PackedModelVariant,
) -> Result<DenseRuntime, NpClassifierError> {
    let activation = Activation::parse(
        layer
            .config
            .get("activation")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("linear"),
    )?;
    let bias = read_array1_f32(archive, &format!("{}__bias", layer.name))?;
    let variant = dense_kernel_variant(layer, variant)?;

    let kernel = match variant {
        PackedModelVariant::F32 => {
            let (values, input, output) =
                read_array2_f32(archive, &format!("{}__kernel", layer.name))?;
            PackedKernel::F32 {
                values,
                input,
                output,
            }
        }
        PackedModelVariant::Q8Kernel => {
            let (values, input, output) =
                read_array2_i8(archive, &format!("{}__kernel", layer.name))?;
            let scale = read_scalar_f32(archive, &format!("{}__kernel__scale", layer.name))?;
            PackedKernel::Q8 {
                values,
                scale,
                input,
                output,
            }
        }
        PackedModelVariant::Q4Kernel => {
            let packed_values = read_array1_u8(archive, &format!("{}__kernel", layer.name))?;
            let shape = read_array1_i32(archive, &format!("{}__kernel__shape", layer.name))?;
            if shape.len() != 2 {
                return Err(NpClassifierError::Model(format!(
                    "q4 kernel {} has invalid shape descriptor length {}",
                    layer.name,
                    shape.len()
                )));
            }
            let input = usize::try_from(shape[0]).map_err(|error| {
                NpClassifierError::Model(format!(
                    "q4 kernel {} has invalid input dimension: {error}",
                    layer.name
                ))
            })?;
            let output = usize::try_from(shape[1]).map_err(|error| {
                NpClassifierError::Model(format!(
                    "q4 kernel {} has invalid output dimension: {error}",
                    layer.name
                ))
            })?;
            let scales = try_read_array1_f32(archive, &format!("{}__kernel__scales", layer.name))
                .unwrap_or_else(|| {
                    vec![
                        read_scalar_f32(archive, &format!("{}__kernel__scale", layer.name))
                            .expect("legacy q4 scale should decode"),
                    ]
                });
            let block_size = infer_q4_block_size(input, output, scales.len()).map_err(|error| {
                NpClassifierError::Model(format!(
                    "q4 kernel {} has invalid block configuration: {error}",
                    layer.name
                ))
            })?;
            PackedKernel::Q4 {
                packed_values,
                scales,
                block_size,
                input,
                output,
            }
        }
    };

    Ok(DenseRuntime {
        name: layer.name.clone(),
        bias,
        activation,
        kernel,
    })
}

fn dense_kernel_variant(
    layer: &PackedArchiveLayer,
    default_variant: PackedModelVariant,
) -> Result<PackedModelVariant, NpClassifierError> {
    match layer
        .config
        .get("kernel_format")
        .and_then(serde_json::Value::as_str)
    {
        Some(kernel_format) => PackedModelVariant::from_str(kernel_format),
        None => Ok(default_variant),
    }
}

fn load_batch_norm_layer<R: Read + Seek>(
    archive: &mut NpzReader<R>,
    layer: &PackedArchiveLayer,
) -> Result<BatchNormRuntime, NpClassifierError> {
    let epsilon = layer
        .config
        .get("epsilon")
        .cloned()
        .map(serde_json::from_value::<f32>)
        .transpose()
        .map_err(|error| {
            NpClassifierError::Model(format!(
                "batch norm {} has invalid epsilon: {error}",
                layer.name
            ))
        })?
        .unwrap_or(1e-3);

    Ok(BatchNormRuntime {
        gamma: read_array1_f32(archive, &format!("{}__gamma", layer.name))?,
        beta: read_array1_f32(archive, &format!("{}__beta", layer.name))?,
        mean: read_array1_f32(archive, &format!("{}__moving_mean", layer.name))?,
        variance: read_array1_f32(archive, &format!("{}__moving_variance", layer.name))?,
        epsilon,
    })
}

fn apply_activation(values: &mut [f32], activation: Activation) {
    match activation {
        Activation::Linear => {}
        Activation::Relu => {
            for value in values {
                if *value < 0.0 {
                    *value = 0.0;
                }
            }
        }
        Activation::Sigmoid => {
            for value in values {
                *value = sigmoid(*value);
            }
        }
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn sign_extend_q4(value: u8) -> i8 {
    ((value & 0x0F) << 4).cast_signed() >> 4
}

fn read_array1_f32<R: Read + Seek>(
    archive: &mut NpzReader<R>,
    name: &str,
) -> Result<Vec<f32>, NpClassifierError> {
    let array: Array1<f32> = archive
        .by_name(name)
        .map_err(|error| NpClassifierError::Model(error.to_string()))?;
    Ok(into_vec1(array))
}

fn try_read_array1_f32<R: Read + Seek>(archive: &mut NpzReader<R>, name: &str) -> Option<Vec<f32>> {
    match archive.by_name(name) {
        Ok(array) => Some(into_vec1(array)),
        Err(_) => None,
    }
}

fn read_array1_i32<R: Read + Seek>(
    archive: &mut NpzReader<R>,
    name: &str,
) -> Result<Vec<i32>, NpClassifierError> {
    let array: Array1<i32> = archive
        .by_name(name)
        .map_err(|error| NpClassifierError::Model(error.to_string()))?;
    Ok(into_vec1(array))
}

fn read_array1_u8<R: Read + Seek>(
    archive: &mut NpzReader<R>,
    name: &str,
) -> Result<Vec<u8>, NpClassifierError> {
    let array: Array1<u8> = archive
        .by_name(name)
        .map_err(|error| NpClassifierError::Model(error.to_string()))?;
    Ok(into_vec1(array))
}

fn read_scalar_f32<R: Read + Seek>(
    archive: &mut NpzReader<R>,
    name: &str,
) -> Result<f32, NpClassifierError> {
    let values = read_array1_f32(archive, name)?;
    values
        .into_iter()
        .next()
        .ok_or_else(|| NpClassifierError::Model(format!("packed archive scalar {name} was empty")))
}

fn read_array2_f32<R: Read + Seek>(
    archive: &mut NpzReader<R>,
    name: &str,
) -> Result<(Vec<f32>, usize, usize), NpClassifierError> {
    let array: Array2<f32> = archive
        .by_name(name)
        .map_err(|error| NpClassifierError::Model(error.to_string()))?;
    Ok(into_vec2(array))
}

fn read_array2_i8<R: Read + Seek>(
    archive: &mut NpzReader<R>,
    name: &str,
) -> Result<(Vec<i8>, usize, usize), NpClassifierError> {
    let array: Array2<i8> = archive
        .by_name(name)
        .map_err(|error| NpClassifierError::Model(error.to_string()))?;
    Ok(into_vec2(array))
}

fn into_vec1<T>(array: Array1<T>) -> Vec<T> {
    array.into_iter().collect()
}

fn into_vec2<T>(array: Array2<T>) -> (Vec<T>, usize, usize) {
    let shape = array.shape().to_vec();
    (array.into_iter().collect(), shape[0], shape[1])
}

fn infer_q4_block_size(
    input: usize,
    output: usize,
    scale_count: usize,
) -> Result<usize, &'static str> {
    if scale_count == 1 {
        return Ok(output);
    }
    if input == 0 || output == 0 {
        return Err("q4 kernel dimensions must be non-zero");
    }
    if !scale_count.is_multiple_of(input) {
        return Err("q4 scale count must be divisible by the input width");
    }

    let output_blocks = scale_count / input;
    if output_blocks == 0 {
        return Err("q4 scale count produced zero output blocks");
    }

    Ok(output.div_ceil(output_blocks))
}

#[cfg(test)]
mod tests {
    use super::{Activation, DenseRuntime, PackedKernel, sign_extend_q4};

    #[test]
    fn q8_dense_matches_reference_computation() {
        let dense = DenseRuntime {
            name: "toy".to_owned(),
            bias: vec![0.1, -0.2],
            activation: Activation::Linear,
            kernel: PackedKernel::Q8 {
                values: vec![2, -4, 6, 8],
                scale: 0.25,
                input: 2,
                output: 2,
            },
        };

        let output = dense.forward(&[1.5, -2.0]).expect("dense should run");

        assert!((output[0] - (-2.15)).abs() < 1e-5);
        assert!((output[1] - (-5.7)).abs() < 1e-5);
    }

    #[test]
    fn q4_dense_matches_reference_computation() {
        let dense = DenseRuntime {
            name: "toy".to_owned(),
            bias: vec![0.0, 1.0],
            activation: Activation::Linear,
            kernel: PackedKernel::Q4 {
                packed_values: vec![0xF8, 0x79],
                scales: vec![0.5],
                block_size: 2,
                input: 2,
                output: 2,
            },
        };

        let output = dense.forward(&[2.0, -1.0]).expect("dense should run");

        assert!((output[0] - (-4.5)).abs() < 1e-5);
        assert!((output[1] - (-3.5)).abs() < 1e-5);
    }

    #[test]
    fn q4_sign_extension_matches_burn_storage() {
        assert_eq!(sign_extend_q4(0x0), 0);
        assert_eq!(sign_extend_q4(0x7), 7);
        assert_eq!(sign_extend_q4(0x8), -8);
        assert_eq!(sign_extend_q4(0xF), -1);
    }
}

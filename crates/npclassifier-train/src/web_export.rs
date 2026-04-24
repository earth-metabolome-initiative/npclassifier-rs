//! Export packed q4 web artifacts from a trained model.

use std::fs::File;
use std::path::Path;

use burn::prelude::Backend;
use ndarray::{Array1, Array2};
use ndarray_npy::NpzWriter;
use serde::Serialize;
use serde_json::json;

use npclassifier_core::ClassificationThresholds;

use crate::error::TrainingError;
use crate::model::{
    ExportedBatchNormLayer, ExportedDenseLayer, ExportedHead, ExportedHeadLayer, ExportedKernel,
    StudentModel,
};

const Q4_BLOCK_SIZE: usize = 32;

#[derive(Debug, Serialize)]
struct PackedArchiveMetadata {
    layers: Vec<PackedArchiveLayer>,
}

#[derive(Debug, Serialize)]
struct PackedArchiveLayer {
    op: &'static str,
    name: String,
    #[serde(default)]
    config: serde_json::Value,
}

/// Exports q4 packed archives and calibrated thresholds for the browser app.
///
/// # Errors
///
/// Returns an error if model tensors cannot be exported or if the archive files
/// cannot be written.
pub fn export_web_model<B: Backend>(
    output_dir: &Path,
    model: &StudentModel<B>,
    thresholds: ClassificationThresholds,
) -> Result<(), TrainingError> {
    std::fs::create_dir_all(output_dir)?;
    let shared = model.export_shared_stem().map_err(TrainingError::Burn)?;
    let [pathway, superclass, class] = model.export_heads().map_err(TrainingError::Burn)?;

    if let Some(shared) = shared {
        export_stack(output_dir, "shared", shared, "f32")?;
    }
    export_head(output_dir, pathway, "f32")?;
    export_head(output_dir, superclass, "f32")?;
    export_head(output_dir, class, "f32")?;
    std::fs::write(
        output_dir.join("thresholds.json"),
        serde_json::to_string_pretty(&thresholds)?,
    )?;

    Ok(())
}

/// Exports a backend-independent q4 packed browser model.
///
/// Hidden dense layers are quantized to signed 4-bit values with per-row
/// output-block scales. Output layers remain `f32` because their widths are not
/// uniformly divisible by the block size and mixed kernels are already part of
/// the packed archive schema.
///
/// # Errors
///
/// Returns an error if model tensors cannot be exported or if the archive files
/// cannot be written.
pub fn export_web_model_q4<B: Backend>(
    output_dir: &Path,
    model: &StudentModel<B>,
    thresholds: ClassificationThresholds,
) -> Result<(), TrainingError> {
    std::fs::create_dir_all(output_dir)?;
    let shared = model
        .export_shared_stem()
        .map_err(TrainingError::Burn)?
        .map(quantize_layers_q4)
        .transpose()?;
    let [pathway, superclass, class] = model.export_heads().map_err(TrainingError::Burn)?;
    let pathway = quantize_head_q4(pathway)?;
    let superclass = quantize_head_q4(superclass)?;
    let class = quantize_head_q4(class)?;

    if let Some(shared) = shared {
        export_stack(output_dir, "shared", shared, "q4-kernel")?;
    }
    export_head(output_dir, pathway, "q4-kernel")?;
    export_head(output_dir, superclass, "q4-kernel")?;
    export_head(output_dir, class, "q4-kernel")?;
    std::fs::write(
        output_dir.join("thresholds.json"),
        serde_json::to_string_pretty(&thresholds)?,
    )?;

    Ok(())
}

fn export_stack(
    output_dir: &Path,
    name: &str,
    layers: Vec<ExportedHeadLayer>,
    suffix: &str,
) -> Result<(), TrainingError> {
    let stack_dir = output_dir.join(name);
    std::fs::create_dir_all(&stack_dir)?;
    let archive_path = stack_dir.join(format!("{name}.{suffix}.npz"));
    let file = File::create(&archive_path)?;
    let mut archive = NpzWriter::new(file);

    let metadata = PackedArchiveMetadata {
        layers: layers
            .iter()
            .map(|layer| match layer {
                ExportedHeadLayer::Concat => PackedArchiveLayer {
                    op: "concat",
                    name: "concat".to_owned(),
                    config: serde_json::Value::Null,
                },
                ExportedHeadLayer::Dropout => PackedArchiveLayer {
                    op: "dropout",
                    name: "dropout".to_owned(),
                    config: serde_json::Value::Null,
                },
                ExportedHeadLayer::Dense(layer) => PackedArchiveLayer {
                    op: "dense",
                    name: layer.name.clone(),
                    config: json!({
                        "activation": layer.activation,
                        "kernel_format": match layer.kernel {
                            ExportedKernel::F32 { .. } => "f32",
                            ExportedKernel::Q4Block { .. } => "q4-kernel",
                        },
                    }),
                },
                ExportedHeadLayer::BatchNorm(layer) => PackedArchiveLayer {
                    op: "batch_norm",
                    name: layer.name.clone(),
                    config: json!({ "epsilon": layer.epsilon }),
                },
            })
            .collect(),
    };
    archive
        .add_array(
            "__metadata__/json",
            &Array1::from_vec(serde_json::to_vec(&metadata)?),
        )
        .map_err(npz_error)?;

    for layer in layers {
        match layer {
            ExportedHeadLayer::Concat | ExportedHeadLayer::Dropout => {}
            ExportedHeadLayer::Dense(layer) => write_dense_layer(&mut archive, layer)?,
            ExportedHeadLayer::BatchNorm(layer) => write_batch_norm_layer(&mut archive, layer)?,
        }
    }

    archive.finish().map_err(npz_error)?;
    Ok(())
}

fn export_head(output_dir: &Path, head: ExportedHead, suffix: &str) -> Result<(), TrainingError> {
    export_stack(output_dir, head.head.as_str(), head.layers, suffix)
}

fn quantize_head_q4(head: ExportedHead) -> Result<ExportedHead, TrainingError> {
    Ok(ExportedHead {
        head: head.head,
        layers: quantize_layers_q4(head.layers)?,
    })
}

fn quantize_layers_q4(
    layers: Vec<ExportedHeadLayer>,
) -> Result<Vec<ExportedHeadLayer>, TrainingError> {
    layers
        .into_iter()
        .map(|layer| match layer {
            ExportedHeadLayer::Dense(layer) => {
                quantize_dense_layer_q4(layer).map(ExportedHeadLayer::Dense)
            }
            other => Ok(other),
        })
        .collect()
}

fn quantize_dense_layer_q4(layer: ExportedDenseLayer) -> Result<ExportedDenseLayer, TrainingError> {
    if layer.name == "output" {
        return Ok(layer);
    }

    let ExportedDenseLayer {
        name,
        activation,
        bias,
        kernel,
    } = layer;
    let kernel = match kernel {
        ExportedKernel::F32 {
            values,
            input,
            output,
        } => quantize_f32_kernel_q4(&values, input, output)?,
        ExportedKernel::Q4Block { .. } => kernel,
    };

    Ok(ExportedDenseLayer {
        name,
        activation,
        bias,
        kernel,
    })
}

fn quantize_f32_kernel_q4(
    values: &[f32],
    input: usize,
    output: usize,
) -> Result<ExportedKernel, TrainingError> {
    if values.len() != input * output {
        return Err(TrainingError::Dataset(format!(
            "dense kernel has {} values but shape is {input}x{output}",
            values.len()
        )));
    }

    let output_blocks = output.div_ceil(Q4_BLOCK_SIZE);
    let mut packed_values = vec![0_u8; values.len().div_ceil(2)];
    let mut scales = Vec::with_capacity(input * output_blocks);

    for input_index in 0..input {
        let row_offset = input_index * output;
        for block_index in 0..output_blocks {
            let block_start = block_index * Q4_BLOCK_SIZE;
            let block_end = (block_start + Q4_BLOCK_SIZE).min(output);
            let scale = q4_scale(&values[row_offset + block_start..row_offset + block_end]);
            scales.push(scale);

            for output_index in block_start..block_end {
                let value_index = row_offset + output_index;
                let quantized = q4_nibble(values[value_index], scale);
                let packed = &mut packed_values[value_index / 2];
                if value_index.is_multiple_of(2) {
                    *packed = (*packed & 0xF0) | quantized;
                } else {
                    *packed = (*packed & 0x0F) | (quantized << 4);
                }
            }
        }
    }

    Ok(ExportedKernel::Q4Block {
        packed_values,
        scales,
        input,
        output,
    })
}

fn q4_scale(values: &[f32]) -> f32 {
    let max_abs = values
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    if max_abs <= f32::EPSILON {
        1.0
    } else {
        max_abs / 7.0
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn q4_nibble(value: f32, scale: f32) -> u8 {
    let quantized = (value / scale).round().clamp(-8.0, 7.0) as i8;
    quantized as u8 & 0x0F
}

fn write_dense_layer(
    archive: &mut NpzWriter<File>,
    layer: ExportedDenseLayer,
) -> Result<(), TrainingError> {
    archive
        .add_array(
            format!("{}__bias", layer.name),
            &Array1::from_vec(layer.bias),
        )
        .map_err(npz_error)?;

    match layer.kernel {
        ExportedKernel::F32 {
            values,
            input,
            output,
        } => {
            let array = Array2::from_shape_vec((input, output), values)
                .map_err(|error| TrainingError::Dataset(error.to_string()))?;
            archive
                .add_array(format!("{}__kernel", layer.name), &array)
                .map_err(npz_error)?;
        }
        ExportedKernel::Q4Block {
            packed_values,
            scales,
            input,
            output,
        } => {
            archive
                .add_array(
                    format!("{}__kernel", layer.name),
                    &Array1::from_vec(packed_values),
                )
                .map_err(npz_error)?;
            archive
                .add_array(
                    format!("{}__kernel__shape", layer.name),
                    &Array1::from_vec(vec![
                        i32::try_from(input)
                            .map_err(|error| TrainingError::Dataset(error.to_string()))?,
                        i32::try_from(output)
                            .map_err(|error| TrainingError::Dataset(error.to_string()))?,
                    ]),
                )
                .map_err(npz_error)?;
            archive
                .add_array(
                    format!("{}__kernel__scales", layer.name),
                    &Array1::from_vec(scales),
                )
                .map_err(npz_error)?;
        }
    }

    Ok(())
}

fn write_batch_norm_layer(
    archive: &mut NpzWriter<File>,
    layer: ExportedBatchNormLayer,
) -> Result<(), TrainingError> {
    archive
        .add_array(
            format!("{}__gamma", layer.name),
            &Array1::from_vec(layer.gamma),
        )
        .map_err(npz_error)?;
    archive
        .add_array(
            format!("{}__beta", layer.name),
            &Array1::from_vec(layer.beta),
        )
        .map_err(npz_error)?;
    archive
        .add_array(
            format!("{}__moving_mean", layer.name),
            &Array1::from_vec(layer.mean),
        )
        .map_err(npz_error)?;
    archive
        .add_array(
            format!("{}__moving_variance", layer.name),
            &Array1::from_vec(layer.variance),
        )
        .map_err(npz_error)?;
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
fn npz_error(error: ndarray_npy::WriteNpzError) -> TrainingError {
    TrainingError::Dataset(error.to_string())
}

//! Export the trained q4 model into packed archives for the web app.

use std::fs::File;
use std::path::{Path, PathBuf};

use burn::backend::Cuda;
use burn::prelude::Module;
use clap::Parser;
use ndarray::{Array1, Array2};
use ndarray_npy::NpzWriter;
use serde::{Deserialize, Serialize};
use serde_json::json;

use npclassifier_core::ClassificationThresholds;
use npclassifier_train::error::TrainingError;
use npclassifier_train::model::{
    ExportedBatchNormLayer, ExportedDenseLayer, ExportedHead, ExportedHeadLayer, ExportedKernel,
    StudentModelConfig,
};

#[derive(Debug, Parser)]
#[command(name = "npclassifier-export-web-model")]
#[command(about = "Export the trained q4 model into packed web archives")]
struct Cli {
    #[arg(long)]
    artifact_dir: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
}

#[derive(Debug, Deserialize)]
struct SavedTrainingConfig {
    model: StudentModelConfig,
    hard_label_weight: f32,
    teacher_weight: f32,
}

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

fn main() -> Result<(), TrainingError> {
    let cli = Cli::parse();
    export_web_model(&cli.artifact_dir, &cli.output_dir, cli.cuda_device)
}

fn export_web_model(
    artifact_dir: &Path,
    output_dir: &Path,
    cuda_device: usize,
) -> Result<(), TrainingError> {
    let config = load_saved_training_config(artifact_dir)?;
    let thresholds = load_calibrated_thresholds(artifact_dir)?;
    type BackendImpl = Cuda<f32, i32>;
    let device = burn::backend::cuda::CudaDevice::new(cuda_device);
    let model = config
        .model
        .init::<BackendImpl>(&device, config.hard_label_weight, config.teacher_weight)
        .load_file(
            artifact_dir
                .join("quantized")
                .join("q4-block32")
                .join("model"),
            &burn::record::CompactRecorder::new(),
            &device,
        )
        .map_err(|error| TrainingError::Burn(error.to_string()))?;

    let shared = model.export_shared_stem().map_err(TrainingError::Burn)?;
    let [pathway, superclass, class] = model.export_heads().map_err(TrainingError::Burn)?;

    if let Some(shared) = shared {
        export_stack(output_dir, "shared", shared)?;
    }
    export_head(output_dir, pathway)?;
    export_head(output_dir, superclass)?;
    export_head(output_dir, class)?;
    std::fs::write(
        output_dir.join("thresholds.json"),
        serde_json::to_string_pretty(&thresholds)?,
    )?;

    Ok(())
}

fn load_saved_training_config(artifact_dir: &Path) -> Result<SavedTrainingConfig, TrainingError> {
    Ok(serde_json::from_str(&std::fs::read_to_string(
        artifact_dir.join("training-config.json"),
    )?)?)
}

fn load_calibrated_thresholds(
    artifact_dir: &Path,
) -> Result<ClassificationThresholds, TrainingError> {
    let path = artifact_dir.join("thresholds.json");
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn export_stack(
    output_dir: &Path,
    name: &str,
    layers: Vec<ExportedHeadLayer>,
) -> Result<(), TrainingError> {
    let stack_dir = output_dir.join(name);
    std::fs::create_dir_all(&stack_dir)?;
    let archive_path = stack_dir.join(format!("{name}.q4-kernel.npz"));
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

fn export_head(output_dir: &Path, head: ExportedHead) -> Result<(), TrainingError> {
    export_stack(output_dir, head.head.as_str(), head.layers)
}

fn write_dense_layer(
    archive: &mut NpzWriter<File>,
    layer: ExportedDenseLayer,
) -> Result<(), TrainingError> {
    archive
        .add_array(
            &format!("{}__bias", layer.name),
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
                .add_array(&format!("{}__kernel", layer.name), &array)
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
                    &format!("{}__kernel", layer.name),
                    &Array1::from_vec(packed_values),
                )
                .map_err(npz_error)?;
            archive
                .add_array(
                    &format!("{}__kernel__shape", layer.name),
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
                    &format!("{}__kernel__scales", layer.name),
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
            &format!("{}__gamma", layer.name),
            &Array1::from_vec(layer.gamma),
        )
        .map_err(npz_error)?;
    archive
        .add_array(
            &format!("{}__beta", layer.name),
            &Array1::from_vec(layer.beta),
        )
        .map_err(npz_error)?;
    archive
        .add_array(
            &format!("{}__moving_mean", layer.name),
            &Array1::from_vec(layer.mean),
        )
        .map_err(npz_error)?;
    archive
        .add_array(
            &format!("{}__moving_variance", layer.name),
            &Array1::from_vec(layer.variance),
        )
        .map_err(npz_error)?;
    Ok(())
}

fn npz_error(error: ndarray_npy::WriteNpzError) -> TrainingError {
    TrainingError::Dataset(error.to_string())
}

//! Burn model and output adapters for `NPClassifier` retraining.

use burn::backend::{NdArray, ndarray::NdArrayDevice};
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::Transaction;
use burn::tensor::activation::{log_sigmoid, relu, sigmoid};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::quantization::{QuantScheme, QuantStore};
use burn::tensor::{DType, TensorData};
use burn::train::metric::{Adaptor, ItemLazy, LossInput};
use burn::train::{InferenceStep, TrainOutput, TrainStep};

use npclassifier_core::{FINGERPRINT_INPUT_WIDTH, ModelHead};

use crate::data::NpClassifierBatch;

const PATHWAY_WIDTH: usize = ModelHead::Pathway.output_width();
const SUPERCLASS_WIDTH: usize = ModelHead::Superclass.output_width();
const CLASS_WIDTH: usize = ModelHead::Class.output_width();
const MINI_HIDDEN_1: usize = 1536;
const MINI_HIDDEN_2: usize = 768;
const MINI_HIDDEN_3: usize = 576;
const MINI_HIDDEN_4: usize = 288;

/// Draft-faithful architecture for retraining `NPClassifier`.
#[derive(Config, Debug)]
pub struct StudentModelConfig {
    /// Width of the first hidden layer in each head-specific tower.
    #[config(default = 6144)]
    pub hidden_1: usize,
    /// Width of the second hidden layer in each head-specific tower.
    #[config(default = 3072)]
    pub hidden_2: usize,
    /// Width of the third hidden layer in each head-specific tower.
    #[config(default = 2304)]
    pub hidden_3: usize,
    /// Width of the fourth hidden layer in each head-specific tower.
    #[config(default = 1152)]
    pub hidden_4: usize,
    /// Dropout probability applied before each output layer.
    #[config(default = 0.1)]
    pub dropout: f64,
    /// `BatchNorm` epsilon. Keras defaults to `0.001`.
    #[config(default = 1e-3)]
    pub batch_norm_epsilon: f64,
    /// `BatchNorm` update fraction. `0.01` matches Keras momentum `0.99`.
    #[config(default = 0.01)]
    pub batch_norm_momentum: f64,
    /// Whether the first hidden block is duplicated per head or shared once.
    #[config(default = false)]
    pub share_first_layer: bool,
}

/// Burn output item used for metrics and history logging.
pub struct NpClassifierOutput<B: Backend> {
    /// Total training or validation loss.
    pub loss: Tensor<B, 1>,
    /// Pathway probabilities after sigmoid.
    pub pathway_probabilities: Tensor<B, 2>,
    /// Superclass probabilities after sigmoid.
    pub superclass_probabilities: Tensor<B, 2>,
    /// Class probabilities after sigmoid.
    pub class_probabilities: Tensor<B, 2>,
    /// Pathway hard labels.
    pub pathway_targets: Tensor<B, 2, Int>,
    /// Superclass hard labels.
    pub superclass_targets: Tensor<B, 2, Int>,
    /// Class hard labels.
    pub class_targets: Tensor<B, 2, Int>,
}

struct DeviceBatch<B: Backend> {
    inputs: Tensor<B, 2>,
    pathway_targets: Tensor<B, 2, Int>,
    superclass_targets: Tensor<B, 2, Int>,
    class_targets: Tensor<B, 2, Int>,
    pathway_teacher: Option<Tensor<B, 2>>,
    superclass_teacher: Option<Tensor<B, 2>>,
    class_teacher: Option<Tensor<B, 2>>,
}

#[derive(Debug)]
pub struct ExportedHead {
    pub head: ModelHead,
    pub layers: Vec<ExportedHeadLayer>,
}

#[derive(Debug)]
pub enum ExportedHeadLayer {
    Concat,
    Dense(ExportedDenseLayer),
    BatchNorm(ExportedBatchNormLayer),
    Dropout,
}

#[derive(Debug)]
pub struct ExportedDenseLayer {
    pub name: String,
    pub activation: &'static str,
    pub bias: Vec<f32>,
    pub kernel: ExportedKernel,
}

#[derive(Debug)]
pub enum ExportedKernel {
    F32 {
        values: Vec<f32>,
        input: usize,
        output: usize,
    },
    Q4Block {
        packed_values: Vec<u8>,
        scales: Vec<f32>,
        input: usize,
        output: usize,
    },
}

#[derive(Debug)]
pub struct ExportedBatchNormLayer {
    pub name: String,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub mean: Vec<f32>,
    pub variance: Vec<f32>,
    pub epsilon: f32,
}

impl<B: Backend> NpClassifierOutput<B> {
    fn new(
        loss: Tensor<B, 1>,
        pathway_probabilities: Tensor<B, 2>,
        superclass_probabilities: Tensor<B, 2>,
        class_probabilities: Tensor<B, 2>,
        pathway_targets: Tensor<B, 2, Int>,
        superclass_targets: Tensor<B, 2, Int>,
        class_targets: Tensor<B, 2, Int>,
    ) -> Self {
        Self {
            loss,
            pathway_probabilities,
            superclass_probabilities,
            class_probabilities,
            pathway_targets,
            superclass_targets,
            class_targets,
        }
    }
}

impl<B: Backend> ItemLazy for NpClassifierOutput<B> {
    type ItemSync = NpClassifierOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [
            loss,
            pathway_probabilities,
            superclass_probabilities,
            class_probabilities,
            pathway_targets,
            superclass_targets,
            class_targets,
        ] = Transaction::default()
            .register(self.loss)
            .register(self.pathway_probabilities)
            .register(self.superclass_probabilities)
            .register(self.class_probabilities)
            .register(self.pathway_targets)
            .register(self.superclass_targets)
            .register(self.class_targets)
            .execute()
            .try_into()
            .expect("correct number of synchronized tensors");

        let device = &NdArrayDevice::default();
        NpClassifierOutput::new(
            Tensor::from_data(loss, device),
            Tensor::from_data(pathway_probabilities, device),
            Tensor::from_data(superclass_probabilities, device),
            Tensor::from_data(class_probabilities, device),
            Tensor::from_data(pathway_targets, device),
            Tensor::from_data(superclass_targets, device),
            Tensor::from_data(class_targets, device),
        )
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for NpClassifierOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

/// One independent multilabel tower for a single `NPClassifier` head.
#[derive(Module, Debug)]
struct StemLayer<B: Backend> {
    dense_1: Linear<B>,
    batch_norm_1: BatchNorm<B>,
}

impl<B: Backend> StemLayer<B> {
    fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.dense_1.forward(inputs));
        self.batch_norm_1.forward(x)
    }

    fn quantize_compatible_linear_weights(self, scheme: &QuantScheme) -> Self {
        Self {
            dense_1: quantize_linear_weight(self.dense_1, scheme),
            batch_norm_1: self.batch_norm_1,
        }
    }

    fn export(&self) -> Result<Vec<ExportedHeadLayer>, String> {
        Ok(vec![
            ExportedHeadLayer::Concat,
            ExportedHeadLayer::Dense(export_linear_layer("dense_1", "relu", &self.dense_1)?),
            ExportedHeadLayer::BatchNorm(export_batch_norm_layer(
                "batch_norm_1",
                &self.batch_norm_1,
            )),
        ])
    }
}

/// One multilabel head tower after the optional shared first layer.
#[derive(Module, Debug)]
struct HeadTower<B: Backend> {
    dense_2: Linear<B>,
    batch_norm_2: BatchNorm<B>,
    dense_3: Linear<B>,
    batch_norm_3: BatchNorm<B>,
    dense_4: Linear<B>,
    dropout: Dropout,
    output: Linear<B>,
}

impl<B: Backend> HeadTower<B> {
    fn forward_logits(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.dense_2.forward(inputs));
        let x = self.batch_norm_2.forward(x);
        let x = relu(self.dense_3.forward(x));
        let x = self.batch_norm_3.forward(x);
        let x = relu(self.dense_4.forward(x));
        let x = self.dropout.forward(x);

        self.output.forward(x)
    }

    fn quantize_compatible_linear_weights(self, scheme: &QuantScheme) -> Self {
        let Self {
            dense_2,
            batch_norm_2,
            dense_3,
            batch_norm_3,
            dense_4,
            dropout,
            output,
        } = self;

        Self {
            dense_2: quantize_linear_weight(dense_2, scheme),
            batch_norm_2,
            dense_3: quantize_linear_weight(dense_3, scheme),
            batch_norm_3,
            dense_4: quantize_linear_weight(dense_4, scheme),
            dropout,
            output: quantize_linear_weight_if_compatible(output, scheme),
        }
    }

    fn export(&self, head: ModelHead) -> Result<ExportedHead, String> {
        Ok(ExportedHead {
            head,
            layers: vec![
                ExportedHeadLayer::Dense(export_linear_layer("dense_2", "relu", &self.dense_2)?),
                ExportedHeadLayer::BatchNorm(export_batch_norm_layer(
                    "batch_norm_2",
                    &self.batch_norm_2,
                )),
                ExportedHeadLayer::Dense(export_linear_layer("dense_3", "relu", &self.dense_3)?),
                ExportedHeadLayer::BatchNorm(export_batch_norm_layer(
                    "batch_norm_3",
                    &self.batch_norm_3,
                )),
                ExportedHeadLayer::Dense(export_linear_layer("dense_4", "relu", &self.dense_4)?),
                ExportedHeadLayer::Dropout,
                ExportedHeadLayer::Dense(export_linear_layer("output", "sigmoid", &self.output)?),
            ],
        })
    }
}

/// Draft-faithful multilabel model with three independent output towers.
#[derive(Module, Debug)]
pub struct StudentModel<B: Backend> {
    shared_stem: Option<StemLayer<B>>,
    pathway_stem: Option<StemLayer<B>>,
    superclass_stem: Option<StemLayer<B>>,
    class_stem: Option<StemLayer<B>>,
    pathway: HeadTower<B>,
    superclass: HeadTower<B>,
    class: HeadTower<B>,
    hard_loss: BinaryCrossEntropyLoss<B>,
    hard_label_weight: f32,
    teacher_weight: f32,
}

impl StudentModelConfig {
    /// Draft-faithful baseline architecture recovered from the original
    /// `NPClassifier` stack.
    #[must_use]
    pub fn baseline() -> Self {
        Self::new()
    }

    /// Smaller variant that shares the first hidden block across all heads.
    #[must_use]
    pub fn mini_shared() -> Self {
        Self::new()
            .with_hidden_1(MINI_HIDDEN_1)
            .with_hidden_2(MINI_HIDDEN_2)
            .with_hidden_3(MINI_HIDDEN_3)
            .with_hidden_4(MINI_HIDDEN_4)
            .with_share_first_layer(true)
    }

    /// Initializes the model with the configured architecture.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        hard_label_weight: f32,
        teacher_weight: f32,
    ) -> StudentModel<B> {
        let hard_loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(device);

        StudentModel {
            shared_stem: self.share_first_layer.then(|| self.init_stem(device)),
            pathway_stem: (!self.share_first_layer).then(|| self.init_stem(device)),
            superclass_stem: (!self.share_first_layer).then(|| self.init_stem(device)),
            class_stem: (!self.share_first_layer).then(|| self.init_stem(device)),
            pathway: self.init_tower(device, PATHWAY_WIDTH),
            superclass: self.init_tower(device, SUPERCLASS_WIDTH),
            class: self.init_tower(device, CLASS_WIDTH),
            hard_loss,
            hard_label_weight,
            teacher_weight,
        }
    }

    fn init_stem<B: Backend>(&self, device: &B::Device) -> StemLayer<B> {
        StemLayer {
            dense_1: LinearConfig::new(FINGERPRINT_INPUT_WIDTH, self.hidden_1).init(device),
            batch_norm_1: BatchNormConfig::new(self.hidden_1)
                .with_epsilon(self.batch_norm_epsilon)
                .with_momentum(self.batch_norm_momentum)
                .init(device),
        }
    }

    fn init_tower<B: Backend>(&self, device: &B::Device, output_width: usize) -> HeadTower<B> {
        let batch_norm = |features| {
            BatchNormConfig::new(features)
                .with_epsilon(self.batch_norm_epsilon)
                .with_momentum(self.batch_norm_momentum)
                .init(device)
        };

        HeadTower {
            dense_2: LinearConfig::new(self.hidden_1, self.hidden_2).init(device),
            batch_norm_2: batch_norm(self.hidden_2),
            dense_3: LinearConfig::new(self.hidden_2, self.hidden_3).init(device),
            batch_norm_3: batch_norm(self.hidden_3),
            dense_4: LinearConfig::new(self.hidden_3, self.hidden_4).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            output: LinearConfig::new(self.hidden_4, output_width).init(device),
        }
    }
}

impl<B: Backend> StudentModel<B> {
    /// Quantize only the compatible `Linear` weight matrices, leaving biases and
    /// normalization parameters in float.
    #[must_use]
    pub fn quantize_compatible_linear_weights(self, scheme: &QuantScheme) -> Self {
        Self {
            shared_stem: self
                .shared_stem
                .map(|stem| stem.quantize_compatible_linear_weights(scheme)),
            pathway_stem: self
                .pathway_stem
                .map(|stem| stem.quantize_compatible_linear_weights(scheme)),
            superclass_stem: self
                .superclass_stem
                .map(|stem| stem.quantize_compatible_linear_weights(scheme)),
            class_stem: self
                .class_stem
                .map(|stem| stem.quantize_compatible_linear_weights(scheme)),
            pathway: self.pathway.quantize_compatible_linear_weights(scheme),
            superclass: self.superclass.quantize_compatible_linear_weights(scheme),
            class: self.class.quantize_compatible_linear_weights(scheme),
            hard_loss: self.hard_loss,
            hard_label_weight: self.hard_label_weight,
            teacher_weight: self.teacher_weight,
        }
    }

    /// Exports the three prediction heads into the packed inference schema.
    ///
    /// # Errors
    ///
    /// Returns an error if the architecture wiring is inconsistent or if any
    /// tensor cannot be decoded into the packed schema.
    pub fn export_heads(&self) -> Result<[ExportedHead; 3], String> {
        let stem_layers = match (
            self.shared_stem.as_ref(),
            self.pathway_stem.as_ref(),
            self.superclass_stem.as_ref(),
            self.class_stem.as_ref(),
        ) {
            (Some(shared), None, None, None) => (None, Some(shared.export()?)),
            (None, Some(pathway), Some(superclass), Some(class)) => (
                Some([pathway.export()?, superclass.export()?, class.export()?]),
                None,
            ),
            _ => {
                return Err(
                    "student model has inconsistent stem wiring; cannot export packed heads"
                        .to_owned(),
                );
            }
        };
        let (per_head_stem, shared_stem) = stem_layers;

        let mut pathway = self.pathway.export(ModelHead::Pathway)?;
        let mut superclass = self.superclass.export(ModelHead::Superclass)?;
        let mut class = self.class.export(ModelHead::Class)?;

        if let Some(shared_layers) = shared_stem {
            // Shared-first-layer models write the shared stem as a separate
            // archive. Head archives start at `dense_2`.
            let _ = shared_layers;
        } else if let Some([pathway_stem, superclass_stem, class_stem]) = per_head_stem {
            pathway.layers.splice(0..0, pathway_stem);
            superclass.layers.splice(0..0, superclass_stem);
            class.layers.splice(0..0, class_stem);
        }

        Ok([pathway, superclass, class])
    }

    /// Exports the shared first layer for mini/shared-stem models.
    ///
    /// # Errors
    ///
    /// Returns an error if the architecture combines shared and per-head stems
    /// in a layout that cannot be represented by the packed schema.
    pub fn export_shared_stem(&self) -> Result<Option<Vec<ExportedHeadLayer>>, String> {
        match (
            self.shared_stem.as_ref(),
            self.pathway_stem.is_some(),
            self.superclass_stem.is_some(),
            self.class_stem.is_some(),
        ) {
            (Some(shared), false, false, false) => Ok(Some(shared.export()?)),
            (None, _, _, _) => Ok(None),
            _ => Err(
                "student model has inconsistent shared-stem wiring; cannot export packed model"
                    .to_owned(),
            ),
        }
    }

    fn device(&self) -> B::Device {
        self.devices()
            .into_iter()
            .next()
            .expect("student model should have an attached device")
    }

    fn optional_float_tensor(
        values: Option<Vec<f32>>,
        shape: [usize; 2],
        device: &B::Device,
    ) -> Option<Tensor<B, 2>> {
        values.map(|values| Tensor::<B, 2>::from_data(TensorData::new(values, shape), device))
    }

    fn to_device_batch(&self, batch: NpClassifierBatch) -> DeviceBatch<B> {
        let batch_len = batch.len();
        let device = self.device();

        DeviceBatch {
            inputs: Tensor::<B, 2>::from_data(
                TensorData::new(batch.inputs, [batch_len, FINGERPRINT_INPUT_WIDTH]),
                &device,
            ),
            pathway_targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(batch.pathway_targets, [batch_len, PATHWAY_WIDTH]),
                &device,
            ),
            superclass_targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(batch.superclass_targets, [batch_len, SUPERCLASS_WIDTH]),
                &device,
            ),
            class_targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(batch.class_targets, [batch_len, CLASS_WIDTH]),
                &device,
            ),
            pathway_teacher: Self::optional_float_tensor(
                batch.pathway_teacher,
                [batch_len, PATHWAY_WIDTH],
                &device,
            ),
            superclass_teacher: Self::optional_float_tensor(
                batch.superclass_teacher,
                [batch_len, SUPERCLASS_WIDTH],
                &device,
            ),
            class_teacher: Self::optional_float_tensor(
                batch.class_teacher,
                [batch_len, CLASS_WIDTH],
                &device,
            ),
        }
    }

    fn forward_logits(&self, inputs: Tensor<B, 2>) -> StudentLogits<B> {
        let pathway_inputs = if let Some(shared_stem) = self.shared_stem.as_ref() {
            let shared = shared_stem.forward(inputs);
            (shared.clone(), shared.clone(), shared)
        } else {
            let pathway_stem = self
                .pathway_stem
                .as_ref()
                .expect("independent architecture should have a pathway stem");
            let superclass_stem = self
                .superclass_stem
                .as_ref()
                .expect("independent architecture should have a superclass stem");
            let class_stem = self
                .class_stem
                .as_ref()
                .expect("independent architecture should have a class stem");
            (
                pathway_stem.forward(inputs.clone()),
                superclass_stem.forward(inputs.clone()),
                class_stem.forward(inputs),
            )
        };

        StudentLogits {
            pathway: self.pathway.forward_logits(pathway_inputs.0),
            superclass: self.superclass.forward_logits(pathway_inputs.1),
            class: self.class.forward_logits(pathway_inputs.2),
        }
    }

    fn forward_batch(&self, batch: DeviceBatch<B>) -> NpClassifierOutput<B> {
        let logits = self.forward_logits(batch.inputs);

        let pathway_hard = self
            .hard_loss
            .forward(logits.pathway.clone(), batch.pathway_targets.clone());
        let superclass_hard = self
            .hard_loss
            .forward(logits.superclass.clone(), batch.superclass_targets.clone());
        let class_hard = self
            .hard_loss
            .forward(logits.class.clone(), batch.class_targets.clone());
        let hard_loss = (pathway_hard + superclass_hard + class_hard).div_scalar(3.0);

        let loss = if self.teacher_weight > 0.0 {
            if let (Some(pathway_teacher), Some(superclass_teacher), Some(class_teacher)) = (
                batch.pathway_teacher.as_ref(),
                batch.superclass_teacher.as_ref(),
                batch.class_teacher.as_ref(),
            ) {
                let pathway_soft = soft_target_binary_cross_entropy(
                    logits.pathway.clone(),
                    pathway_teacher.clone(),
                );
                let superclass_soft = soft_target_binary_cross_entropy(
                    logits.superclass.clone(),
                    superclass_teacher.clone(),
                );
                let class_soft =
                    soft_target_binary_cross_entropy(logits.class.clone(), class_teacher.clone());
                let soft_loss = (pathway_soft + superclass_soft + class_soft).div_scalar(3.0);
                hard_loss.mul_scalar(self.hard_label_weight)
                    + soft_loss.mul_scalar(self.teacher_weight)
            } else {
                hard_loss.mul_scalar(self.hard_label_weight)
            }
        } else {
            hard_loss.mul_scalar(self.hard_label_weight)
        };

        NpClassifierOutput::new(
            loss,
            sigmoid(logits.pathway),
            sigmoid(logits.superclass),
            sigmoid(logits.class),
            batch.pathway_targets,
            batch.superclass_targets,
            batch.class_targets,
        )
    }
}

fn export_linear_layer<B: Backend>(
    name: &str,
    activation: &'static str,
    linear: &Linear<B>,
) -> Result<ExportedDenseLayer, String> {
    let weight = linear.weight.val();
    let [input, output] = weight.dims();
    let weight_data = weight.into_data();
    let bias = linear
        .bias
        .as_ref()
        .map(|bias| tensor_to_f32_vec(&bias.val().into_data()))
        .transpose()?
        .unwrap_or_else(|| vec![0.0; output]);
    let dtype = weight_data.dtype;
    let kernel = match dtype {
        DType::QFloat(scheme) => export_q4_kernel(name, &weight_data, input, output, &scheme)?,
        _ => ExportedKernel::F32 {
            values: tensor_to_f32_vec(&weight_data)?,
            input,
            output,
        },
    };

    Ok(ExportedDenseLayer {
        name: name.to_owned(),
        activation,
        bias,
        kernel,
    })
}

#[allow(clippy::cast_possible_truncation)]
fn export_batch_norm_layer<B: Backend>(
    name: &str,
    batch_norm: &BatchNorm<B>,
) -> ExportedBatchNormLayer {
    ExportedBatchNormLayer {
        name: name.to_owned(),
        gamma: tensor_to_f32_vec(&batch_norm.gamma.val().into_data())
            .expect("batch norm gamma should decode"),
        beta: tensor_to_f32_vec(&batch_norm.beta.val().into_data())
            .expect("batch norm beta should decode"),
        mean: tensor_to_f32_vec(&batch_norm.running_mean.value().into_data())
            .expect("batch norm mean should decode"),
        variance: tensor_to_f32_vec(&batch_norm.running_var.value().into_data())
            .expect("batch norm variance should decode"),
        epsilon: batch_norm.epsilon as f32,
    }
}

fn tensor_to_f32_vec(data: &TensorData) -> Result<Vec<f32>, String> {
    if matches!(data.dtype, DType::QFloat(_)) {
        return Err("expected a float tensor, found a quantized tensor".to_owned());
    }

    Ok(data.iter::<f32>().collect())
}

fn export_q4_kernel(
    name: &str,
    data: &TensorData,
    input: usize,
    output: usize,
    scheme: &burn::tensor::quantization::QuantScheme,
) -> Result<ExportedKernel, String> {
    let burn::tensor::quantization::QuantScheme {
        level,
        mode,
        value,
        store,
        param,
    } = scheme;

    if !matches!(mode, burn::tensor::quantization::QuantMode::Symmetric) {
        return Err(format!(
            "linear layer {name} uses unsupported q4 mode {mode:?}"
        ));
    }
    if !matches!(value, burn::tensor::quantization::QuantValue::Q4S) {
        return Err(format!(
            "linear layer {name} uses unsupported q4 value {value:?}"
        ));
    }
    if !matches!(store, burn::tensor::quantization::QuantStore::PackedU32(0)) {
        return Err(format!(
            "linear layer {name} uses unsupported q4 store {store:?}"
        ));
    }
    if !matches!(param, burn::tensor::quantization::QuantParam::F32) {
        return Err(format!(
            "linear layer {name} uses unsupported q4 params {param:?}"
        ));
    }

    let bytes = data.bytes.to_vec();
    let num_elements = input * output;
    let scale_count = match level {
        burn::tensor::quantization::QuantLevel::Tensor => 1,
        burn::tensor::quantization::QuantLevel::Block(block_size) => {
            let block_elements = block_size.num_elements();
            if block_elements == 0 || !num_elements.is_multiple_of(block_elements) {
                return Err(format!(
                    "linear layer {name} has invalid block size {block_size:?} for {input}x{output}"
                ));
            }
            num_elements / block_elements
        }
    };
    let scale_bytes_len = scale_count * core::mem::size_of::<f32>();
    if bytes.len() < scale_bytes_len {
        return Err(format!(
            "linear layer {name} q4 payload is too short for {scale_count} scales"
        ));
    }

    let split_at = bytes.len() - scale_bytes_len;
    let (packed_values, scale_bytes) = bytes.split_at(split_at);
    if scale_bytes.len() % 4 != 0 {
        return Err(format!(
            "linear layer {name} q4 scale payload has invalid byte length {}",
            scale_bytes.len()
        ));
    }
    let scales = scale_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();

    Ok(ExportedKernel::Q4Block {
        packed_values: packed_values.to_vec(),
        scales,
        input,
        output,
    })
}

impl<B: AutodiffBackend> TrainStep for StudentModel<B> {
    type Input = NpClassifierBatch;
    type Output = NpClassifierOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let item = self.forward_batch(self.to_device_batch(batch));
        TrainOutput::new(self, item.loss.clone().backward(), item)
    }
}

impl<B: Backend> InferenceStep for StudentModel<B> {
    type Input = NpClassifierBatch;
    type Output = NpClassifierOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_batch(self.to_device_batch(batch))
    }
}

struct StudentLogits<B: Backend> {
    pathway: Tensor<B, 2>,
    superclass: Tensor<B, 2>,
    class: Tensor<B, 2>,
}

fn soft_target_binary_cross_entropy<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let positive = targets.clone() * log_sigmoid(logits.clone());
    let negative = (targets.ones_like() - targets) * log_sigmoid(logits.neg());
    -(positive + negative).mean()
}

fn quantize_linear_weight<B: Backend>(linear: Linear<B>, scheme: &QuantScheme) -> Linear<B> {
    Linear {
        weight: quantize_param_dynamic(linear.weight, scheme),
        bias: linear.bias,
    }
}

fn quantize_linear_weight_if_compatible<B: Backend>(
    linear: Linear<B>,
    scheme: &QuantScheme,
) -> Linear<B> {
    let weight_dims = linear.weight.val().dims();
    if last_dim_is_pack_compatible(weight_dims[1], scheme) {
        quantize_linear_weight(linear, scheme)
    } else {
        linear
    }
}

fn quantize_param_dynamic<B: Backend, const D: usize>(
    param: Param<Tensor<B, D>>,
    scheme: &QuantScheme,
) -> Param<Tensor<B, D>> {
    param.map(|tensor| tensor.quantize_dynamic(scheme))
}

fn last_dim_is_pack_compatible(last_dim: usize, scheme: &QuantScheme) -> bool {
    match scheme.store {
        QuantStore::PackedU32(_) | QuantStore::PackedNative(_) => {
            last_dim.is_multiple_of(scheme.num_quants())
        }
        QuantStore::Native => true,
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    use super::*;

    #[test]
    fn default_config_matches_recovered_npclassifier_architecture() {
        let config = StudentModelConfig::baseline();

        assert_eq!(config.hidden_1, 6144);
        assert_eq!(config.hidden_2, 3072);
        assert_eq!(config.hidden_3, 2304);
        assert_eq!(config.hidden_4, 1152);
        assert!((config.dropout - 0.1).abs() < f64::EPSILON);
        assert!((config.batch_norm_epsilon - 1e-3).abs() < f64::EPSILON);
        assert!((config.batch_norm_momentum - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn mini_shared_config_enables_first_layer_sharing() {
        let config = StudentModelConfig::mini_shared();

        assert_eq!(config.hidden_1, MINI_HIDDEN_1);
        assert_eq!(config.hidden_2, MINI_HIDDEN_2);
        assert_eq!(config.hidden_3, MINI_HIDDEN_3);
        assert_eq!(config.hidden_4, MINI_HIDDEN_4);
        assert!(config.share_first_layer);
    }

    #[test]
    fn forward_logits_match_head_output_shapes() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = tiny_test_config().init::<NdArray>(&device, 1.0, 0.0);
        let inputs = Tensor::<NdArray, 2>::zeros([2, FINGERPRINT_INPUT_WIDTH], &device);
        let logits = model.forward_logits(inputs);

        assert_eq!(logits.pathway.dims(), [2, PATHWAY_WIDTH]);
        assert_eq!(logits.superclass.dims(), [2, SUPERCLASS_WIDTH]);
        assert_eq!(logits.class.dims(), [2, CLASS_WIDTH]);
    }

    #[test]
    fn parameter_count_is_close_to_the_original_three_head_total() {
        assert_eq!(
            parameter_count(&StudentModelConfig::baseline()),
            200_129_666
        );
    }

    #[test]
    fn mini_shared_architecture_is_far_smaller_than_baseline() {
        let mini_shared = parameter_count(&StudentModelConfig::mini_shared());
        let baseline = parameter_count(&StudentModelConfig::baseline());

        assert!(mini_shared < baseline);
        assert!(mini_shared < 20_000_000);
    }

    #[test]
    fn soft_target_loss_accepts_dense_probabilities() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let logits = Tensor::<NdArray, 2>::from_data(
            TensorData::from([[0.0_f32, 1.0_f32], [2.0_f32, -1.0_f32]]),
            &device,
        );
        let targets = Tensor::<NdArray, 2>::from_data(
            TensorData::from([[0.1_f32, 0.9_f32], [0.8_f32, 0.2_f32]]),
            &device,
        );
        let loss = soft_target_binary_cross_entropy(logits, targets);

        assert_eq!(loss.dims(), [1]);
    }

    #[test]
    fn q4_pack_compatibility_matches_npclassifier_head_widths() {
        let scheme = QuantScheme {
            level: burn::tensor::quantization::QuantLevel::Block(
                burn::tensor::quantization::BlockSize::new([32]),
            ),
            mode: burn::tensor::quantization::QuantMode::Symmetric,
            value: burn::tensor::quantization::QuantValue::Q4S,
            store: QuantStore::PackedU32(0),
            param: burn::tensor::quantization::QuantParam::F32,
        };

        assert!(last_dim_is_pack_compatible(1152, &scheme));
        assert!(!last_dim_is_pack_compatible(PATHWAY_WIDTH, &scheme));
        assert!(!last_dim_is_pack_compatible(SUPERCLASS_WIDTH, &scheme));
        assert!(!last_dim_is_pack_compatible(CLASS_WIDTH, &scheme));
    }

    fn tiny_test_config() -> StudentModelConfig {
        StudentModelConfig::mini_shared()
            .with_hidden_1(8)
            .with_hidden_2(8)
            .with_hidden_3(8)
            .with_hidden_4(8)
    }

    const fn parameter_count(config: &StudentModelConfig) -> usize {
        let stem_count = if config.share_first_layer { 1 } else { 3 };
        stem_count * stem_parameter_count(config.hidden_1)
            + head_parameter_count(config, PATHWAY_WIDTH)
            + head_parameter_count(config, SUPERCLASS_WIDTH)
            + head_parameter_count(config, CLASS_WIDTH)
    }

    const fn stem_parameter_count(hidden_1: usize) -> usize {
        FINGERPRINT_INPUT_WIDTH * hidden_1 + hidden_1 + 4 * hidden_1
    }

    const fn head_parameter_count(config: &StudentModelConfig, output_width: usize) -> usize {
        linear_parameter_count(config.hidden_1, config.hidden_2)
            + 4 * config.hidden_2
            + linear_parameter_count(config.hidden_2, config.hidden_3)
            + 4 * config.hidden_3
            + linear_parameter_count(config.hidden_3, config.hidden_4)
            + linear_parameter_count(config.hidden_4, output_width)
    }

    const fn linear_parameter_count(input: usize, output: usize) -> usize {
        input * output + output
    }
}

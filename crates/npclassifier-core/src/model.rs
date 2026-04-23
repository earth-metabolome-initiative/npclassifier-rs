use serde::{Deserialize, Serialize};

/// One dense layer in the recovered classifier tower stack.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DenseLayerSpec {
    /// Input width for the layer.
    pub input: usize,
    /// Output width for the layer.
    pub output: usize,
    /// Activation function applied after the affine transform.
    pub activation: &'static str,
    /// Whether the layer is followed by batch normalization.
    pub batch_norm: bool,
    /// Optional dropout rate applied after activation.
    pub dropout: Option<f32>,
}

/// Dense hidden stack replicated across the three classifier heads.
pub const BACKBONE_LAYERS: [DenseLayerSpec; 4] = [
    DenseLayerSpec {
        input: 6144,
        output: 6144,
        activation: "relu",
        batch_norm: true,
        dropout: None,
    },
    DenseLayerSpec {
        input: 6144,
        output: 3072,
        activation: "relu",
        batch_norm: true,
        dropout: None,
    },
    DenseLayerSpec {
        input: 3072,
        output: 2304,
        activation: "relu",
        batch_norm: true,
        dropout: None,
    },
    DenseLayerSpec {
        input: 2304,
        output: 1152,
        activation: "relu",
        batch_norm: false,
        dropout: Some(0.1),
    },
];

/// One of the three recovered `NPClassifier` output heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelHead {
    /// Seven-way pathway head.
    Pathway,
    /// Seventy-six-way superclass head.
    Superclass,
    /// Six-hundred-eighty-seven-way class head.
    Class,
}

impl ModelHead {
    /// Returns the expected output width for the head.
    #[must_use]
    pub const fn output_width(self) -> usize {
        match self {
            Self::Pathway => 7,
            Self::Superclass => 76,
            Self::Class => 687,
        }
    }

    /// Returns the score threshold used by the draft pipeline.
    #[must_use]
    pub const fn threshold(self) -> f32 {
        match self {
            Self::Pathway => 0.5,
            Self::Superclass => 0.3,
            Self::Class => 0.1,
        }
    }

    /// Returns the lowercase archive name used on disk.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pathway => "pathway",
            Self::Superclass => "superclass",
            Self::Class => "class",
        }
    }
}

impl core::fmt::Display for ModelHead {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Static description of one output head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelHeadSpec {
    /// Head identifier.
    pub head: ModelHead,
    /// Expected output width.
    pub output_width: usize,
}

/// Static metadata for all recovered classifier heads.
pub const MODEL_HEADS: [ModelHeadSpec; 3] = [
    ModelHeadSpec {
        head: ModelHead::Pathway,
        output_width: ModelHead::Pathway.output_width(),
    },
    ModelHeadSpec {
        head: ModelHead::Superclass,
        output_width: ModelHead::Superclass.output_width(),
    },
    ModelHeadSpec {
        head: ModelHead::Class,
        output_width: ModelHead::Class.output_width(),
    },
];

use serde::{Deserialize, Serialize};

use crate::{
    ClassificationThresholds, FingerprintGenerator, InferenceEngine, Ontology, PredictionLabels,
    classify_scores,
};

/// Browser-facing model bundles available in the web app.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebModelVariant {
    /// Smaller shared-stem browser model.
    #[default]
    MiniShared,
    /// Larger faithful browser model.
    Full,
}

impl WebModelVariant {
    /// Stable filesystem / JSON slug for the model bundle.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::MiniShared => "mini-shared",
            Self::Full => "full",
        }
    }

    /// Short UI label used by the model toggle.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::MiniShared => "Mini",
            Self::Full => "Faithful",
        }
    }

    /// Human-readable model name for progress labels.
    #[must_use]
    pub const fn loading_name(self) -> &'static str {
        match self {
            Self::MiniShared => "mini model",
            Self::Full => "faithful model",
        }
    }

    /// Whether this browser bundle includes a shared stem archive.
    #[must_use]
    pub const fn has_shared_archive(self) -> bool {
        matches!(self, Self::MiniShared)
    }
}

/// One named score entry in ontology index order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebScoredLabel {
    /// Ontology index for the score.
    pub index: usize,
    /// Human-readable ontology label.
    pub name: String,
    /// Raw sigmoid score.
    pub score: f32,
}

/// One browser-facing classification result row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebBatchEntry {
    /// Input SMILES string after trimming.
    pub smiles: String,
    /// Error text when parsing, fingerprinting, or inference failed.
    pub error: Option<String>,
    /// Thresholded and reconciled labels.
    pub labels: PredictionLabels,
    /// Full named pathway score vector.
    pub pathway_scores: Vec<WebScoredLabel>,
    /// Full named superclass score vector.
    pub superclass_scores: Vec<WebScoredLabel>,
    /// Full named class score vector.
    pub class_scores: Vec<WebScoredLabel>,
}

/// Messages sent from the browser UI to the dedicated classifier worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebWorkerRequest {
    /// Cancels any in-flight request older than the given token.
    Cancel {
        /// Monotonic request token.
        token: u64,
    },
    /// Classifies one batch of SMILES lines.
    Classify {
        /// Monotonic request token.
        token: u64,
        /// Selected browser model bundle.
        model: WebModelVariant,
        /// One trimmed non-empty SMILES line per entry.
        lines: Vec<String>,
    },
}

/// Messages sent from the dedicated classifier worker back to the browser UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WebWorkerResponse {
    /// Announces that the worker has installed its message loop and can accept requests.
    Ready,
    /// Reports worker-side load or classification progress.
    Progress {
        /// Request token this progress update belongs to.
        token: u64,
        /// Human-readable stage label.
        label: String,
        /// Completed units for the current stage.
        completed: usize,
        /// Total units for the current stage.
        total: usize,
    },
    /// Returns the completed batch entries.
    Complete {
        /// Request token this completion belongs to.
        token: u64,
        /// One result entry per submitted SMILES line.
        entries: Vec<WebBatchEntry>,
    },
    /// Returns one unrecoverable worker error for the request.
    Fatal {
        /// Request token this failure belongs to.
        token: u64,
        /// Human-readable error message.
        message: String,
    },
}

impl WebWorkerResponse {
    /// Returns the request token carried by the response.
    #[must_use]
    pub const fn token(&self) -> u64 {
        match self {
            Self::Ready => 0,
            Self::Progress { token, .. }
            | Self::Complete { token, .. }
            | Self::Fatal { token, .. } => *token,
        }
    }
}

/// Classifies one SMILES string into the browser-facing batch entry shape.
#[must_use]
pub fn classify_web_entry<G, M>(
    smiles: &str,
    ontology: &Ontology,
    generator: &G,
    model: &M,
) -> WebBatchEntry
where
    G: FingerprintGenerator,
    M: InferenceEngine,
{
    classify_web_entry_with_thresholds(
        smiles,
        ontology,
        generator,
        model,
        ClassificationThresholds::default(),
    )
}

/// Classifies one SMILES string into the browser-facing batch entry shape
/// using explicit decision thresholds.
#[must_use]
pub fn classify_web_entry_with_thresholds<G, M>(
    smiles: &str,
    ontology: &Ontology,
    generator: &G,
    model: &M,
    thresholds: ClassificationThresholds,
) -> WebBatchEntry
where
    G: FingerprintGenerator,
    M: InferenceEngine,
{
    let normalized = smiles.trim().to_owned();

    let prepared = match generator.generate(&normalized) {
        Ok(prepared) => prepared,
        Err(error) => return error_entry(normalized, error.to_string()),
    };

    let output = match model
        .predict(prepared.fingerprint())
        .and_then(|raw| classify_scores(raw, ontology, thresholds, prepared.is_glycoside()))
    {
        Ok(output) => output,
        Err(error) => return error_entry(normalized, error.to_string()),
    };

    let pathway_scores = collect_named_scores(
        output.raw.pathway.as_slice(),
        ontology.pathway_count(),
        |index| ontology.pathway_name(index),
    );
    let superclass_scores = collect_named_scores(
        output.raw.superclass.as_slice(),
        ontology.superclass_count(),
        |index| ontology.superclass_name(index),
    );
    let class_scores = collect_named_scores(
        output.raw.class.as_slice(),
        ontology.class_count(),
        |index| ontology.class_name(index),
    );

    WebBatchEntry {
        smiles: normalized,
        error: None,
        labels: PredictionLabels::from(&output),
        pathway_scores,
        superclass_scores,
        class_scores,
    }
}

fn error_entry(smiles: String, error: String) -> WebBatchEntry {
    WebBatchEntry {
        smiles,
        error: Some(error),
        labels: PredictionLabels::new(Vec::new(), Vec::new(), Vec::new(), None),
        pathway_scores: Vec::new(),
        superclass_scores: Vec::new(),
        class_scores: Vec::new(),
    }
}

fn collect_named_scores<'a>(
    values: &[f32],
    count: usize,
    mut name_at: impl FnMut(usize) -> Option<&'a str>,
) -> Vec<WebScoredLabel> {
    let mut scores = Vec::with_capacity(count);
    for (index, score) in values.iter().copied().enumerate().take(count) {
        if let Some(name) = name_at(index) {
            scores.push(WebScoredLabel {
                index,
                name: String::from(name),
                score,
            });
        }
    }
    scores.sort_by(|left, right| right.score.total_cmp(&left.score));
    scores
}

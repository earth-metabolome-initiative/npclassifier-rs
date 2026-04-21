use serde::{Deserialize, Serialize};

use crate::{
    error::NpClassifierError,
    fingerprint::{FingerprintGenerator, FingerprintInput},
    model::ModelHead,
    ontology::{EmbeddedOntology, Ontology},
    voting::{VoteInput, VoteOutcome, vote_classification},
};

/// Per-head decision thresholds recovered from the Python draft.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClassificationThresholds {
    /// Minimum sigmoid score required to keep a pathway label.
    pub pathway: f32,
    /// Minimum sigmoid score required to keep a superclass label.
    pub superclass: f32,
    /// Minimum sigmoid score required to keep a class label.
    pub class: f32,
}

impl ClassificationThresholds {
    /// Creates an explicit threshold triple.
    #[must_use]
    pub const fn new(pathway: f32, superclass: f32, class: f32) -> Self {
        Self {
            pathway,
            superclass,
            class,
        }
    }

    /// Returns the thresholds recovered from the original Python draft.
    #[must_use]
    pub const fn legacy_draft() -> Self {
        Self::new(
            ModelHead::Pathway.threshold(),
            ModelHead::Superclass.threshold(),
            ModelHead::Class.threshold(),
        )
    }
}

impl Default for ClassificationThresholds {
    fn default() -> Self {
        Self::legacy_draft()
    }
}

/// Raw sigmoid outputs from the three packed classifier heads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawPredictions {
    /// Pathway head scores in ontology index order.
    pub pathway: Vec<f32>,
    /// Superclass head scores in ontology index order.
    pub superclass: Vec<f32>,
    /// Class head scores in ontology index order.
    pub class: Vec<f32>,
}

impl RawPredictions {
    /// Validates that every score vector matches the expected head width.
    ///
    /// # Errors
    ///
    /// Returns [`NpClassifierError::InvalidPredictionWidth`] when any head has
    /// an unexpected output width.
    pub fn validate(&self) -> Result<(), NpClassifierError> {
        validate_head(ModelHead::Pathway, &self.pathway)?;
        validate_head(ModelHead::Superclass, &self.superclass)?;
        validate_head(ModelHead::Class, &self.class)?;
        Ok(())
    }
}

/// Final classifier output containing raw scores and ontology-aware labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationOutput {
    /// Raw per-head sigmoid outputs.
    pub raw: RawPredictions,
    /// Thresholded and reconciled labels after voting.
    pub voted: VoteOutcome,
}

/// Inference backend capable of scoring a prepared fingerprint.
pub trait InferenceEngine {
    /// Runs the recovered model heads for one prepared fingerprint input.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] when the backend cannot score the
    /// supplied fingerprint.
    fn predict(&self, fingerprint: &FingerprintInput) -> Result<RawPredictions, NpClassifierError>;
}

/// High-level pipeline combining fingerprint generation, inference, and voting.
pub struct ClassifierPipeline<G, M> {
    generator: G,
    model: M,
    ontology: Ontology,
    thresholds: ClassificationThresholds,
}

impl<G, M> ClassifierPipeline<G, M> {
    /// Builds a pipeline from explicit generator, model, and ontology parts.
    #[must_use]
    pub fn new(generator: G, model: M, ontology: Ontology) -> Self {
        Self {
            generator,
            model,
            ontology,
            thresholds: ClassificationThresholds::default(),
        }
    }

    /// Builds a pipeline using the embedded draft ontology.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if the embedded ontology cannot be
    /// decoded.
    pub fn with_embedded_ontology(generator: G, model: M) -> Result<Self, NpClassifierError> {
        Ok(Self::new(generator, model, EmbeddedOntology::load()?))
    }

    /// Overrides the default per-head decision thresholds.
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: ClassificationThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Returns the ontology used by this pipeline.
    #[must_use]
    pub fn ontology(&self) -> &Ontology {
        &self.ontology
    }
}

impl<G, M> ClassifierPipeline<G, M>
where
    G: FingerprintGenerator,
    M: InferenceEngine,
{
    /// Classifies one SMILES string using the configured generator and model.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if fingerprint generation, model
    /// inference, or ontology-based voting fails.
    pub fn classify_smiles(&self, smiles: &str) -> Result<ClassificationOutput, NpClassifierError> {
        let prepared = self.generator.generate(smiles)?;
        let raw = self.model.predict(prepared.fingerprint())?;
        classify_scores(
            raw,
            &self.ontology,
            self.thresholds,
            prepared.is_glycoside(),
        )
    }
}

/// Applies thresholds to raw scores and resolves a final ontology-consistent vote.
///
/// # Errors
///
/// Returns [`NpClassifierError::InvalidPredictionWidth`] when any head width
/// does not match the recovered model contract.
pub fn classify_scores(
    raw: RawPredictions,
    ontology: &Ontology,
    thresholds: ClassificationThresholds,
    is_glycoside: Option<bool>,
) -> Result<ClassificationOutput, NpClassifierError> {
    raw.validate()?;

    let pathway_hits = above_threshold(&raw.pathway, thresholds.pathway);
    let superclass_hits = above_threshold(&raw.superclass, thresholds.superclass);
    let class_hits = above_threshold(&raw.class, thresholds.class);

    let pathways_from_classes = flatten_unique(
        class_hits
            .iter()
            .flat_map(|index| ontology.class_pathways(*index).iter().copied()),
    );
    let pathways_from_superclasses = flatten_unique(
        superclass_hits
            .iter()
            .flat_map(|index| ontology.superclass_pathways(*index).iter().copied()),
    );

    let voted = vote_classification(
        VoteInput {
            pathways_above_threshold: &pathway_hits,
            classes_above_threshold: &class_hits,
            superclasses_above_threshold: &superclass_hits,
            class_scores: &raw.class,
            superclass_scores: &raw.superclass,
            pathways_from_classes: &pathways_from_classes,
            pathways_from_superclasses: &pathways_from_superclasses,
            is_glycoside,
        },
        ontology,
    );

    Ok(ClassificationOutput { raw, voted })
}

fn above_threshold(values: &[f32], threshold: f32) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(index, score)| (*score >= threshold).then_some(index))
        .collect()
}

fn flatten_unique(iter: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut values = iter.collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

fn validate_head(head: ModelHead, values: &[f32]) -> Result<(), NpClassifierError> {
    let expected = head.output_width();
    let actual = values.len();

    if actual == expected {
        Ok(())
    } else {
        Err(NpClassifierError::InvalidPredictionWidth {
            head,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::ontology::Ontology;

    use super::{ClassificationThresholds, RawPredictions, classify_scores};

    fn fixture_ontology() -> Ontology {
        let pathways = (0..7)
            .map(|index| format!(r#""Path {index}": {index}"#))
            .collect::<Vec<_>>()
            .join(", ");
        let superclasses = (0..77)
            .map(|index| format!(r#""Super {index}": {index}"#))
            .collect::<Vec<_>>()
            .join(", ");
        let classes = (0..687)
            .map(|index| format!(r#""Class {index}": {index}"#))
            .collect::<Vec<_>>()
            .join(", ");
        let json = format!(
            r#"{{
                "Pathway": {{{pathways}}},
                "Superclass": {{{superclasses}}},
                "Class": {{{classes}}},
                "Class_hierarchy": {{
                    "0": {{"Pathway": [0], "Superclass": [0]}}
                }},
                "Super_hierarchy": {{
                    "0": {{"Pathway": [0]}}
                }}
            }}"#,
        );

        Ontology::from_json_str(&json).expect("fixture ontology should parse")
    }

    #[test]
    fn score_thresholds_feed_the_voter() {
        let ontology = fixture_ontology();
        let mut pathway = vec![0.0; 7];
        let mut superclass = vec![0.0; 77];
        let mut class = vec![0.0; 687];
        pathway[0] = 0.7;
        superclass[0] = 0.8;
        class[0] = 0.9;

        let result = classify_scores(
            RawPredictions {
                pathway,
                superclass,
                class,
            },
            &ontology,
            ClassificationThresholds::default(),
            Some(false),
        )
        .expect("classification should succeed");

        assert_eq!(result.voted.pathways[0].name, "Path 0");
        assert_eq!(result.voted.superclasses[0].name, "Super 0");
        assert_eq!(result.voted.classes[0].name, "Class 0");
        assert_eq!(result.voted.is_glycoside, Some(false));
    }
}

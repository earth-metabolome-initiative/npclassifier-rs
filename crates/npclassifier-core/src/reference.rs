use serde::{Deserialize, Serialize};

use crate::classifier::ClassificationOutput;

/// Zenodo record id for the published `PubChem` reference snapshot.
pub const PUBCHEM_REFERENCE_RECORD_ID: u64 = 19_513_825;
/// DOI for the published `PubChem` reference snapshot.
pub const PUBCHEM_REFERENCE_DOI: &str = "10.5281/zenodo.19513825";
/// Manifest filename within the Zenodo record.
pub const PUBCHEM_REFERENCE_MANIFEST_KEY: &str = "manifest.json";
/// Main completed output filename within the Zenodo record.
pub const PUBCHEM_REFERENCE_COMPLETED_KEY: &str = "completed.jsonl.zst";

/// One chunk entry inside the published reference manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PubchemReferenceChunk {
    /// Chunk filename relative to the dataset root.
    pub filename: String,
    /// Number of rows stored in the chunk.
    pub row_count: u64,
    /// Compressed chunk size in bytes.
    pub bytes: u64,
    /// SHA-256 checksum of the chunk payload.
    pub sha256: String,
}

/// Top-level manifest published alongside the `PubChem` reference dump.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PubchemReferenceManifest {
    /// Manifest schema version.
    pub manifest_version: u64,
    /// Dataset schema version.
    pub dataset_schema_version: u64,
    /// Manifest creation timestamp.
    pub created_at: String,
    /// Final output filename.
    pub output_filename: String,
    /// Final output size in bytes.
    pub output_bytes: u64,
    /// SHA-256 checksum of the final output.
    pub output_sha256: String,
    /// Number of successfully classified rows.
    pub successful_rows: u64,
    /// Number of rows rejected as invalid input.
    pub invalid_rows: u64,
    /// Number of rows that failed classification.
    pub failed_rows: u64,
    /// Individual chunk metadata.
    pub chunks: Vec<PubchemReferenceChunk>,
}

/// One row from the reference `completed.jsonl.zst` export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PubchemReferenceRow {
    /// `PubChem` compound identifier.
    pub cid: u64,
    /// Canonical SMILES string used for classification.
    pub smiles: String,
    /// Expected class labels.
    #[serde(rename = "class_results")]
    pub classes: Vec<String>,
    /// Expected superclass labels.
    #[serde(rename = "superclass_results")]
    pub superclasses: Vec<String>,
    /// Expected pathway labels.
    #[serde(rename = "pathway_results")]
    pub pathways: Vec<String>,
    /// Expected glycoside flag.
    #[serde(rename = "isglycoside")]
    pub is_glycoside: bool,
}

impl PubchemReferenceRow {
    /// Decodes one JSON Lines row from the reference dataset.
    ///
    /// # Errors
    ///
    /// Returns a [`serde_json::Error`] if the line does not match the expected
    /// schema.
    pub fn from_jsonl_line(line: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(line)
    }

    /// Returns the normalized expected labels for the row.
    #[must_use]
    pub fn expected_labels(&self) -> PredictionLabels {
        PredictionLabels::new(
            self.pathways.clone(),
            self.superclasses.clone(),
            self.classes.clone(),
            Some(self.is_glycoside),
        )
    }

    /// Builds a reference-shaped row from one classifier output.
    #[must_use]
    pub fn from_classification(
        cid: u64,
        smiles: impl Into<String>,
        output: &ClassificationOutput,
    ) -> Self {
        let labels = PredictionLabels::from(output);
        Self {
            cid,
            smiles: smiles.into(),
            classes: labels.classes,
            superclasses: labels.superclasses,
            pathways: labels.pathways,
            is_glycoside: labels.is_glycoside.unwrap_or(false),
        }
    }
}

/// Normalized label bundle used for regression comparisons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictionLabels {
    /// Sorted unique pathway labels.
    pub pathways: Vec<String>,
    /// Sorted unique superclass labels.
    pub superclasses: Vec<String>,
    /// Sorted unique class labels.
    pub classes: Vec<String>,
    /// Optional glycoside flag.
    pub is_glycoside: Option<bool>,
}

impl PredictionLabels {
    /// Builds a normalized label bundle with deterministic ordering.
    #[must_use]
    pub fn new(
        pathways: Vec<String>,
        superclasses: Vec<String>,
        classes: Vec<String>,
        is_glycoside: Option<bool>,
    ) -> Self {
        Self {
            pathways: normalize_labels(pathways),
            superclasses: normalize_labels(superclasses),
            classes: normalize_labels(classes),
            is_glycoside,
        }
    }
}

impl From<&ClassificationOutput> for PredictionLabels {
    fn from(value: &ClassificationOutput) -> Self {
        Self::new(
            value
                .voted
                .pathways
                .iter()
                .map(|label| label.name.clone())
                .collect(),
            value
                .voted
                .superclasses
                .iter()
                .map(|label| label.name.clone())
                .collect(),
            value
                .voted
                .classes
                .iter()
                .map(|label| label.name.clone())
                .collect(),
            value.voted.is_glycoside,
        )
    }
}

/// Reason why a candidate prediction diverged from the reference snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionComparisonReason {
    /// Compound identifiers or SMILES no longer align row-for-row.
    StructureMismatch,
    /// Labels differ for the same compound row.
    LabelMismatch,
    /// The reference row has no candidate counterpart.
    MissingCandidateRow,
    /// The candidate stream contains an extra row after the reference ends.
    ExtraCandidateRow,
}

/// One captured mismatch between a reference row and a candidate row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictionMismatch {
    /// High-level mismatch category.
    pub reason: PredictionComparisonReason,
    /// Reference `PubChem` identifier.
    pub reference_cid: u64,
    /// Reference SMILES string.
    pub reference_smiles: String,
    /// Candidate `PubChem` identifier, when present.
    pub candidate_cid: Option<u64>,
    /// Candidate SMILES string, when present.
    pub candidate_smiles: Option<String>,
    /// Expected labels from the reference row.
    pub expected: PredictionLabels,
    /// Actual labels from the candidate row, when present.
    pub actual: Option<PredictionLabels>,
}

/// Aggregated comparison statistics for a candidate prediction stream.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictionComparison {
    /// Total number of reference rows inspected.
    pub checked_rows: u64,
    /// Number of exact matches.
    pub matched_rows: u64,
    /// Number of mismatches encountered.
    pub mismatched_rows: u64,
    /// Captured mismatch examples up to the requested limit.
    pub mismatches: Vec<PredictionMismatch>,
}

impl PredictionComparison {
    /// Records one exact match.
    pub fn push_match(&mut self) {
        self.checked_rows += 1;
        self.matched_rows += 1;
    }

    /// Records one mismatch and optionally stores its details.
    pub fn push_mismatch(&mut self, mismatch: PredictionMismatch, capture_limit: usize) {
        self.checked_rows += 1;
        self.mismatched_rows += 1;
        if self.mismatches.len() < capture_limit {
            self.mismatches.push(mismatch);
        }
    }
}

/// Compares one reference row with an optional candidate row.
#[must_use]
pub fn compare_reference_prediction(
    reference: &PubchemReferenceRow,
    candidate: Option<&PubchemReferenceRow>,
) -> Option<PredictionMismatch> {
    let expected = reference.expected_labels();

    let Some(candidate) = candidate else {
        return Some(PredictionMismatch {
            reason: PredictionComparisonReason::MissingCandidateRow,
            reference_cid: reference.cid,
            reference_smiles: reference.smiles.clone(),
            candidate_cid: None,
            candidate_smiles: None,
            expected,
            actual: None,
        });
    };

    let actual = PredictionLabels::new(
        candidate.pathways.clone(),
        candidate.superclasses.clone(),
        candidate.classes.clone(),
        Some(candidate.is_glycoside),
    );

    if reference.cid != candidate.cid || reference.smiles != candidate.smiles {
        return Some(PredictionMismatch {
            reason: PredictionComparisonReason::StructureMismatch,
            reference_cid: reference.cid,
            reference_smiles: reference.smiles.clone(),
            candidate_cid: Some(candidate.cid),
            candidate_smiles: Some(candidate.smiles.clone()),
            expected,
            actual: Some(actual),
        });
    }

    if expected == actual {
        None
    } else {
        Some(PredictionMismatch {
            reason: PredictionComparisonReason::LabelMismatch,
            reference_cid: reference.cid,
            reference_smiles: reference.smiles.clone(),
            candidate_cid: Some(candidate.cid),
            candidate_smiles: Some(candidate.smiles.clone()),
            expected,
            actual: Some(actual),
        })
    }
}

/// Compares two row streams in lockstep and captures a bounded mismatch sample.
pub fn compare_reference_rows<I, J>(
    reference_rows: I,
    candidate_rows: J,
    capture_limit: usize,
) -> PredictionComparison
where
    I: IntoIterator<Item = PubchemReferenceRow>,
    J: IntoIterator<Item = PubchemReferenceRow>,
{
    let mut comparison = PredictionComparison::default();
    let mut candidate_iter = candidate_rows.into_iter();

    for reference in reference_rows {
        let candidate = candidate_iter.next();
        match compare_reference_prediction(&reference, candidate.as_ref()) {
            None => comparison.push_match(),
            Some(mismatch) => comparison.push_mismatch(mismatch, capture_limit),
        }
    }

    for extra_candidate in candidate_iter {
        comparison.push_mismatch(
            PredictionMismatch {
                reason: PredictionComparisonReason::ExtraCandidateRow,
                reference_cid: 0,
                reference_smiles: String::new(),
                candidate_cid: Some(extra_candidate.cid),
                candidate_smiles: Some(extra_candidate.smiles.clone()),
                expected: PredictionLabels::new(Vec::new(), Vec::new(), Vec::new(), None),
                actual: Some(PredictionLabels::new(
                    extra_candidate.pathways,
                    extra_candidate.superclasses,
                    extra_candidate.classes,
                    Some(extra_candidate.is_glycoside),
                )),
            },
            capture_limit,
        );
    }

    comparison
}

fn normalize_labels(mut labels: Vec<String>) -> Vec<String> {
    labels.sort_unstable();
    labels.dedup();
    labels
}

#[cfg(test)]
mod tests {
    use crate::classifier::ClassificationOutput;
    use crate::classifier::RawPredictions;
    use crate::reference::{
        PredictionComparisonReason, compare_reference_prediction, compare_reference_rows,
    };
    use crate::voting::{IndexedLabel, VoteOutcome};

    use super::{PredictionLabels, PubchemReferenceManifest, PubchemReferenceRow};

    #[test]
    fn parses_the_observed_pubchem_row_shape() {
        let row = PubchemReferenceRow::from_jsonl_line(
            r#"{"cid":1,"smiles":"CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C","class_results":["Fatty acyl carnitines"],"superclass_results":["Fatty esters"],"pathway_results":["Fatty acids"],"isglycoside":false}"#,
        )
        .expect("sample row should parse");

        assert_eq!(row.cid, 1);
        assert_eq!(row.pathways, vec!["Fatty acids"]);
        assert_eq!(row.superclasses, vec!["Fatty esters"]);
        assert_eq!(row.classes, vec!["Fatty acyl carnitines"]);
        assert!(!row.is_glycoside);
    }

    #[test]
    fn normalizes_label_order_when_building_prediction_labels() {
        let labels = PredictionLabels::new(
            vec!["b".to_owned(), "a".to_owned(), "a".to_owned()],
            vec!["z".to_owned(), "y".to_owned()],
            vec!["d".to_owned(), "c".to_owned()],
            Some(false),
        );

        assert_eq!(labels.pathways, vec!["a", "b"]);
        assert_eq!(labels.superclasses, vec!["y", "z"]);
        assert_eq!(labels.classes, vec!["c", "d"]);
    }

    #[test]
    fn detects_label_mismatches() {
        let reference = PubchemReferenceRow {
            cid: 7,
            smiles: "CCO".to_owned(),
            pathways: vec!["Pathway".to_owned()],
            superclasses: vec!["Superclass".to_owned()],
            classes: vec!["Class".to_owned()],
            is_glycoside: false,
        };
        let candidate = PubchemReferenceRow {
            cid: 7,
            smiles: "CCO".to_owned(),
            pathways: vec!["Other pathway".to_owned()],
            superclasses: vec!["Superclass".to_owned()],
            classes: vec!["Class".to_owned()],
            is_glycoside: false,
        };

        let mismatch =
            compare_reference_prediction(&reference, Some(&candidate)).expect("should mismatch");
        assert_eq!(mismatch.reason, PredictionComparisonReason::LabelMismatch);
    }

    #[test]
    fn compares_reference_rows_in_order() {
        let reference_rows = vec![
            PubchemReferenceRow {
                cid: 1,
                smiles: "CCO".to_owned(),
                pathways: vec!["Path A".to_owned()],
                superclasses: vec!["Super A".to_owned()],
                classes: vec!["Class A".to_owned()],
                is_glycoside: false,
            },
            PubchemReferenceRow {
                cid: 2,
                smiles: "CCC".to_owned(),
                pathways: vec!["Path B".to_owned()],
                superclasses: vec!["Super B".to_owned()],
                classes: vec!["Class B".to_owned()],
                is_glycoside: false,
            },
        ];
        let candidate_rows = vec![
            reference_rows[0].clone(),
            PubchemReferenceRow {
                cid: 2,
                smiles: "CCC".to_owned(),
                pathways: vec!["Wrong".to_owned()],
                superclasses: vec!["Super B".to_owned()],
                classes: vec!["Class B".to_owned()],
                is_glycoside: false,
            },
        ];

        let comparison = compare_reference_rows(reference_rows, candidate_rows, 8);

        assert_eq!(comparison.checked_rows, 2);
        assert_eq!(comparison.matched_rows, 1);
        assert_eq!(comparison.mismatched_rows, 1);
        assert_eq!(
            comparison.mismatches[0].reason,
            PredictionComparisonReason::LabelMismatch
        );
    }

    #[test]
    fn builds_reference_rows_from_classification_outputs() {
        let output = ClassificationOutput {
            raw: RawPredictions {
                pathway: vec![0.8; 7],
                superclass: vec![0.4; 77],
                class: vec![0.2; 687],
            },
            voted: VoteOutcome {
                pathways: vec![IndexedLabel {
                    index: 0,
                    name: "Path A".to_owned(),
                }],
                superclasses: vec![IndexedLabel {
                    index: 0,
                    name: "Super A".to_owned(),
                }],
                classes: vec![IndexedLabel {
                    index: 0,
                    name: "Class A".to_owned(),
                }],
                is_glycoside: Some(false),
            },
        };

        let row = PubchemReferenceRow::from_classification(42, "CCO", &output);

        assert_eq!(row.cid, 42);
        assert_eq!(row.smiles, "CCO");
        assert_eq!(row.pathways, vec!["Path A"]);
        assert_eq!(row.superclasses, vec!["Super A"]);
        assert_eq!(row.classes, vec!["Class A"]);
        assert!(!row.is_glycoside);
    }

    #[test]
    fn parses_the_pubchem_manifest_shape() {
        let manifest = serde_json::from_str::<PubchemReferenceManifest>(
            r#"{
                "manifest_version": 1,
                "dataset_schema_version": 1,
                "created_at": "2026-04-11T15:25:53.010874179+00:00",
                "output_filename": "completed.jsonl.zst",
                "output_bytes": 115846554,
                "output_sha256": "fb7e340e4915bdf74326bdfd114c718963c94d85d222afe4e8a46b986d193e00",
                "successful_rows": 5480150,
                "invalid_rows": 734,
                "failed_rows": 26,
                "chunks": [
                    {
                        "filename": "part-000001.jsonl.zst",
                        "row_count": 45,
                        "bytes": 1357,
                        "sha256": "740b919cac6b375f9cd6631f762e354a184e5aa88c58acf0ed0cc21799b84bd8"
                    }
                ]
            }"#,
        )
        .expect("manifest should parse");

        assert_eq!(manifest.successful_rows, 5_480_150);
        assert_eq!(manifest.chunks[0].filename, "part-000001.jsonl.zst");
    }
}

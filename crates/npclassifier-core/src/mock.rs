use std::{collections::BTreeMap, sync::OnceLock};

use serde::{Deserialize, Serialize};

use crate::{
    NpClassifierError, PredictionLabels,
    fingerprint::{FingerprintGenerator, FingerprintInput, PreparedInput},
};

const PROBE_FIXTURE: &str = include_str!("../assets/mock_fingerprints.json");
const REFERENCE_128_FIXTURE: &str = include_str!("../assets/reference_fingerprints_128.json");

/// Expected labels bundled with one embedded mock fingerprint row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MockExpectedLabels {
    /// Expected pathway labels.
    #[serde(rename = "pathway_results")]
    pub pathways: Vec<String>,
    /// Expected superclass labels.
    #[serde(rename = "superclass_results")]
    pub superclasses: Vec<String>,
    /// Expected class labels.
    #[serde(rename = "class_results")]
    pub classes: Vec<String>,
    /// Expected glycoside flag.
    #[serde(rename = "isglycoside")]
    pub is_glycoside: bool,
}

impl MockExpectedLabels {
    /// Converts the fixture row into normalized comparison labels.
    #[must_use]
    pub fn prediction_labels(&self) -> PredictionLabels {
        PredictionLabels::new(
            self.pathways.clone(),
            self.superclasses.clone(),
            self.classes.clone(),
            Some(self.is_glycoside),
        )
    }
}

/// One embedded fingerprint row used for deterministic mock inference checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MockFingerprintRecord {
    /// Human-readable probe name.
    pub name: String,
    /// SMILES key used to retrieve this fixture row.
    pub smiles: String,
    /// Radius-0 count bins.
    pub formula_counts: Vec<f32>,
    /// Radius-1 and radius-2 count bins.
    pub radius_counts: Vec<f32>,
    /// Glycoside signal associated with the structure.
    pub is_glycoside: bool,
    /// Expected labels for regression checks.
    pub expected: MockExpectedLabels,
}

impl MockFingerprintRecord {
    /// Builds a validated fingerprint from the stored count vectors.
    ///
    /// # Errors
    ///
    /// Returns [`NpClassifierError::InvalidFingerprintLength`] if the embedded
    /// vectors do not match the expected model input widths.
    pub fn fingerprint(&self) -> Result<FingerprintInput, NpClassifierError> {
        FingerprintInput::new(self.formula_counts.clone(), self.radius_counts.clone())
    }

    /// Builds a prepared classifier input from the stored fixture row.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::fingerprint`].
    pub fn prepared_input(&self) -> Result<PreparedInput, NpClassifierError> {
        Ok(PreparedInput::new(
            self.fingerprint()?,
            Some(self.is_glycoside),
        ))
    }
}

/// Fingerprint generator backed by embedded regression fixtures.
#[derive(Debug, Clone)]
pub struct MockFingerprintGenerator {
    records: BTreeMap<String, MockFingerprintRecord>,
}

impl MockFingerprintGenerator {
    /// Loads the small hand-picked probe fixture.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if the embedded JSON fixture cannot be
    /// decoded.
    pub fn embedded() -> Result<Self, NpClassifierError> {
        Self::from_fixture(PROBE_FIXTURE, "probe")
    }

    /// Loads the larger 128-row reference-backed fixture.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if the embedded JSON fixture cannot be
    /// decoded.
    pub fn reference_128() -> Result<Self, NpClassifierError> {
        Self::from_fixture(REFERENCE_128_FIXTURE, "reference128")
    }

    /// Builds a generator from explicit fixture rows.
    #[must_use]
    pub fn from_records(records: Vec<MockFingerprintRecord>) -> Self {
        let records = records
            .into_iter()
            .map(|record| (record.smiles.clone(), record))
            .collect();
        Self { records }
    }

    /// Returns the fixture row for one SMILES string, if present.
    #[must_use]
    pub fn record(&self, smiles: &str) -> Option<&MockFingerprintRecord> {
        self.records.get(smiles)
    }

    /// Iterates over all embedded fixture rows.
    pub fn records(&self) -> impl Iterator<Item = &MockFingerprintRecord> {
        self.records.values()
    }

    /// Returns the sorted list of supported SMILES strings.
    #[must_use]
    pub fn supported_smiles(&self) -> Vec<&str> {
        self.records.keys().map(String::as_str).collect()
    }

    fn from_fixture(
        fixture: &'static str,
        fixture_name: &'static str,
    ) -> Result<Self, NpClassifierError> {
        static PROBE_RECORDS: OnceLock<Result<Vec<MockFingerprintRecord>, String>> =
            OnceLock::new();
        static REFERENCE_128_RECORDS: OnceLock<Result<Vec<MockFingerprintRecord>, String>> =
            OnceLock::new();

        let cell = match fixture_name {
            "probe" => &PROBE_RECORDS,
            "reference128" => &REFERENCE_128_RECORDS,
            other => {
                return Err(NpClassifierError::Fingerprint(format!(
                    "unsupported embedded mock fixture: {other}"
                )));
            }
        };

        let records = cell
            .get_or_init(|| {
                serde_json::from_str::<Vec<MockFingerprintRecord>>(fixture)
                    .map_err(|error| error.to_string())
            })
            .clone()
            .map_err(NpClassifierError::Fingerprint)?;

        Ok(Self::from_records(records))
    }
}

impl FingerprintGenerator for MockFingerprintGenerator {
    /// Looks up a prepared input in the embedded fixture table.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if the SMILES string is not present in
    /// the selected mock fixture or if the stored fingerprint shape is invalid.
    fn generate(&self, smiles: &str) -> Result<PreparedInput, NpClassifierError> {
        let record = self.records.get(smiles).ok_or_else(|| {
            NpClassifierError::Fingerprint(format!(
                "mock generator only supports embedded probe smiles; got {smiles}"
            ))
        })?;
        record.prepared_input()
    }
}

#[cfg(test)]
mod tests {
    use crate::fingerprint::FingerprintGenerator;

    use super::MockFingerprintGenerator;

    #[test]
    fn embedded_mock_fixture_contains_probe_records() {
        let generator = MockFingerprintGenerator::embedded().expect("fixture should parse");

        assert!(generator.records().count() >= 6);
        assert!(generator.record("CCO").is_some());
    }

    #[test]
    fn generator_preserves_glycoside_signal() {
        let generator = MockFingerprintGenerator::embedded().expect("fixture should parse");
        let prepared = generator
            .generate("OCC1OC(O)C(O)C(O)C1O")
            .expect("glucose probe should be embedded");

        assert_eq!(prepared.is_glycoside(), Some(true));
        assert_eq!(prepared.fingerprint().formula_counts().len(), 2048);
        assert_eq!(prepared.fingerprint().radius_counts().len(), 4096);
    }

    #[test]
    fn reference_fixture_contains_128_rows() {
        let generator = MockFingerprintGenerator::reference_128().expect("fixture should parse");

        assert_eq!(generator.records().count(), 128);
    }
}

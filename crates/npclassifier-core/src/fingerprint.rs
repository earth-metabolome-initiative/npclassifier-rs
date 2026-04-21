use serde::{Deserialize, Serialize};

use crate::error::NpClassifierError;

/// Width of the radius-0 fingerprint section.
pub const FINGERPRINT_FORMULA_BITS: usize = 2048;
/// Width of the concatenated radius-1 and radius-2 section.
pub const FINGERPRINT_RADIUS_BITS: usize = 4096;
/// Total fingerprint width expected by the recovered models.
pub const FINGERPRINT_INPUT_WIDTH: usize = FINGERPRINT_FORMULA_BITS + FINGERPRINT_RADIUS_BITS;
/// Morgan radius used by the Python draft.
pub const DRAFT_MORGAN_RADIUS: u8 = 2;

/// Fingerprint layout expected by the recovered `NPClassifier` models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FingerprintSpec {
    /// Number of radius-0 count bins.
    pub formula_bits: usize,
    /// Number of radius-1 and radius-2 count bins.
    pub radius_bits: usize,
    /// Maximum Morgan radius used to build the counts.
    pub morgan_radius: u8,
}

impl Default for FingerprintSpec {
    fn default() -> Self {
        Self {
            formula_bits: FINGERPRINT_FORMULA_BITS,
            radius_bits: FINGERPRINT_RADIUS_BITS,
            morgan_radius: DRAFT_MORGAN_RADIUS,
        }
    }
}

/// Input tensor contract expected by the recovered Keras models.
///
/// The Python draft feeds two separate tensors:
///
/// - `input_2048`: radius-0 counts stored as formula-like bins
/// - `input_4096`: radius-1 and radius-2 counts concatenated into one vector
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FingerprintInput {
    formula_counts: Vec<f32>,
    radius_counts: Vec<f32>,
}

impl FingerprintInput {
    /// Builds a validated two-part fingerprint input.
    ///
    /// # Errors
    ///
    /// Returns [`NpClassifierError::InvalidFingerprintLength`] when either
    /// input vector has the wrong width.
    pub fn new(
        formula_counts: Vec<f32>,
        radius_counts: Vec<f32>,
    ) -> Result<Self, NpClassifierError> {
        if formula_counts.len() != FINGERPRINT_FORMULA_BITS {
            return Err(NpClassifierError::InvalidFingerprintLength {
                section: "formula",
                expected: FINGERPRINT_FORMULA_BITS,
                actual: formula_counts.len(),
            });
        }

        if radius_counts.len() != FINGERPRINT_RADIUS_BITS {
            return Err(NpClassifierError::InvalidFingerprintLength {
                section: "radius",
                expected: FINGERPRINT_RADIUS_BITS,
                actual: radius_counts.len(),
            });
        }

        Ok(Self {
            formula_counts,
            radius_counts,
        })
    }

    /// Returns the radius-0 count bins.
    #[must_use]
    pub fn formula_counts(&self) -> &[f32] {
        &self.formula_counts
    }

    /// Returns the radius-1 and radius-2 count bins.
    #[must_use]
    pub fn radius_counts(&self) -> &[f32] {
        &self.radius_counts
    }

    /// Returns a single concatenated input vector for dense inference.
    #[must_use]
    pub fn concatenated(&self) -> Vec<f32> {
        let mut combined = Vec::with_capacity(FINGERPRINT_INPUT_WIDTH);
        combined.extend_from_slice(&self.formula_counts);
        combined.extend_from_slice(&self.radius_counts);
        combined
    }
}

/// Prepared classifier input with an optional glycoside hint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedInput {
    fingerprint: FingerprintInput,
    is_glycoside: Option<bool>,
}

impl PreparedInput {
    /// Builds a prepared input from a validated fingerprint and glycoside hint.
    #[must_use]
    pub fn new(fingerprint: FingerprintInput, is_glycoside: Option<bool>) -> Self {
        Self {
            fingerprint,
            is_glycoside,
        }
    }

    /// Returns the validated fingerprint input.
    #[must_use]
    pub fn fingerprint(&self) -> &FingerprintInput {
        &self.fingerprint
    }

    /// Consumes the wrapper and returns the validated fingerprint input.
    #[must_use]
    pub fn into_fingerprint(self) -> FingerprintInput {
        self.fingerprint
    }

    /// Returns the glycoside signal carried alongside the fingerprint, if any.
    #[must_use]
    pub fn is_glycoside(&self) -> Option<bool> {
        self.is_glycoside
    }
}

/// Converts SMILES strings into the counted fingerprint contract used by the model.
pub trait FingerprintGenerator {
    /// Builds a prepared classifier input for one SMILES string.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] when the generator cannot parse or
    /// encode the supplied structure.
    fn generate(&self, smiles: &str) -> Result<PreparedInput, NpClassifierError>;
}

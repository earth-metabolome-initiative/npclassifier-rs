//! Counted Morgan fingerprint generation backed by `finge-rs`.

use std::sync::Mutex;

use finge_rs::{Fingerprint, LayeredCountEcfpFingerprint, SmilesRdkitScratch};
use smiles_parser::smiles::Smiles;

use crate::{
    NpClassifierError,
    fingerprint::{
        FINGERPRINT_FORMULA_BITS, FINGERPRINT_RADIUS_BITS, FingerprintGenerator, FingerprintInput,
        PreparedInput,
    },
};

/// Counted Morgan fingerprint generator backed by `finge-rs`.
///
/// The recovered Python draft builds three folded `2048`-bit count vectors:
///
/// - exact radius-0 counts
/// - exact radius-1 counts
/// - exact radius-2 counts
///
/// The model then feeds those as two tensors:
///
/// - `input_2048`: radius-0 counts
/// - `input_4096`: radius-1 counts concatenated with radius-2 counts
///
#[derive(Debug)]
pub struct CountedMorganGenerator {
    scratch: Mutex<SmilesRdkitScratch>,
    layered: LayeredCountEcfpFingerprint,
}

impl CountedMorganGenerator {
    /// Creates a counted Morgan generator matching the recovered draft layout.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for CountedMorganGenerator {
    fn default() -> Self {
        Self {
            scratch: Mutex::new(SmilesRdkitScratch::default()),
            layered: LayeredCountEcfpFingerprint::new(2, FINGERPRINT_FORMULA_BITS),
        }
    }
}

impl FingerprintGenerator for CountedMorganGenerator {
    /// Generates the draft-compatible counted Morgan inputs for one SMILES
    /// string.
    ///
    /// # Errors
    ///
    /// Returns an [`NpClassifierError`] if the SMILES string cannot be parsed,
    /// normalized, or transformed into the expected `2048 + 4096` count
    /// layout.
    fn generate(&self, smiles: &str) -> Result<PreparedInput, NpClassifierError> {
        let smiles = smiles.parse::<Smiles>().map_err(|error| {
            NpClassifierError::Fingerprint(format!("failed to parse SMILES: {error}"))
        })?;
        let explicit_hydrogens = smiles.with_explicit_hydrogens();
        let mut scratch = self.scratch.lock().map_err(|_| {
            NpClassifierError::Fingerprint("fingerprint scratch mutex poisoned".to_owned())
        })?;
        let graph = scratch.try_prepare(&explicit_hydrogens).map_err(|error| {
            NpClassifierError::Fingerprint(format!("failed to normalize SMILES: {error}"))
        })?;

        let layered = self.layered.compute(&graph);
        let formula_counts = counts_to_f32(layered.formula().as_slice());
        let mut radius_counts = Vec::with_capacity(FINGERPRINT_RADIUS_BITS);
        for radius in [1_usize, 2_usize] {
            let layer = layered.layer(radius).ok_or_else(|| {
                NpClassifierError::Fingerprint(format!(
                    "missing exact ECFP layer {radius} from layered fingerprint"
                ))
            })?;
            radius_counts.extend(counts_to_f32(layer.as_slice()));
        }

        Ok(PreparedInput::new(
            FingerprintInput::new(formula_counts, radius_counts)?,
            None,
        ))
    }
}

#[allow(clippy::cast_precision_loss)]
fn counts_to_f32(counts: &[u32]) -> Vec<f32> {
    counts.iter().map(|count| *count as f32).collect()
}

#[cfg(test)]
mod tests {
    use crate::{FingerprintGenerator, MockFingerprintGenerator};

    use super::CountedMorganGenerator;

    #[test]
    fn counted_morgan_generates_valid_widths_for_embedded_probes() {
        let generator = CountedMorganGenerator::default();
        let fixture = MockFingerprintGenerator::embedded().expect("fixture should parse");

        for record in fixture.records() {
            let generated = generator
                .generate(&record.smiles)
                .expect("probe fingerprint generation should succeed");

            assert_eq!(generated.fingerprint().formula_counts().len(), 2048);
            assert_eq!(generated.fingerprint().radius_counts().len(), 4096);
        }
    }

    #[test]
    fn counted_morgan_is_deterministic_for_repeated_calls() {
        let generator = CountedMorganGenerator::default();
        let first = generator
            .generate("CCO")
            .expect("ethanol fingerprint generation should succeed");
        let second = generator
            .generate("CCO")
            .expect("ethanol fingerprint generation should succeed");

        assert_eq!(first, second);
    }

    #[test]
    fn counted_morgan_produces_nonzero_counts_for_ethanol() {
        let generator = CountedMorganGenerator::default();
        let fingerprint = generator
            .generate("CCO")
            .expect("ethanol fingerprint generation should succeed");

        assert!(
            fingerprint
                .fingerprint()
                .formula_counts()
                .iter()
                .any(|count| *count > 0.0)
        );
        assert!(
            fingerprint
                .fingerprint()
                .radius_counts()
                .iter()
                .any(|count| *count > 0.0)
        );
    }
}

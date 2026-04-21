//! Demo CLI for running packed classifier heads against embedded mock fixtures.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use npclassifier_core::{
    ClassificationOutput, ClassificationThresholds, ClassifierPipeline, MockFingerprintGenerator,
    NpClassifierError, PackedModelSet, PackedModelVariant, PredictionLabels,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(name = "npclassifier-demo")]
#[command(about = "Run packed NPClassifier heads against embedded mock fingerprints.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List the embedded probe molecules available to the mock fingerprint generator.
    ListMocks {
        #[arg(long, value_enum, default_value_t = CliFixture::Probes)]
        fixture: CliFixture,
    },
    /// Run one embedded probe molecule through the packed model set.
    Classify {
        #[arg(long)]
        smiles: String,
        #[arg(long, default_value = "/tmp/npclassifier-packed")]
        models: PathBuf,
        #[arg(long, value_enum, default_value_t = CliVariant::Q8Kernel)]
        variant: CliVariant,
        #[arg(long, value_enum, default_value_t = CliFixture::Probes)]
        fixture: CliFixture,
        #[arg(long)]
        thresholds_json: Option<PathBuf>,
    },
    /// Evaluate all embedded probe molecules and report mismatches against the draft labels.
    Compare {
        #[arg(long, default_value = "/tmp/npclassifier-packed")]
        models: PathBuf,
        #[arg(long, value_enum, default_value_t = CliVariant::Q8Kernel)]
        variant: CliVariant,
        #[arg(long, value_enum, default_value_t = CliFixture::Probes)]
        fixture: CliFixture,
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Evaluate f32, q8-kernel, and q4-kernel against the same mocked fingerprint fixture.
    CompareVariants {
        #[arg(long, default_value = "/tmp/npclassifier-packed")]
        models: PathBuf,
        #[arg(long, value_enum, default_value_t = CliFixture::Probes)]
        fixture: CliFixture,
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliVariant {
    F32,
    Q8Kernel,
    Q4Kernel,
}

impl From<CliVariant> for PackedModelVariant {
    fn from(value: CliVariant) -> Self {
        match value {
            CliVariant::F32 => Self::F32,
            CliVariant::Q8Kernel => Self::Q8Kernel,
            CliVariant::Q4Kernel => Self::Q4Kernel,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliFixture {
    Probes,
    Reference128,
}

impl CliFixture {
    fn load(self) -> Result<MockFingerprintGenerator, NpClassifierError> {
        match self {
            Self::Probes => MockFingerprintGenerator::embedded(),
            Self::Reference128 => MockFingerprintGenerator::reference_128(),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Probes => "probes",
            Self::Reference128 => "reference128",
        }
    }
}

#[derive(Debug, Serialize)]
struct MockRecordSummary {
    name: String,
    smiles: String,
    is_glycoside: bool,
    expected: PredictionLabels,
}

#[derive(Debug, Serialize)]
struct ClassificationReport {
    smiles: String,
    fixture: String,
    variant: String,
    output: ClassificationOutput,
    expected: Option<PredictionLabels>,
    exact_match: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
struct ComparisonMismatch {
    name: String,
    smiles: String,
    expected: PredictionLabels,
    actual: PredictionLabels,
}

#[derive(Debug, Clone, Serialize)]
struct ComparisonReport {
    fixture: String,
    variant: String,
    checked: usize,
    matched: usize,
    mismatched: usize,
    mismatches: Vec<ComparisonMismatch>,
}

#[derive(Debug, Clone)]
struct VariantEvaluation {
    variant: PackedModelVariant,
    report: ComparisonReport,
    outputs: Vec<RecordEvaluation>,
}

#[derive(Debug, Clone)]
struct RecordEvaluation {
    name: String,
    smiles: String,
    actual: PredictionLabels,
    output: ClassificationOutput,
}

#[derive(Debug, Serialize)]
struct VariantSuiteReport {
    fixture: String,
    checked: usize,
    available_fixture_size: usize,
    variants: Vec<ComparisonReport>,
    drift_from_f32: Vec<VariantDriftReport>,
}

#[derive(Debug, Serialize)]
struct VariantDriftReport {
    variant: String,
    exact_label_match_with_f32: usize,
    label_disagreements_with_f32: usize,
    disagreements: Vec<VariantLabelDisagreement>,
    pathway: HeadDriftStats,
    superclass: HeadDriftStats,
    class: HeadDriftStats,
}

#[derive(Debug, Serialize)]
struct VariantLabelDisagreement {
    name: String,
    smiles: String,
    f32: PredictionLabels,
    candidate: PredictionLabels,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct HeadDriftStats {
    compared_scores: usize,
    mean_abs_delta: f64,
    max_abs_delta: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct RunningHeadDrift {
    compared_scores: usize,
    abs_delta_sum: f64,
    max_abs_delta: f32,
}

impl RunningHeadDrift {
    fn observe(&mut self, baseline: &[f32], candidate: &[f32]) -> Result<(), NpClassifierError> {
        if baseline.len() != candidate.len() {
            return Err(NpClassifierError::Model(format!(
                "raw score width mismatch: {} vs {}",
                baseline.len(),
                candidate.len()
            )));
        }

        for (baseline_value, candidate_value) in baseline.iter().zip(candidate) {
            let abs_delta = (*baseline_value - *candidate_value).abs();
            self.compared_scores += 1;
            self.abs_delta_sum += f64::from(abs_delta);
            self.max_abs_delta = self.max_abs_delta.max(abs_delta);
        }

        Ok(())
    }

    fn finish(self) -> HeadDriftStats {
        let compared_scores = u32::try_from(self.compared_scores)
            .expect("mock drift comparisons should fit into u32");

        HeadDriftStats {
            compared_scores: self.compared_scores,
            mean_abs_delta: if self.compared_scores == 0 {
                0.0
            } else {
                self.abs_delta_sum / f64::from(compared_scores)
            },
            max_abs_delta: self.max_abs_delta,
        }
    }
}

fn main() -> Result<(), NpClassifierError> {
    let cli = Cli::parse();

    match cli.command {
        Command::ListMocks { fixture } => list_mocks(fixture),
        Command::Classify {
            smiles,
            models,
            variant,
            fixture,
            thresholds_json,
        } => classify(
            smiles,
            &models,
            variant.into(),
            fixture,
            thresholds_json.as_deref(),
        ),
        Command::Compare {
            models,
            variant,
            fixture,
            limit,
        } => compare(&models, variant.into(), fixture, limit),
        Command::CompareVariants {
            models,
            fixture,
            limit,
        } => compare_variants(&models, fixture, limit),
    }
}

fn list_mocks(fixture: CliFixture) -> Result<(), NpClassifierError> {
    let generator = fixture.load()?;
    let records = generator
        .records()
        .map(|record| MockRecordSummary {
            name: record.name.clone(),
            smiles: record.smiles.clone(),
            is_glycoside: record.is_glycoside,
            expected: record.expected.prediction_labels(),
        })
        .collect::<Vec<_>>();
    print_json(&records)
}

fn classify(
    smiles: String,
    models_dir: &Path,
    variant: PackedModelVariant,
    fixture: CliFixture,
    thresholds_json: Option<&Path>,
) -> Result<(), NpClassifierError> {
    let generator = fixture.load()?;
    let model = PackedModelSet::from_dir(models_dir, variant)?;
    let pipeline = ClassifierPipeline::with_embedded_ontology(generator.clone(), model)?
        .with_thresholds(load_thresholds(thresholds_json)?);
    let output = pipeline.classify_smiles(&smiles)?;
    let expected = generator
        .record(&smiles)
        .map(|record| record.expected.prediction_labels());
    let exact_match = expected
        .as_ref()
        .map(|labels| *labels == PredictionLabels::from(&output));

    print_json(&ClassificationReport {
        smiles,
        fixture: fixture.as_str().to_owned(),
        variant: variant.to_string(),
        output,
        expected,
        exact_match,
    })
}

fn load_thresholds(path: Option<&Path>) -> Result<ClassificationThresholds, NpClassifierError> {
    match path {
        Some(path) => {
            serde_json::from_str(&std::fs::read_to_string(path)?).map_err(NpClassifierError::from)
        }
        None => Ok(ClassificationThresholds::default()),
    }
}

fn compare(
    models_dir: &Path,
    variant: PackedModelVariant,
    fixture: CliFixture,
    limit: Option<usize>,
) -> Result<(), NpClassifierError> {
    let generator = fixture.load()?;
    let evaluation = evaluate_variant(&generator, models_dir, variant, fixture, limit)?;
    print_json(&evaluation.report)
}

fn compare_variants(
    models_dir: &Path,
    fixture: CliFixture,
    limit: Option<usize>,
) -> Result<(), NpClassifierError> {
    let generator = fixture.load()?;
    let variants = [
        PackedModelVariant::F32,
        PackedModelVariant::Q8Kernel,
        PackedModelVariant::Q4Kernel,
    ];
    let evaluations = variants
        .into_iter()
        .map(|variant| evaluate_variant(&generator, models_dir, variant, fixture, limit))
        .collect::<Result<Vec<_>, _>>()?;

    let baseline = evaluations
        .iter()
        .find(|evaluation| evaluation.variant == PackedModelVariant::F32)
        .expect("f32 baseline should always be present");
    let drift_from_f32 = evaluations
        .iter()
        .filter(|evaluation| evaluation.variant != PackedModelVariant::F32)
        .map(|evaluation| build_drift_report(baseline, evaluation))
        .collect::<Result<Vec<_>, _>>()?;

    print_json(&VariantSuiteReport {
        fixture: fixture.as_str().to_owned(),
        checked: baseline.report.checked,
        available_fixture_size: generator.records().count(),
        variants: evaluations
            .into_iter()
            .map(|evaluation| evaluation.report)
            .collect(),
        drift_from_f32,
    })
}

fn evaluate_variant(
    generator: &MockFingerprintGenerator,
    models_dir: &Path,
    variant: PackedModelVariant,
    fixture: CliFixture,
    limit: Option<usize>,
) -> Result<VariantEvaluation, NpClassifierError> {
    let model = PackedModelSet::from_dir(models_dir, variant)?;
    let pipeline = ClassifierPipeline::with_embedded_ontology(generator.clone(), model)?;
    let take_limit = limit.unwrap_or(usize::MAX);

    let mut outputs = Vec::new();
    let mut mismatches = Vec::new();
    let mut checked = 0usize;
    let mut matched = 0usize;

    for record in generator.records().take(take_limit) {
        checked += 1;
        let output = pipeline.classify_smiles(&record.smiles)?;
        let actual = PredictionLabels::from(&output);
        let expected = record.expected.prediction_labels();

        if actual == expected {
            matched += 1;
        } else {
            mismatches.push(ComparisonMismatch {
                name: record.name.clone(),
                smiles: record.smiles.clone(),
                expected: expected.clone(),
                actual: actual.clone(),
            });
        }

        outputs.push(RecordEvaluation {
            name: record.name.clone(),
            smiles: record.smiles.clone(),
            actual,
            output,
        });
    }

    Ok(VariantEvaluation {
        variant,
        report: ComparisonReport {
            fixture: fixture.as_str().to_owned(),
            variant: variant.to_string(),
            checked,
            matched,
            mismatched: mismatches.len(),
            mismatches,
        },
        outputs,
    })
}

fn build_drift_report(
    baseline: &VariantEvaluation,
    candidate: &VariantEvaluation,
) -> Result<VariantDriftReport, NpClassifierError> {
    if baseline.outputs.len() != candidate.outputs.len() {
        return Err(NpClassifierError::Model(format!(
            "variant output length mismatch: {} vs {}",
            baseline.outputs.len(),
            candidate.outputs.len()
        )));
    }

    let mut disagreements = Vec::new();
    let mut exact_label_match_with_f32 = 0usize;
    let mut pathway = RunningHeadDrift::default();
    let mut superclass = RunningHeadDrift::default();
    let mut class = RunningHeadDrift::default();

    for (baseline_record, candidate_record) in baseline.outputs.iter().zip(&candidate.outputs) {
        if baseline_record.smiles != candidate_record.smiles {
            return Err(NpClassifierError::Model(format!(
                "variant comparison alignment mismatch: {} vs {}",
                baseline_record.smiles, candidate_record.smiles
            )));
        }

        if baseline_record.actual == candidate_record.actual {
            exact_label_match_with_f32 += 1;
        } else {
            disagreements.push(VariantLabelDisagreement {
                name: candidate_record.name.clone(),
                smiles: candidate_record.smiles.clone(),
                f32: baseline_record.actual.clone(),
                candidate: candidate_record.actual.clone(),
            });
        }

        pathway.observe(
            &baseline_record.output.raw.pathway,
            &candidate_record.output.raw.pathway,
        )?;
        superclass.observe(
            &baseline_record.output.raw.superclass,
            &candidate_record.output.raw.superclass,
        )?;
        class.observe(
            &baseline_record.output.raw.class,
            &candidate_record.output.raw.class,
        )?;
    }

    Ok(VariantDriftReport {
        variant: candidate.variant.to_string(),
        exact_label_match_with_f32,
        label_disagreements_with_f32: disagreements.len(),
        disagreements,
        pathway: pathway.finish(),
        superclass: superclass.finish(),
        class: class.finish(),
    })
}

fn print_json<T: Serialize>(value: &T) -> Result<(), NpClassifierError> {
    serde_json::to_writer_pretty(std::io::stdout(), value)?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::RunningHeadDrift;

    #[test]
    fn running_head_drift_tracks_mean_and_max() {
        let mut drift = RunningHeadDrift::default();
        drift
            .observe(&[0.0, 1.0, 0.25], &[0.5, 0.5, 0.0])
            .expect("same width should compare");

        let report = drift.finish();
        assert_eq!(report.compared_scores, 3);
        assert!((report.mean_abs_delta - 0.416_666_66).abs() < 1e-6);
        assert!((report.max_abs_delta - 0.5).abs() < 1e-6);
    }
}

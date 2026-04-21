//! CLI for fetching and comparing the published Zenodo reference snapshot.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "fingerprints")]
use npclassifier_core::{
    ClassificationOutput, ClassifierPipeline, CountedMorganGenerator, PackedModelSet,
    PackedModelVariant,
};
use npclassifier_core::{
    FingerprintGenerator, MockFingerprintGenerator, NpClassifierError,
    PUBCHEM_REFERENCE_COMPLETED_KEY, PUBCHEM_REFERENCE_DOI, PUBCHEM_REFERENCE_MANIFEST_KEY,
    PUBCHEM_REFERENCE_RECORD_ID, PredictionComparison, PredictionLabels, PubchemReferenceManifest,
    PubchemReferenceRow, compare_reference_prediction,
};
#[cfg(feature = "fingerprints")]
use serde::Serialize;
use zenodo_rs::{Auth, RecordId, ZenodoClient};

const PROGRESS_TICK_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Parser)]
#[command(name = "npclassifier-reference")]
#[command(about = "Fetch and compare the NPClassifier PubChem Zenodo snapshot")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Download the manifest and completed dataset from Zenodo.
    Fetch {
        #[arg(long, default_value = "data/reference/zenodo-19513825")]
        out: PathBuf,
    },
    /// Print a few rows from a local reference file.
    Sample {
        input: PathBuf,
        #[arg(long, default_value_t = 3)]
        limit: usize,
    },
    /// Compare two aligned JSONL or JSONL.ZST files row by row.
    Compare {
        reference: PathBuf,
        candidate: PathBuf,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 20)]
        capture_limit: usize,
    },
    /// Reclassify a reference file with the local Rust stack and report label matches.
    #[cfg(feature = "fingerprints")]
    Evaluate {
        reference: PathBuf,
        #[arg(long, default_value = "/tmp/npclassifier-packed")]
        models: PathBuf,
        #[arg(long, value_enum, default_value_t = CliVariant::Q8Kernel)]
        variant: CliVariant,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 20)]
        capture_limit: usize,
        #[arg(long)]
        include_glycoside: bool,
    },
    /// Compare one generated fingerprint against an embedded legacy fixture row.
    #[cfg(feature = "fingerprints")]
    InspectFingerprint {
        smiles: String,
        #[arg(long, value_enum, default_value_t = CliFixture::Probes)]
        fixture: CliFixture,
        #[arg(long, default_value_t = 16)]
        diff_limit: usize,
    },
}

#[cfg(feature = "fingerprints")]
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliVariant {
    F32,
    Q8Kernel,
    Q4Kernel,
}

#[cfg(feature = "fingerprints")]
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliFixture {
    Probes,
    Reference128,
}

#[cfg(feature = "fingerprints")]
impl From<CliVariant> for PackedModelVariant {
    fn from(value: CliVariant) -> Self {
        match value {
            CliVariant::F32 => Self::F32,
            CliVariant::Q8Kernel => Self::Q8Kernel,
            CliVariant::Q4Kernel => Self::Q4Kernel,
        }
    }
}

#[cfg(feature = "fingerprints")]
#[derive(Debug, Serialize)]
struct EvaluationFailure {
    cid: u64,
    smiles: String,
    error: String,
}

#[cfg(feature = "fingerprints")]
#[derive(Debug, Serialize)]
struct EvaluationMismatch {
    cid: u64,
    smiles: String,
    expected: PredictionLabels,
    actual: PredictionLabels,
}

#[cfg(feature = "fingerprints")]
#[derive(Debug, Serialize)]
struct EvaluationReport {
    variant: String,
    include_glycoside: bool,
    checked_rows: u64,
    matched_rows: u64,
    mismatched_rows: u64,
    failed_rows: u64,
    pathway_exact_rows: u64,
    superclass_exact_rows: u64,
    class_exact_rows: u64,
    mismatches: Vec<EvaluationMismatch>,
    failures: Vec<EvaluationFailure>,
}

#[cfg(feature = "fingerprints")]
#[derive(Debug, Serialize)]
struct FingerprintBinDifference {
    index: usize,
    reference: f32,
    observed: f32,
    delta: f32,
}

#[cfg(feature = "fingerprints")]
#[derive(Debug, Serialize)]
struct FingerprintSectionReport {
    width: usize,
    reference_sum: f64,
    observed_sum: f64,
    reference_nonzero: usize,
    observed_nonzero: usize,
    matching_bins: usize,
    differing_bins: usize,
    sampled_differences: Vec<FingerprintBinDifference>,
}

#[cfg(feature = "fingerprints")]
#[derive(Debug, Serialize)]
struct FingerprintInspectReport {
    fixture: String,
    smiles: String,
    formula: FingerprintSectionReport,
    radius: FingerprintSectionReport,
}

#[cfg(feature = "fingerprints")]
impl EvaluationReport {
    fn new(variant: PackedModelVariant, include_glycoside: bool) -> Self {
        Self {
            variant: variant.to_string(),
            include_glycoside,
            checked_rows: 0,
            matched_rows: 0,
            mismatched_rows: 0,
            failed_rows: 0,
            pathway_exact_rows: 0,
            superclass_exact_rows: 0,
            class_exact_rows: 0,
            mismatches: Vec::new(),
            failures: Vec::new(),
        }
    }

    fn push_failure(
        &mut self,
        row: &PubchemReferenceRow,
        error: &NpClassifierError,
        capture_limit: usize,
    ) {
        self.failed_rows += 1;
        if self.failures.len() < capture_limit {
            self.failures.push(EvaluationFailure {
                cid: row.cid,
                smiles: row.smiles.clone(),
                error: error.to_string(),
            });
        }
    }

    fn observe(
        &mut self,
        row: &PubchemReferenceRow,
        output: &ClassificationOutput,
        include_glycoside: bool,
        capture_limit: usize,
    ) {
        self.checked_rows += 1;

        let expected = normalized_labels(row.expected_labels(), include_glycoside);
        let actual = normalized_labels(PredictionLabels::from(output), include_glycoside);

        if expected.pathways == actual.pathways {
            self.pathway_exact_rows += 1;
        }
        if expected.superclasses == actual.superclasses {
            self.superclass_exact_rows += 1;
        }
        if expected.classes == actual.classes {
            self.class_exact_rows += 1;
        }

        if expected == actual {
            self.matched_rows += 1;
        } else {
            self.mismatched_rows += 1;
            if self.mismatches.len() < capture_limit {
                self.mismatches.push(EvaluationMismatch {
                    cid: row.cid,
                    smiles: row.smiles.clone(),
                    expected,
                    actual,
                });
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), NpClassifierError> {
    let cli = Cli::parse();
    match cli.command {
        Command::Fetch { out } => fetch_reference_dataset(&out).await?,
        Command::Sample { input, limit } => sample_rows(&input, limit)?,
        Command::Compare {
            reference,
            candidate,
            limit,
            capture_limit,
        } => compare_files(&reference, &candidate, limit, capture_limit)?,
        #[cfg(feature = "fingerprints")]
        Command::Evaluate {
            reference,
            models,
            variant,
            limit,
            capture_limit,
            include_glycoside,
        } => evaluate_reference(
            &reference,
            &models,
            variant.into(),
            limit,
            capture_limit,
            include_glycoside,
        )?,
        #[cfg(feature = "fingerprints")]
        Command::InspectFingerprint {
            smiles,
            fixture,
            diff_limit,
        } => inspect_fingerprint(&smiles, fixture, diff_limit)?,
    }

    Ok(())
}

#[cfg(feature = "fingerprints")]
fn inspect_fingerprint(
    smiles: &str,
    fixture: CliFixture,
    diff_limit: usize,
) -> Result<(), NpClassifierError> {
    let generator = CountedMorganGenerator::default();
    let observed = generator.generate(smiles)?;
    let fixture_name = match fixture {
        CliFixture::Probes => "probes",
        CliFixture::Reference128 => "reference128",
    };
    let fixture = match fixture {
        CliFixture::Probes => MockFingerprintGenerator::embedded()?,
        CliFixture::Reference128 => MockFingerprintGenerator::reference_128()?,
    };
    let reference = fixture.record(smiles).ok_or_else(|| {
        NpClassifierError::Fingerprint(format!(
            "SMILES {smiles} not present in embedded {fixture_name} fixture"
        ))
    })?;

    let report = FingerprintInspectReport {
        fixture: fixture_name.to_owned(),
        smiles: smiles.to_owned(),
        formula: compare_section(
            reference.formula_counts.as_slice(),
            observed.fingerprint().formula_counts(),
            diff_limit,
        ),
        radius: compare_section(
            reference.radius_counts.as_slice(),
            observed.fingerprint().radius_counts(),
            diff_limit,
        ),
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(feature = "fingerprints")]
fn compare_section(
    reference: &[f32],
    observed: &[f32],
    diff_limit: usize,
) -> FingerprintSectionReport {
    let mut matching_bins = 0_usize;
    let mut differing_bins = 0_usize;
    let mut sampled_differences = Vec::new();

    for (index, (&reference_value, &observed_value)) in reference.iter().zip(observed).enumerate() {
        if reference_value.to_bits() == observed_value.to_bits() {
            matching_bins += 1;
            continue;
        }

        differing_bins += 1;
        if sampled_differences.len() < diff_limit {
            sampled_differences.push(FingerprintBinDifference {
                index,
                reference: reference_value,
                observed: observed_value,
                delta: observed_value - reference_value,
            });
        }
    }

    FingerprintSectionReport {
        width: reference.len(),
        reference_sum: reference.iter().map(|value| f64::from(*value)).sum(),
        observed_sum: observed.iter().map(|value| f64::from(*value)).sum(),
        reference_nonzero: reference.iter().filter(|value| **value != 0.0).count(),
        observed_nonzero: observed.iter().filter(|value| **value != 0.0).count(),
        matching_bins,
        differing_bins,
        sampled_differences,
    }
}

#[cfg(feature = "fingerprints")]
fn evaluate_reference(
    reference: &Path,
    models: &Path,
    variant: PackedModelVariant,
    limit: Option<usize>,
    capture_limit: usize,
    include_glycoside: bool,
) -> Result<(), NpClassifierError> {
    let generator = CountedMorganGenerator::default();
    let model = PackedModelSet::from_dir(models, variant)?;
    let pipeline = ClassifierPipeline::with_embedded_ontology(generator, model)?;
    let mut report = EvaluationReport::new(variant, include_glycoside);
    let mut lines = open_row_reader(reference)?.lines();
    let progress = counter_progress(format!("evaluate {}", reference.display()))?;

    loop {
        if limit.is_some_and(|maximum| report.checked_rows + report.failed_rows >= maximum as u64) {
            break;
        }
        let Some(row) = next_row(&mut lines)? else {
            break;
        };
        match pipeline.classify_smiles(&row.smiles) {
            Ok(output) => report.observe(&row, &output, include_glycoside, capture_limit),
            Err(error) => report.push_failure(&row, &error, capture_limit),
        }
        progress.inc(1);
    }
    progress.finish_with_message(format!(
        "evaluated {} rows",
        report.checked_rows + report.failed_rows
    ));

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

async fn fetch_reference_dataset(out: &Path) -> Result<(), NpClassifierError> {
    std::fs::create_dir_all(out)?;

    let manifest_path = out.join(PUBCHEM_REFERENCE_MANIFEST_KEY);
    let completed_path = out.join(PUBCHEM_REFERENCE_COMPLETED_KEY);
    let client = ZenodoClient::builder(Auth::new(
        std::env::var(Auth::TOKEN_ENV_VAR).unwrap_or_default(),
    ))
    .user_agent("npclassifier-rs/reference-fetch")
    .build()?;

    client
        .download_record_file_by_key_to_path(
            RecordId(PUBCHEM_REFERENCE_RECORD_ID),
            PUBCHEM_REFERENCE_MANIFEST_KEY,
            &manifest_path,
        )
        .await?;
    client
        .download_record_file_by_key_to_path(
            RecordId(PUBCHEM_REFERENCE_RECORD_ID),
            PUBCHEM_REFERENCE_COMPLETED_KEY,
            &completed_path,
        )
        .await?;

    let manifest =
        serde_json::from_reader::<_, PubchemReferenceManifest>(File::open(&manifest_path)?)?;
    println!(
        "Downloaded {} and {} from {} into {}",
        PUBCHEM_REFERENCE_MANIFEST_KEY,
        PUBCHEM_REFERENCE_COMPLETED_KEY,
        PUBCHEM_REFERENCE_DOI,
        out.display(),
    );
    println!(
        "Snapshot rows: {} successful, {} invalid, {} failed",
        manifest.successful_rows, manifest.invalid_rows, manifest.failed_rows,
    );

    Ok(())
}

fn sample_rows(input: &Path, limit: usize) -> Result<(), NpClassifierError> {
    for (index, row) in read_rows(input, Some(limit))?.into_iter().enumerate() {
        println!("#{} {}", index + 1, serde_json::to_string_pretty(&row)?);
    }

    Ok(())
}

fn compare_files(
    reference: &Path,
    candidate: &Path,
    limit: Option<usize>,
    capture_limit: usize,
) -> Result<(), NpClassifierError> {
    let mut comparison = PredictionComparison::default();
    let mut reference_lines = open_row_reader(reference)?.lines();
    let mut candidate_lines = open_row_reader(candidate)?.lines();
    let mut seen = 0_usize;
    let progress = counter_progress(format!(
        "compare {} vs {}",
        reference.display(),
        candidate.display()
    ))?;

    loop {
        if limit.is_some_and(|maximum| seen >= maximum) {
            break;
        }

        let reference_row = next_row(&mut reference_lines)?;
        let candidate_row = next_row(&mut candidate_lines)?;

        match (reference_row, candidate_row) {
            (None, None) => break,
            (Some(reference_row), candidate_row) => {
                seen += 1;
                match compare_reference_prediction(&reference_row, candidate_row.as_ref()) {
                    None => comparison.push_match(),
                    Some(mismatch) => comparison.push_mismatch(mismatch, capture_limit),
                }
                progress.inc(1);
            }
            (None, Some(candidate_row)) => {
                comparison.push_mismatch(
                    npclassifier_core::PredictionMismatch {
                        reason: npclassifier_core::PredictionComparisonReason::ExtraCandidateRow,
                        reference_cid: 0,
                        reference_smiles: String::new(),
                        candidate_cid: Some(candidate_row.cid),
                        candidate_smiles: Some(candidate_row.smiles.clone()),
                        expected: npclassifier_core::PredictionLabels::new(
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            None,
                        ),
                        actual: Some(npclassifier_core::PredictionLabels::new(
                            candidate_row.pathways,
                            candidate_row.superclasses,
                            candidate_row.classes,
                            Some(candidate_row.is_glycoside),
                        )),
                    },
                    capture_limit,
                );
                seen += 1;
                progress.inc(1);
            }
        }
    }
    progress.finish_with_message(format!("compared {seen} rows"));

    println!("checked_rows: {}", comparison.checked_rows);
    println!("matched_rows: {}", comparison.matched_rows);
    println!("mismatched_rows: {}", comparison.mismatched_rows);

    for mismatch in &comparison.mismatches {
        println!("{}", serde_json::to_string_pretty(mismatch)?);
    }

    Ok(())
}

fn next_row(
    lines: &mut impl Iterator<Item = Result<String, std::io::Error>>,
) -> Result<Option<PubchemReferenceRow>, NpClassifierError> {
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        return Ok(Some(PubchemReferenceRow::from_jsonl_line(&line)?));
    }

    Ok(None)
}

fn read_rows(
    path: &Path,
    limit: Option<usize>,
) -> Result<Vec<PubchemReferenceRow>, NpClassifierError> {
    let reader = open_row_reader(path)?;
    let mut rows = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        rows.push(PubchemReferenceRow::from_jsonl_line(&line)?);
        if limit.is_some_and(|maximum| rows.len() >= maximum) {
            break;
        }
    }

    Ok(rows)
}

fn open_row_reader(path: &Path) -> Result<Box<dyn BufRead>, NpClassifierError> {
    let file = File::open(path)?;
    if path.extension().is_some_and(|extension| extension == "zst") {
        let decoder = zstd::stream::read::Decoder::new(file)?;
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn counter_progress(label: String) -> Result<ProgressBar, NpClassifierError> {
    let progress = ProgressBar::new_spinner();
    progress.set_style(progress_style(
        "[{elapsed_precise}] {spinner:.cyan} {pos:>9} {msg}",
    )?);
    progress.set_message(label);
    progress.enable_steady_tick(PROGRESS_TICK_INTERVAL);
    Ok(progress)
}

fn progress_style(template: &str) -> Result<ProgressStyle, NpClassifierError> {
    ProgressStyle::with_template(template).map_err(|error| {
        NpClassifierError::Dataset(format!("invalid progress bar template: {error}"))
    })
}

#[cfg(feature = "fingerprints")]
fn normalized_labels(mut labels: PredictionLabels, include_glycoside: bool) -> PredictionLabels {
    if !include_glycoside {
        labels.is_glycoside = None;
    }

    labels
}

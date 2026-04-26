use dioxus::prelude::*;
use npclassifier_core::{
    WebBatchEntry as BatchEntry, WebModelVariant, WebScoredLabel as ScoredLabel,
};
use serde::Serialize;
use std::fmt::Write as _;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys::{Blob, BlobPropertyBag, HtmlAnchorElement, Url};

#[derive(Serialize)]
struct CopiedBatchEntry {
    smiles: String,
    error: Option<String>,
    predicted: CopiedPredictions,
    pathways: Vec<CopiedScore>,
    superclasses: Vec<CopiedScore>,
    classes: Vec<CopiedScore>,
}

#[derive(Serialize)]
struct CopiedPredictions {
    pathways: Vec<String>,
    superclasses: Vec<String>,
    classes: Vec<String>,
}

#[derive(Serialize)]
struct CopiedScore {
    name: String,
    score: f32,
}

pub fn build_copy_json(entries: &[BatchEntry]) -> Result<String, serde_json::Error> {
    let copied = entries
        .iter()
        .map(|entry| CopiedBatchEntry {
            smiles: entry.smiles.clone(),
            error: entry.error.clone(),
            predicted: CopiedPredictions {
                pathways: entry.labels.pathways.clone(),
                superclasses: entry.labels.superclasses.clone(),
                classes: entry.labels.classes.clone(),
            },
            pathways: copy_scores(&entry.pathway_scores),
            superclasses: copy_scores(&entry.superclass_scores),
            classes: copy_scores(&entry.class_scores),
        })
        .collect::<Vec<_>>();

    serde_json::to_string(&copied)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(clippy::unused_async))]
pub async fn copy_text_to_clipboard(text: String) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    {
        let window = web_sys::window().ok_or_else(|| String::from("window is not available"))?;
        let clipboard = window.navigator().clipboard();
        let promise = clipboard.write_text(&text);
        JsFuture::from(promise)
            .await
            .map_err(|error| js_error_text(&error))?;
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = text;
        Ok(())
    }
}

pub async fn copy_entries_as_json(entries: &[BatchEntry]) -> Result<String, String> {
    let json =
        build_copy_json(entries).map_err(|error| format!("Could not serialize JSON: {error}"))?;
    copy_text_to_clipboard(json).await?;

    Ok(if entries.len() == 1 {
        String::from("Copied 1 classification as JSON.")
    } else {
        format!("Copied {} classifications as JSON.", entries.len())
    })
}

pub fn download_filename(entries: &[BatchEntry]) -> String {
    if let [entry] = entries {
        return format!(
            "npclassifier-entry-{}.json",
            short_smiles_hash(&entry.smiles)
        );
    }

    let batch_key = entries
        .iter()
        .map(|entry| entry.smiles.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "npclassifier-batch-{}-{}.json",
        entries.len(),
        short_smiles_hash(&batch_key)
    )
}

#[cfg_attr(not(target_arch = "wasm32"), allow(clippy::unnecessary_wraps))]
pub fn download_json_file(filename: &str, text: &str) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    {
        let window = web_sys::window().ok_or_else(|| String::from("window is not available"))?;
        let document = window
            .document()
            .ok_or_else(|| String::from("document is not available"))?;

        let parts = js_sys::Array::new();
        parts.push(&JsValue::from_str(text));

        let options = BlobPropertyBag::new();
        options.set_type("application/json");
        let blob = Blob::new_with_str_sequence_and_options(&parts, &options)
            .map_err(|error| js_error_text(&error))?;
        let object_url =
            Url::create_object_url_with_blob(&blob).map_err(|error| js_error_text(&error))?;

        let anchor = document
            .create_element("a")
            .map_err(|error| js_error_text(&error))?
            .dyn_into::<HtmlAnchorElement>()
            .map_err(|_| String::from("failed to create anchor element"))?;
        anchor.set_href(&object_url);
        anchor.set_download(filename);
        anchor.click();
        Url::revoke_object_url(&object_url).map_err(|error| js_error_text(&error))?;
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (filename, text);
        Ok(())
    }
}

pub fn download_entries_as_json(entries: &[BatchEntry]) -> Result<String, String> {
    let json =
        build_copy_json(entries).map_err(|error| format!("Could not serialize JSON: {error}"))?;
    let filename = download_filename(entries);
    download_json_file(&filename, &json)
        .map_err(|error| format!("Could not download JSON: {error}"))?;

    Ok(if entries.len() == 1 {
        format!("Downloaded JSON as {filename}.")
    } else {
        format!(
            "Downloaded {} classifications as {filename}.",
            entries.len()
        )
    })
}

const PREDICTION_REPORT_TEMPLATE: &str = "mistaken-prediction.yml";
const MAX_REPORT_URL_BYTES: usize = 6_000;
const MAX_REPORT_SMILES_CHARS: usize = 900;
const MAX_REPORT_FIELD_CHARS: usize = 1_200;
const MAX_REPORT_CONTEXT_CHARS: usize = 700;
const REPORT_SCORE_LIMIT_PER_HEAD: usize = 6;

pub fn build_prediction_report_url(
    repository_url: &str,
    entry: &BatchEntry,
    model: WebModelVariant,
    git_commit: &str,
    page_url: Option<&str>,
) -> String {
    let base_url = format!("{}/issues/new", repository_url.trim_end_matches('/'));
    let url = assemble_prediction_report_url(&base_url, entry, model, git_commit, page_url, true);

    if url.len() <= MAX_REPORT_URL_BYTES {
        return url;
    }

    assemble_prediction_report_url(&base_url, entry, model, git_commit, page_url, false)
}

fn assemble_prediction_report_url(
    base_url: &str,
    entry: &BatchEntry,
    model: WebModelVariant,
    git_commit: &str,
    page_url: Option<&str>,
    include_scores: bool,
) -> String {
    let title = format!(
        "Mistaken prediction: {}",
        truncate_inline(&entry.smiles, 72)
    );
    let scores = if include_scores {
        format_report_scores(entry)
    } else {
        String::from("Scores omitted because the prefilled report URL would be too long.")
    };
    let params = [
        ("template", PREDICTION_REPORT_TEMPLATE.to_owned()),
        ("title", title),
        (
            "smiles",
            truncate_block(&entry.smiles, MAX_REPORT_SMILES_CHARS),
        ),
        (
            "model",
            format!("{} ({})", model.display_name(), model.slug()),
        ),
        (
            "prediction",
            truncate_block(&format_report_prediction(entry), MAX_REPORT_FIELD_CHARS),
        ),
        ("scores", truncate_block(&scores, MAX_REPORT_FIELD_CHARS)),
        (
            "app_context",
            truncate_block(
                &format_report_context(model, git_commit, page_url),
                MAX_REPORT_CONTEXT_CHARS,
            ),
        ),
    ];

    let query = params
        .iter()
        .map(|(key, value)| format!("{key}={}", percent_encode(value)))
        .collect::<Vec<_>>()
        .join("&");
    format!("{base_url}?{query}")
}

fn format_report_prediction(entry: &BatchEntry) -> String {
    let mut report = String::new();
    if let Some(error) = &entry.error {
        let _ = writeln!(report, "Error: {error}");
    }
    let _ = writeln!(
        report,
        "Pathways: {}",
        join_report_labels(&entry.labels.pathways)
    );
    let _ = writeln!(
        report,
        "Superclasses: {}",
        join_report_labels(&entry.labels.superclasses)
    );
    let _ = writeln!(
        report,
        "Classes: {}",
        join_report_labels(&entry.labels.classes)
    );
    report
}

fn format_report_scores(entry: &BatchEntry) -> String {
    let mut report = String::new();
    write_score_section(&mut report, "Pathway scores", &entry.pathway_scores);
    write_score_section(&mut report, "Superclass scores", &entry.superclass_scores);
    write_score_section(&mut report, "Class scores", &entry.class_scores);
    report
}

fn write_score_section(report: &mut String, title: &str, scores: &[ScoredLabel]) {
    let _ = writeln!(report, "{title}:");
    let mut sorted = scores.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.index.cmp(&right.index))
    });

    if sorted.is_empty() {
        let _ = writeln!(report, "- None");
        return;
    }

    for score in sorted.into_iter().take(REPORT_SCORE_LIMIT_PER_HEAD) {
        let _ = writeln!(
            report,
            "- {} ({}, {:.3})",
            score.name, score.index, score.score
        );
    }
}

fn format_report_context(
    model: WebModelVariant,
    git_commit: &str,
    page_url: Option<&str>,
) -> String {
    let mut context = String::new();
    let _ = writeln!(
        context,
        "Model: {} ({})",
        model.display_name(),
        model.slug()
    );
    let _ = writeln!(context, "Web commit: {git_commit}");
    if let Some(page_url) = page_url.filter(|url| !url.trim().is_empty()) {
        let _ = writeln!(context, "Page URL: {page_url}");
    }
    context
}

fn join_report_labels(labels: &[String]) -> String {
    if labels.is_empty() {
        String::from("None")
    } else {
        labels.join("; ")
    }
}

fn truncate_inline(value: &str, max_chars: usize) -> String {
    truncate_text(value, max_chars).replace(['\r', '\n'], " ")
}

fn truncate_block(value: &str, max_chars: usize) -> String {
    truncate_text(value, max_chars)
}

fn truncate_text(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}\n[truncated by NPClassifier.rs web report link]")
    } else {
        truncated
    }
}

fn percent_encode(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(char::from(byte));
            }
            _ => {
                let _ = write!(encoded, "%{byte:02X}");
            }
        }
    }
    encoded
}

fn copy_scores(scores: &[ScoredLabel]) -> Vec<CopiedScore> {
    scores
        .iter()
        .map(|score| CopiedScore {
            name: score.name.clone(),
            score: score.score,
        })
        .collect()
}

fn short_smiles_hash(text: &str) -> String {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{:06x}", hash & 0x00ff_ffff)
}

#[cfg(target_arch = "wasm32")]
fn js_error_text(error: &JsValue) -> String {
    error
        .as_string()
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| format!("{error:?}"))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use npclassifier_core::{
        PredictionLabels, WebBatchEntry as BatchEntry, WebScoredLabel as ScoredLabel,
    };

    use super::{
        MAX_REPORT_URL_BYTES, build_copy_json, build_prediction_report_url, download_filename,
    };

    #[test]
    fn build_copy_json_contains_named_vectors() {
        let entries = vec![BatchEntry {
            smiles: "CCO".to_owned(),
            error: None,
            labels: PredictionLabels::new(
                vec!["Fatty acids".to_owned()],
                vec!["Fatty acyls".to_owned()],
                vec!["Fatty alcohols".to_owned()],
                Some(false),
            ),
            pathway_scores: vec![ScoredLabel {
                index: 0,
                name: "Fatty acids".to_owned(),
                score: 0.91,
            }],
            superclass_scores: vec![ScoredLabel {
                index: 0,
                name: "Fatty acyls".to_owned(),
                score: 0.82,
            }],
            class_scores: vec![ScoredLabel {
                index: 0,
                name: "Fatty alcohols".to_owned(),
                score: 0.73,
            }],
        }];
        let copied = build_copy_json(&entries).expect("copy json should serialize");

        assert!(copied.contains("\"pathways\""));
        assert!(copied.contains("\"superclasses\""));
        assert!(copied.contains("\"classes\""));
        assert!(copied.contains("\"score\""));
    }

    #[test]
    fn download_filename_uses_short_smiles_hash() {
        let single = vec![BatchEntry {
            smiles: "CCO".to_owned(),
            error: None,
            labels: PredictionLabels::new(Vec::new(), Vec::new(), Vec::new(), None),
            pathway_scores: Vec::new(),
            superclass_scores: Vec::new(),
            class_scores: Vec::new(),
        }];
        let batch = vec![
            BatchEntry {
                smiles: "CCO".to_owned(),
                error: None,
                labels: PredictionLabels::new(Vec::new(), Vec::new(), Vec::new(), None),
                pathway_scores: Vec::new(),
                superclass_scores: Vec::new(),
                class_scores: Vec::new(),
            },
            BatchEntry {
                smiles: "CCN".to_owned(),
                error: None,
                labels: PredictionLabels::new(Vec::new(), Vec::new(), Vec::new(), None),
                pathway_scores: Vec::new(),
                superclass_scores: Vec::new(),
                class_scores: Vec::new(),
            },
        ];

        let single_name = download_filename(&single);
        let batch_name = download_filename(&batch);

        assert_eq!(single_name, download_filename(&single));
        assert_eq!(batch_name, download_filename(&batch));
        assert!(single_name.starts_with("npclassifier-entry-"));
        assert!(has_json_extension(&single_name));
        assert!(batch_name.starts_with("npclassifier-batch-2-"));
        assert!(has_json_extension(&batch_name));
        assert_ne!(single_name, batch_name);

        let single_hash = single_name
            .trim_start_matches("npclassifier-entry-")
            .trim_end_matches(".json");
        let batch_hash = batch_name
            .trim_start_matches("npclassifier-batch-2-")
            .trim_end_matches(".json");
        assert_eq!(single_hash.len(), 6);
        assert_eq!(batch_hash.len(), 6);
        assert!(single_hash.chars().all(|ch| ch.is_ascii_hexdigit()));
        assert!(batch_hash.chars().all(|ch| ch.is_ascii_hexdigit()));
    }

    #[test]
    fn prediction_report_url_prefills_issue_form_fields() {
        let entry = BatchEntry {
            smiles: "CCO".to_owned(),
            error: None,
            labels: PredictionLabels::new(
                vec!["Amino acids and Peptides".to_owned()],
                vec!["Small peptides".to_owned()],
                vec!["Aminoacids".to_owned()],
                Some(false),
            ),
            pathway_scores: vec![ScoredLabel {
                index: 1,
                name: "Amino acids and Peptides".to_owned(),
                score: 0.99,
            }],
            superclass_scores: vec![ScoredLabel {
                index: 62,
                name: "Small peptides".to_owned(),
                score: 0.98,
            }],
            class_scores: vec![ScoredLabel {
                index: 34,
                name: "Aminoacids".to_owned(),
                score: 0.97,
            }],
        };

        let url = build_prediction_report_url(
            "https://github.com/earth-metabolome-initiative/npclassifier-rs/",
            &entry,
            npclassifier_core::WebModelVariant::MiniShared,
            "abcdef123456",
            Some("https://npc.earthmetabolome.org/"),
        );

        assert!(url.starts_with(
            "https://github.com/earth-metabolome-initiative/npclassifier-rs/issues/new?"
        ));
        assert!(url.contains("template=mistaken-prediction.yml"));
        assert!(url.contains("title=Mistaken%20prediction%3A%20CCO"));
        assert!(url.contains("smiles=CCO"));
        assert!(url.contains("model=Mini%20%28mini-shared%29"));
        assert!(url.contains("prediction=Pathways%3A%20Amino%20acids%20and%20Peptides"));
        assert!(url.contains("scores=Pathway%20scores%3A"));
        assert!(url.contains("app_context=Model%3A%20Mini%20%28mini-shared%29"));
        assert!(!url.contains("labels="));
    }

    #[test]
    fn prediction_report_url_stays_bounded_for_large_smiles() {
        let entry = BatchEntry {
            smiles: "C".repeat(20_000),
            error: Some("synthetic failure".to_owned()),
            labels: PredictionLabels::new(Vec::new(), Vec::new(), Vec::new(), None),
            pathway_scores: Vec::new(),
            superclass_scores: Vec::new(),
            class_scores: (0..200)
                .map(|index| ScoredLabel {
                    index,
                    name: format!("Class {index}"),
                    score: 1.0,
                })
                .collect(),
        };

        let url = build_prediction_report_url(
            "https://github.com/earth-metabolome-initiative/npclassifier-rs",
            &entry,
            npclassifier_core::WebModelVariant::Full,
            "abcdef123456",
            None,
        );

        assert!(url.len() <= MAX_REPORT_URL_BYTES);
        assert!(url.contains("%5Btruncated%20by%20NPClassifier.rs%20web%20report%20link%5D"));
        assert!(url.contains("model=Faithful%20%28full%29"));
    }

    fn has_json_extension(filename: &str) -> bool {
        Path::new(filename)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
    }
}

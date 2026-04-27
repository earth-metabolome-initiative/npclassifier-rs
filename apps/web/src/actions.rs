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
struct ExportRow {
    entry_index: usize,
    smiles: String,
    layer: &'static str,
    rank: usize,
    label: String,
    score: f32,
    selected: bool,
}

pub struct BuiltExport {
    pub text: String,
    pub row_count: usize,
    pub entry_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportDetail {
    Summary,
    Complete,
}

impl ExportDetail {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Summary => "Summary",
            Self::Complete => "Complete scores",
        }
    }

    pub const fn description(self) -> &'static str {
        match self {
            Self::Summary => "Final selected classifications only.",
            Self::Complete => "All scores, with selected rows marked.",
        }
    }

    const fn slug(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::Complete => "complete",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportFormat {
    Csv,
    Tsv,
    Json,
}

impl ExportFormat {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Csv => "Spreadsheet (.csv)",
            Self::Tsv => "TSV",
            Self::Json => "JSON",
        }
    }

    pub const fn description(self) -> &'static str {
        match self {
            Self::Csv => "Best for Excel and Google Sheets.",
            Self::Tsv => "Plain tab-separated rows.",
            Self::Json => "JSON rows for scripts.",
        }
    }

    const fn extension(self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Tsv => "tsv",
            Self::Json => "json",
        }
    }

    const fn content_type(self) -> &'static str {
        match self {
            Self::Csv => "text/csv;charset=utf-8",
            Self::Tsv => "text/tab-separated-values;charset=utf-8",
            Self::Json => "application/json;charset=utf-8",
        }
    }

    const fn short_label(self) -> &'static str {
        match self {
            Self::Csv => "CSV",
            Self::Tsv => "TSV",
            Self::Json => "JSON",
        }
    }
}

pub fn build_export(
    entries: &[BatchEntry],
    detail: ExportDetail,
    format: ExportFormat,
) -> Result<BuiltExport, serde_json::Error> {
    let entry_count = entries.iter().filter(|entry| entry.error.is_none()).count();
    let rows = export_rows(entries, detail);
    let row_count = rows.len();
    let text = match format {
        ExportFormat::Json => serde_json::to_string(&rows)?,
        ExportFormat::Csv => format_delimited_rows(&rows, ','),
        ExportFormat::Tsv => format_delimited_rows(&rows, '\t'),
    };

    Ok(BuiltExport {
        text,
        row_count,
        entry_count,
    })
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

pub async fn copy_entries_export(
    entries: &[BatchEntry],
    detail: ExportDetail,
    format: ExportFormat,
) -> Result<String, String> {
    let export = build_export(entries, detail, format)
        .map_err(|error| format!("Could not export: {error}"))?;
    let message = format_export_message("Copied", detail, format, &export);
    copy_text_to_clipboard(export.text).await?;

    Ok(message)
}

pub fn download_filename(
    entries: &[BatchEntry],
    detail: ExportDetail,
    format: ExportFormat,
) -> String {
    let exportable_entries = entries
        .iter()
        .filter(|entry| entry.error.is_none())
        .collect::<Vec<_>>();
    if let [entry] = exportable_entries.as_slice() {
        return format!(
            "npclassifier-{}-entry-{}.{}",
            detail.slug(),
            short_smiles_hash(&entry.smiles),
            format.extension()
        );
    }

    let batch_key = exportable_entries
        .iter()
        .map(|entry| entry.smiles.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "npclassifier-{}-batch-{}-{}.{}",
        detail.slug(),
        exportable_entries.len(),
        short_smiles_hash(&batch_key),
        format.extension()
    )
}

#[cfg_attr(not(target_arch = "wasm32"), allow(clippy::unnecessary_wraps))]
pub fn download_text_file(filename: &str, text: &str, content_type: &str) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    {
        let window = web_sys::window().ok_or_else(|| String::from("window is not available"))?;
        let document = window
            .document()
            .ok_or_else(|| String::from("document is not available"))?;

        let parts = js_sys::Array::new();
        parts.push(&JsValue::from_str(text));

        let options = BlobPropertyBag::new();
        options.set_type(content_type);
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
        let _ = (filename, text, content_type);
        Ok(())
    }
}

pub fn download_entries_export(
    entries: &[BatchEntry],
    detail: ExportDetail,
    format: ExportFormat,
) -> Result<String, String> {
    let export = build_export(entries, detail, format)
        .map_err(|error| format!("Could not export: {error}"))?;
    let filename = download_filename(entries, detail, format);
    download_text_file(&filename, &export.text, format.content_type())
        .map_err(|error| format!("Could not download export: {error}"))?;

    Ok(format_export_message("Downloaded", detail, format, &export))
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

fn export_rows(entries: &[BatchEntry], detail: ExportDetail) -> Vec<ExportRow> {
    entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.error.is_none())
        .flat_map(|(index, entry)| entry_export_rows(index + 1, entry, detail))
        .collect()
}

fn entry_export_rows(
    entry_index: usize,
    entry: &BatchEntry,
    detail: ExportDetail,
) -> Vec<ExportRow> {
    let mut rows = Vec::new();
    append_layer_rows(
        &mut rows,
        entry_index,
        &entry.smiles,
        "pathway",
        &entry.labels.pathways,
        &entry.pathway_scores,
        detail,
    );
    append_layer_rows(
        &mut rows,
        entry_index,
        &entry.smiles,
        "superclass",
        &entry.labels.superclasses,
        &entry.superclass_scores,
        detail,
    );
    append_layer_rows(
        &mut rows,
        entry_index,
        &entry.smiles,
        "class",
        &entry.labels.classes,
        &entry.class_scores,
        detail,
    );
    rows
}

fn append_layer_rows(
    rows: &mut Vec<ExportRow>,
    entry_index: usize,
    smiles: &str,
    layer: &'static str,
    selected_labels: &[String],
    scores: &[ScoredLabel],
    detail: ExportDetail,
) {
    let mut ranked_scores = scores.iter().collect::<Vec<_>>();
    ranked_scores.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.index.cmp(&right.index))
    });

    for (rank_index, score) in ranked_scores.into_iter().enumerate() {
        let selected = selected_labels
            .iter()
            .any(|selected_label| selected_label == &score.name);
        if detail == ExportDetail::Summary && !selected {
            continue;
        }

        rows.push(ExportRow {
            entry_index,
            smiles: smiles.to_owned(),
            layer,
            rank: rank_index + 1,
            label: score.name.clone(),
            score: score.score,
            selected,
        });
    }
}

fn format_delimited_rows(rows: &[ExportRow], delimiter: char) -> String {
    let mut table = String::from("entry_index,smiles,layer,rank,label,score,selected\n");
    if delimiter == '\t' {
        table = table.replace(',', "\t");
    }

    for row in rows {
        write_delimited_field(&mut table, delimiter, &row.entry_index.to_string());
        table.push(delimiter);
        write_delimited_field(&mut table, delimiter, &row.smiles);
        table.push(delimiter);
        write_delimited_field(&mut table, delimiter, row.layer);
        table.push(delimiter);
        write_delimited_field(&mut table, delimiter, &row.rank.to_string());
        table.push(delimiter);
        write_delimited_field(&mut table, delimiter, &row.label);
        table.push(delimiter);
        write_delimited_field(&mut table, delimiter, &format!("{:.6}", row.score));
        table.push(delimiter);
        write_delimited_field(
            &mut table,
            delimiter,
            if row.selected { "true" } else { "false" },
        );
        table.push('\n');
    }
    table
}

fn write_delimited_field(output: &mut String, delimiter: char, field: &str) {
    if delimiter == ',' {
        write_csv_field(output, field);
    } else {
        output.push_str(&sanitize_tsv_field(field));
    }
}

fn write_csv_field(output: &mut String, field: &str) {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        output.push('"');
        output.push_str(&field.replace('"', "\"\""));
        output.push('"');
    } else {
        output.push_str(field);
    }
}

fn sanitize_tsv_field(field: &str) -> String {
    field
        .chars()
        .map(|character| match character {
            '\t' | '\n' | '\r' => ' ',
            _ => character,
        })
        .collect()
}

fn format_export_message(
    verb: &str,
    detail: ExportDetail,
    format: ExportFormat,
    export: &BuiltExport,
) -> String {
    format!(
        "{verb} {} {} export: {} row{} from {} classification{}.",
        detail.slug(),
        format.short_label(),
        export.row_count,
        plural_suffix(export.row_count),
        export.entry_count,
        plural_suffix(export.entry_count)
    )
}

const fn plural_suffix(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
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
        BuiltExport, ExportDetail, ExportFormat, MAX_REPORT_URL_BYTES, build_export,
        build_prediction_report_url, download_filename, format_export_message,
    };

    #[test]
    fn summary_export_filters_to_selected_rows_and_omits_errors() {
        let entries = sample_entries();
        let exported = build_export(&entries, ExportDetail::Summary, ExportFormat::Csv)
            .expect("summary csv should export");

        assert_eq!(exported.entry_count, 1);
        assert_eq!(exported.row_count, 3);
        assert!(
            exported
                .text
                .starts_with("entry_index,smiles,layer,rank,label,score,selected\n")
        );
        assert!(
            exported
                .text
                .contains("1,CCO,pathway,1,Fatty acids,0.910000,true")
        );
        assert!(
            exported
                .text
                .contains("1,CCO,class,1,\"Fatty alcohols, oxidized\",0.730000,true")
        );
        assert!(!exported.text.contains("Shikimates"));
        assert!(!exported.text.contains("invalid"));
        assert!(!exported.text.contains("error"));
    }

    #[test]
    fn complete_export_uses_same_rows_with_unselected_scores() {
        let entries = sample_entries();
        let exported = build_export(&entries, ExportDetail::Complete, ExportFormat::Tsv)
            .expect("complete tsv should export");

        assert_eq!(exported.entry_count, 1);
        assert_eq!(exported.row_count, 6);
        assert!(
            exported
                .text
                .starts_with("entry_index\tsmiles\tlayer\trank\tlabel\tscore\tselected\n")
        );
        assert!(
            exported
                .text
                .contains("1\tCCO\tpathway\t1\tFatty acids\t0.910000\ttrue")
        );
        assert!(
            exported
                .text
                .contains("1\tCCO\tpathway\t2\tShikimates\t0.420000\tfalse")
        );
        assert!(!exported.text.contains("invalid"));
        assert!(!exported.text.contains("error"));
    }

    #[test]
    fn json_export_uses_the_same_row_schema() {
        let entries = sample_entries();
        let exported = build_export(&entries, ExportDetail::Summary, ExportFormat::Json)
            .expect("summary json should export");

        assert_eq!(exported.row_count, 3);
        assert!(exported.text.contains("\"entry_index\":1"));
        assert!(exported.text.contains("\"layer\":\"pathway\""));
        assert!(exported.text.contains("\"selected\":true"));
        assert!(!exported.text.contains("\"error\""));
    }

    #[test]
    fn export_messages_pluralize_counts() {
        let single = BuiltExport {
            text: String::new(),
            row_count: 1,
            entry_count: 1,
        };
        let batch = BuiltExport {
            text: String::new(),
            row_count: 6,
            entry_count: 2,
        };

        assert_eq!(
            format_export_message("Copied", ExportDetail::Summary, ExportFormat::Csv, &single),
            "Copied summary CSV export: 1 row from 1 classification."
        );
        assert_eq!(
            format_export_message(
                "Downloaded",
                ExportDetail::Complete,
                ExportFormat::Json,
                &batch
            ),
            "Downloaded complete JSON export: 6 rows from 2 classifications."
        );
    }

    fn sample_entries() -> Vec<BatchEntry> {
        vec![
            BatchEntry {
                smiles: "CCO".to_owned(),
                error: None,
                labels: PredictionLabels::new(
                    vec!["Fatty acids".to_owned()],
                    vec!["Fatty acyls".to_owned()],
                    vec!["Fatty alcohols, oxidized".to_owned()],
                    Some(false),
                ),
                pathway_scores: vec![
                    ScoredLabel {
                        index: 0,
                        name: "Fatty acids".to_owned(),
                        score: 0.91,
                    },
                    ScoredLabel {
                        index: 1,
                        name: "Shikimates".to_owned(),
                        score: 0.42,
                    },
                ],
                superclass_scores: vec![
                    ScoredLabel {
                        index: 0,
                        name: "Fatty acyls".to_owned(),
                        score: 0.82,
                    },
                    ScoredLabel {
                        index: 1,
                        name: "Alkaloids".to_owned(),
                        score: 0.31,
                    },
                ],
                class_scores: vec![
                    ScoredLabel {
                        index: 0,
                        name: "Fatty alcohols, oxidized".to_owned(),
                        score: 0.73,
                    },
                    ScoredLabel {
                        index: 1,
                        name: "Aminoacids".to_owned(),
                        score: 0.22,
                    },
                ],
            },
            BatchEntry {
                smiles: "invalid".to_owned(),
                error: Some("parse error".to_owned()),
                labels: PredictionLabels::new(Vec::new(), Vec::new(), Vec::new(), None),
                pathway_scores: Vec::new(),
                superclass_scores: Vec::new(),
                class_scores: Vec::new(),
            },
        ]
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

        let single_name = download_filename(&single, ExportDetail::Summary, ExportFormat::Csv);
        let batch_name = download_filename(&batch, ExportDetail::Complete, ExportFormat::Json);

        assert_eq!(
            single_name,
            download_filename(&single, ExportDetail::Summary, ExportFormat::Csv)
        );
        assert_eq!(
            batch_name,
            download_filename(&batch, ExportDetail::Complete, ExportFormat::Json)
        );
        assert!(single_name.starts_with("npclassifier-summary-entry-"));
        assert!(has_extension(&single_name, "csv"));
        assert!(batch_name.starts_with("npclassifier-complete-batch-2-"));
        assert!(has_json_extension(&batch_name));
        assert_ne!(single_name, batch_name);

        let single_hash = single_name
            .trim_start_matches("npclassifier-summary-entry-")
            .trim_end_matches(".csv");
        let batch_hash = batch_name
            .trim_start_matches("npclassifier-complete-batch-2-")
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
        has_extension(filename, "json")
    }

    fn has_extension(filename: &str, expected: &str) -> bool {
        Path::new(filename)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
    }
}

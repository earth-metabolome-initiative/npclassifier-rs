use dioxus::prelude::*;
use npclassifier_core::{WebBatchEntry as BatchEntry, WebScoredLabel as ScoredLabel};
use serde::Serialize;
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

    use super::{build_copy_json, download_filename};

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

    fn has_json_extension(filename: &str) -> bool {
        Path::new(filename)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
    }
}

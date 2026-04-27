use dioxus::prelude::*;
use npclassifier_core::{WebBatchEntry as BatchEntry, WebModelVariant};

use crate::actions::{ExportDetail, ExportFormat};
use crate::classifier::{BatchState, LoadingState};
use crate::hooks::TransientMessage;
use crate::presentation::{
    GroupKind, VisibleLabel, WebOverview, format_score, visible_scored_labels,
};

const VISIBLE_PATHWAY_LIMIT: usize = 3;
const VISIBLE_SUPERCLASS_LIMIT: usize = 8;
const VISIBLE_CLASS_LIMIT: usize = 12;
// Inline paths adapted from Font Awesome Free 6.7.2, CC BY 4.0.
const GITHUB_ICON_PATH: &str = "M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z";
const GAUGE_HIGH_ICON_PATH: &str = "M0 256a256 256 0 1 1 512 0A256 256 0 1 1 0 256zM288 96a32 32 0 1 0 -64 0 32 32 0 1 0 64 0zM256 416c35.3 0 64-28.7 64-64c0-17.4-6.9-33.1-18.1-44.6L366 161.7c5.3-12.1-.2-26.3-12.3-31.6s-26.3 .2-31.6 12.3L257.9 288c-.6 0-1.3 0-1.9 0c-35.3 0-64 28.7-64 64s28.7 64 64 64zM176 144a32 32 0 1 0 -64 0 32 32 0 1 0 64 0zM96 288a32 32 0 1 0 0-64 32 32 0 1 0 0 64zm352-32a32 32 0 1 0 -64 0 32 32 0 1 0 64 0z";
const SCALE_BALANCED_ICON_PATH: &str = "M384 32l128 0c17.7 0 32 14.3 32 32s-14.3 32-32 32L398.4 96c-5.2 25.8-22.9 47.1-46.4 57.3L352 448l160 0c17.7 0 32 14.3 32 32s-14.3 32-32 32l-192 0-192 0c-17.7 0-32-14.3-32-32s14.3-32 32-32l160 0 0-294.7c-23.5-10.3-41.2-31.6-46.4-57.3L128 96c-17.7 0-32-14.3-32-32s14.3-32 32-32l128 0c14.6-19.4 37.8-32 64-32s49.4 12.6 64 32zm55.6 288l144.9 0L512 195.8 439.6 320zM512 416c-62.9 0-115.2-34-126-78.9c-2.6-11 1-22.3 6.7-32.1l95.2-163.2c5-8.6 14.2-13.8 24.1-13.8s19.1 5.3 24.1 13.8l95.2 163.2c5.7 9.8 9.3 21.1 6.7 32.1C627.2 382 574.9 416 512 416zM126.8 195.8L54.4 320l144.9 0L126.8 195.8zM.9 337.1c-2.6-11 1-22.3 6.7-32.1l95.2-163.2c5-8.6 14.2-13.8 24.1-13.8s19.1 5.3 24.1 13.8l95.2 163.2c5.7 9.8 9.3 21.1 6.7 32.1C242 382 189.7 416 126.8 416S11.7 382 .9 337.1z";
const LIST_CHECK_ICON_PATH: &str = "M152.1 38.2c9.9 8.9 10.7 24 1.8 33.9l-72 80c-4.4 4.9-10.6 7.8-17.2 7.9s-12.9-2.4-17.6-7L7 113C-2.3 103.6-2.3 88.4 7 79s24.6-9.4 33.9 0l22.1 22.1 55.1-61.2c8.9-9.9 24-10.7 33.9-1.8zm0 160c9.9 8.9 10.7 24 1.8 33.9l-72 80c-4.4 4.9-10.6 7.8-17.2 7.9s-12.9-2.4-17.6-7L7 273c-9.4-9.4-9.4-24.6 0-33.9s24.6-9.4 33.9 0l22.1 22.1 55.1-61.2c8.9-9.9 24-10.7 33.9-1.8zM224 96c0-17.7 14.3-32 32-32l224 0c17.7 0 32 14.3 32 32s-14.3 32-32 32l-224 0c-17.7 0-32-14.3-32-32zm0 160c0-17.7 14.3-32 32-32l224 0c17.7 0 32 14.3 32 32s-14.3 32-32 32l-224 0c-17.7 0-32-14.3-32-32zM160 416c0-17.7 14.3-32 32-32l288 0c17.7 0 32 14.3 32 32s-14.3 32-32 32l-288 0c-17.7 0-32-14.3-32-32zM48 368a48 48 0 1 1 0 96 48 48 0 1 1 0-96z";
const FILE_EXPORT_ICON_PATH: &str = "M0 64C0 28.7 28.7 0 64 0L224 0l0 128c0 17.7 14.3 32 32 32l128 0 0 128-168 0c-13.3 0-24 10.7-24 24s10.7 24 24 24l168 0 0 112c0 35.3-28.7 64-64 64L64 512c-35.3 0-64-28.7-64-64L0 64zM384 336l0-48 110.1 0-39-39c-9.4-9.4-9.4-24.6 0-33.9s24.6-9.4 33.9 0l80 80c9.4 9.4 9.4 24.6 0 33.9l-80 80c-9.4 9.4-24.6 9.4-33.9 0s-9.4-24.6 0-33.9l39-39L384 336zm0-208l-128 0L256 0 384 128z";
const COPY_ICON_PATH: &str = "M384 336l-192 0c-8.8 0-16-7.2-16-16l0-256c0-8.8 7.2-16 16-16l140.1 0L400 115.9 400 320c0 8.8-7.2 16-16 16zM192 384l192 0c35.3 0 64-28.7 64-64l0-204.1c0-12.7-5.1-24.9-14.1-33.9L366.1 14.1c-9-9-21.2-14.1-33.9-14.1L192 0c-35.3 0-64 28.7-64 64l0 256c0 35.3 28.7 64 64 64zM64 128c-35.3 0-64 28.7-64 64L0 448c0 35.3 28.7 64 64 64l192 0c35.3 0 64-28.7 64-64l0-32-48 0 0 32c0 8.8-7.2 16-16 16L64 464c-8.8 0-16-7.2-16-16l0-256c0-8.8 7.2-16 16-16l32 0 0-48-32 0z";
const DOWNLOAD_ICON_PATH: &str = "M288 32c0-17.7-14.3-32-32-32s-32 14.3-32 32l0 242.7-73.4-73.4c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3l128 128c12.5 12.5 32.8 12.5 45.3 0l128-128c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L288 274.7 288 32zM64 352c-35.3 0-64 28.7-64 64l0 32c0 35.3 28.7 64 64 64l384 0c35.3 0 64-28.7 64-64l0-32c0-35.3-28.7-64-64-64l-101.5 0-45.3 45.3c-25 25-65.5 25-90.5 0L165.5 352 64 352zm368 56a24 24 0 1 1 0 48 24 24 0 1 1 0-48z";
const CHEVRON_LEFT_ICON_PATH: &str = "M9.4 233.4c-12.5 12.5-12.5 32.8 0 45.3l192 192c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L77.3 256 246.6 86.6c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0l-192 192z";
const CHEVRON_RIGHT_ICON_PATH: &str = "M310.6 233.4c12.5 12.5 12.5 32.8 0 45.3l-192 192c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3L242.7 256 73.4 86.6c-12.5-12.5-12.5-32.8 0-45.3s32.8-12.5 45.3 0l192 192z";
const SPINNER_ICON_PATH: &str = "M304 48a48 48 0 1 0 -96 0 48 48 0 1 0 96 0zm0 416a48 48 0 1 0 -96 0 48 48 0 1 0 96 0zM48 304a48 48 0 1 0 0-96 48 48 0 1 0 0 96zm464-48a48 48 0 1 0 -96 0 48 48 0 1 0 96 0zM142.9 437A48 48 0 1 0 75 369.1 48 48 0 1 0 142.9 437zm0-294.2A48 48 0 1 0 75 75a48 48 0 1 0 67.9 67.9zM369.1 437A48 48 0 1 0 437 369.1 48 48 0 1 0 369.1 437z";
const CIRCLE_EXCLAMATION_ICON_PATH: &str = "M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zm0-384c13.3 0 24 10.7 24 24l0 112c0 13.3-10.7 24-24 24s-24-10.7-24-24l0-112c0-13.3 10.7-24 24-24zM224 352a32 32 0 1 1 64 0 32 32 0 1 1 -64 0z";
const TRIANGLE_EXCLAMATION_ICON_PATH: &str = "M256 32c14.2 0 27.3 7.5 34.5 19.8l216 368c7.3 12.4 7.3 27.7 .2 40.1S486.3 480 472 480L40 480c-14.3 0-27.6-7.7-34.7-20.1s-7-27.8 .2-40.1l216-368C228.7 39.5 241.8 32 256 32zm0 128c-13.3 0-24 10.7-24 24l0 112c0 13.3 10.7 24 24 24s24-10.7 24-24l0-112c0-13.3-10.7-24-24-24zm32 224a32 32 0 1 0 -64 0 32 32 0 1 0 64 0z";
const BAN_ICON_PATH: &str = "M367.2 412.5L99.5 144.8C77.1 176.1 64 214.5 64 256c0 106 86 192 192 192c41.5 0 79.9-13.1 111.2-35.5zm45.3-45.3C434.9 335.9 448 297.5 448 256c0-106-86-192-192-192c-41.5 0-79.9 13.1-111.2 35.5L412.5 367.2zM0 256a256 256 0 1 1 512 0A256 256 0 1 1 0 256z";
const FLAG_ICON_PATH: &str = "M64 32C64 14.3 49.7 0 32 0S0 14.3 0 32L0 64 0 368 0 480c0 17.7 14.3 32 32 32s32-14.3 32-32l0-128 64.3-16.1c41.1-10.3 84.6-5.5 122.5 13.4c44.2 22.1 95.5 24.8 141.7 7.4l34.7-13c12.5-4.7 20.8-16.6 20.8-30l0-247.7c0-23-24.2-38-44.8-27.7l-9.6 4.8c-46.3 23.2-100.8 23.2-147.1 0c-35.1-17.6-75.4-22-113.5-12.5L64 48l0-16z";
const ROUTE_ICON_PATH: &str = "M512 96c0 50.2-59.1 125.1-84.6 155c-3.8 4.4-9.4 6.1-14.5 5L320 256c-17.7 0-32 14.3-32 32s14.3 32 32 32l96 0c53 0 96 43 96 96s-43 96-96 96l-276.4 0c8.7-9.9 19.3-22.6 30-36.8c6.3-8.4 12.8-17.6 19-27.2L416 448c17.7 0 32-14.3 32-32s-14.3-32-32-32l-96 0c-53 0-96-43-96-96s43-96 96-96l39.8 0c-21-31.5-39.8-67.7-39.8-96c0-53 43-96 96-96s96 43 96 96zM117.1 489.1c-3.8 4.3-7.2 8.1-10.1 11.3l-1.8 2-.2-.2c-6 4.6-14.6 4-20-1.8C59.8 473 0 402.5 0 352c0-53 43-96 96-96s96 43 96 96c0 30-21.1 67-43.5 97.9c-10.7 14.7-21.7 28-30.8 38.5l-.6 .7zM128 352a32 32 0 1 0 -64 0 32 32 0 1 0 64 0zM416 128a32 32 0 1 0 0-64 32 32 0 1 0 0 64z";
const SITEMAP_ICON_PATH: &str = "M208 80c0-26.5 21.5-48 48-48l64 0c26.5 0 48 21.5 48 48l0 64c0 26.5-21.5 48-48 48l-8 0 0 40 152 0c30.9 0 56 25.1 56 56l0 32 8 0c26.5 0 48 21.5 48 48l0 64c0 26.5-21.5 48-48 48l-64 0c-26.5 0-48-21.5-48-48l0-64c0-26.5 21.5-48 48-48l8 0 0-32c0-4.4-3.6-8-8-8l-152 0 0 40 8 0c26.5 0 48 21.5 48 48l0 64c0 26.5-21.5 48-48 48l-64 0c-26.5 0-48-21.5-48-48l0-64c0-26.5 21.5-48 48-48l8 0 0-40-152 0c-4.4 0-8 3.6-8 8l0 32 8 0c26.5 0 48 21.5 48 48l0 64c0 26.5-21.5 48-48 48l-64 0c-26.5 0-48-21.5-48-48l0-64c0-26.5 21.5-48 48-48l8 0 0-32c0-30.9 25.1-56 56-56l152 0 0-40-8 0c-26.5 0-48-21.5-48-48l0-64z";
const TAG_ICON_PATH: &str = "M0 80L0 229.5c0 17 6.7 33.3 18.7 45.3l176 176c25 25 65.5 25 90.5 0L418.7 317.3c25-25 25-65.5 0-90.5l-176-176c-12-12-28.3-18.7-45.3-18.7L48 32C21.5 32 0 53.5 0 80zm112 32a32 32 0 1 1 0 64 32 32 0 1 1 0-64z";

#[derive(Clone, Copy)]
enum IconKind {
    Repository,
    Mini,
    Faithful,
    Classification,
    Export,
    Copy,
    Download,
    ArrowLeft,
    ArrowRight,
    Spinner,
    Error,
    Warning,
    Ban,
    Report,
    Pathway,
    Superclass,
    Class,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ResultTab {
    Classification,
    Export,
}

#[component]
pub fn Header(repository_href: &'static str, dataset_href: &'static str) -> Element {
    rsx! {
        header { class: "hero",
            div { class: "hero-main",
                div { class: "hero-head",
                    p { class: "eyebrow", "Earth Metabolome Initiative" }
                    h1 {
                        "NPClassifier"
                        span { class: "hero-rust-suffix", ".rs" }
                    }
                }
                p { class: "hero-copy",
                    "Classify natural products from SMILES in your browser."
                }
            }
            div { class: "hero-links",
                a {
                    class: "hero-link",
                    href: dataset_href,
                    target: "_blank",
                    rel: "noopener noreferrer",
                    title: "Open the Zenodo distilled dataset",
                    aria_label: "Open the Zenodo distilled dataset",
                    {app_icon(IconKind::Download)}
                    span { "Zenodo" }
                }
                a {
                    class: "hero-link",
                    href: repository_href,
                    target: "_blank",
                    rel: "noopener noreferrer",
                    title: "Go to the GitHub repository",
                    aria_label: "Go to the GitHub repository",
                    {app_icon(IconKind::Repository)}
                    span { "GitHub" }
                }
            }
        }
    }
}

#[component]
pub fn InputPanel(
    current_input: String,
    input_notice: Option<String>,
    placeholder: &'static str,
    max_input_bytes: usize,
    current_model: WebModelVariant,
    mini_tooltip: &'static str,
    faithful_tooltip: &'static str,
    on_input: EventHandler<String>,
    on_select_model: EventHandler<WebModelVariant>,
) -> Element {
    let mini_button_class = if current_model == WebModelVariant::MiniShared {
        "model-toggle-button is-active"
    } else {
        "model-toggle-button"
    };
    let faithful_button_class = if current_model == WebModelVariant::Full {
        "model-toggle-button is-reference is-active"
    } else {
        "model-toggle-button is-reference"
    };

    rsx! {
        section { class: "panel input-panel",
            div { class: "panel-head",
                div { class: "panel-head-row",
                    h2 { class: "title-with-icon",
                        span { "SMILES, one per line" }
                    }
                    div { class: "model-toggle", role: "group", aria_label: "Classifier model",
                        button {
                            class: mini_button_class,
                            title: "{mini_tooltip}",
                            aria_label: "Mini model. {mini_tooltip}",
                            onclick: move |_| on_select_model.call(WebModelVariant::MiniShared),
                            {app_icon(model_toggle_icon(WebModelVariant::MiniShared))}
                            span { "{WebModelVariant::MiniShared.display_name()}" }
                        }
                        button {
                            class: faithful_button_class,
                            title: "{faithful_tooltip}",
                            aria_label: "Faithful model. {faithful_tooltip}",
                            onclick: move |_| on_select_model.call(WebModelVariant::Full),
                            {app_icon(model_toggle_icon(WebModelVariant::Full))}
                            span { "{WebModelVariant::Full.display_name()}" }
                        }
                    }
                }
            }

            textarea {
                class: "smiles-input",
                value: current_input,
                placeholder,
                maxlength: "{max_input_bytes}",
                oninput: move |event| on_input.call(event.value()),
            }

            if let Some(input_notice) = input_notice {
                p {
                    class: "panel-copy input-note",
                    role: "alert",
                    "{input_notice}"
                }
            }
        }
    }
}

#[component]
pub fn ResultPanel(
    state: BatchState,
    active_entry: Option<BatchEntry>,
    entry_count: usize,
    active_index: usize,
    selected_tab: ResultTab,
    export_detail: ExportDetail,
    export_format: ExportFormat,
    export_entries_disabled: bool,
    report_issue_href: Option<String>,
    copy_message: Option<TransientMessage>,
    on_select_tab: EventHandler<ResultTab>,
    on_select_export_detail: EventHandler<ExportDetail>,
    on_select_export_format: EventHandler<ExportFormat>,
    on_copy_export: EventHandler<()>,
    on_download_export: EventHandler<()>,
    on_select_previous: EventHandler<()>,
    on_select_next: EventHandler<()>,
) -> Element {
    let classification_tab_class = result_tab_class(selected_tab, ResultTab::Classification);
    let export_tab_class = result_tab_class(selected_tab, ResultTab::Export);
    let result_heading = result_heading(selected_tab, active_entry.as_ref());
    let result_heading_class =
        if selected_tab == ResultTab::Classification && active_entry.is_some() {
            "active-smiles"
        } else {
            ""
        };

    rsx! {
        section { class: "panel result-panel",
            div { class: "result-head",
                div { class: "result-head-copy",
                    div { class: "result-head-main",
                        h2 {
                            class: result_heading_class,
                            title: "{result_heading}",
                            "{result_heading}"
                        }
                    }
                }
                div { class: "result-tabs", role: "tablist", aria_label: "Result panel tabs",
                    button {
                        class: classification_tab_class,
                        role: "tab",
                        aria_selected: "{selected_tab == ResultTab::Classification}",
                        onclick: move |_| on_select_tab.call(ResultTab::Classification),
                        {app_icon(IconKind::Classification)}
                        span { "Classification" }
                    }
                    button {
                        class: export_tab_class,
                        role: "tab",
                        aria_selected: "{selected_tab == ResultTab::Export}",
                        onclick: move |_| on_select_tab.call(ResultTab::Export),
                        {app_icon(IconKind::Export)}
                        span { "Export" }
                    }
                }
            }

            match selected_tab {
                ResultTab::Classification => rsx! {
                    div { class: result_body_class(&state, active_entry.as_ref()),
                        {render_result_state(&state, active_entry.as_ref())}
                    }

                    if entry_count > 1 {
                        div { class: "result-nav",
                            button {
                                class: "result-nav-button",
                                aria_label: "Show previous entry",
                                title: "Show previous entry",
                                onclick: move |_| on_select_previous.call(()),
                                {app_icon(IconKind::ArrowLeft)}
                            }
                            p { class: "result-nav-status",
                                "Showing entry {active_index + 1} of {entry_count}"
                            }
                            button {
                                class: "result-nav-button",
                                aria_label: "Show next entry",
                                title: "Show next entry",
                                onclick: move |_| on_select_next.call(()),
                                {app_icon(IconKind::ArrowRight)}
                            }
                        }
                    }

                    if let Some(report_issue_href) = report_issue_href {
                        div { class: "result-report",
                            a {
                                class: "report-link",
                                href: "{report_issue_href}",
                                target: "_blank",
                                rel: "noopener noreferrer",
                                title: "Report a mistaken prediction on GitHub",
                                aria_label: "Report this prediction as mistaken on GitHub",
                                span { class: "report-link-icon", {app_icon(IconKind::Report)} }
                                span { class: "report-link-text", "Report mistaken prediction" }
                            }
                        }
                    }
                },
                ResultTab::Export => rsx! {
                    {export_panel(
                        export_detail,
                        export_format,
                        export_entries_disabled,
                        on_select_export_detail,
                        on_select_export_format,
                        on_copy_export,
                        on_download_export,
                    )}
                },
            }

            if selected_tab == ResultTab::Export {
                if let Some(message) = copy_message {
                    div { class: "copy-feedback",
                        p {
                            key: "{message.id}",
                            class: "copy-toast",
                            aria_live: "polite",
                            "{message.text}"
                        }
                    }
                }
            }
        }
    }
}

fn result_heading(selected_tab: ResultTab, active_entry: Option<&BatchEntry>) -> String {
    match selected_tab {
        ResultTab::Classification => active_entry.map_or_else(
            || String::from("No SMILES added"),
            |entry| entry.smiles.clone(),
        ),
        ResultTab::Export => String::from("Export classifications"),
    }
}

fn app_icon(icon: IconKind) -> Element {
    match icon {
        IconKind::Repository => repository_icon(),
        IconKind::Mini => mini_icon(),
        IconKind::Faithful => faithful_icon(),
        IconKind::Classification => classification_icon(),
        IconKind::Export => export_icon(),
        IconKind::Copy => copy_icon(),
        IconKind::Download => download_icon(),
        IconKind::ArrowLeft => arrow_left_icon(),
        IconKind::ArrowRight => arrow_right_icon(),
        IconKind::Spinner => spinner_icon(),
        IconKind::Error => error_icon(),
        IconKind::Warning => warning_icon(),
        IconKind::Ban => ban_icon(),
        IconKind::Report => report_icon(),
        IconKind::Pathway => pathway_icon(),
        IconKind::Superclass => superclass_icon(),
        IconKind::Class => class_icon(),
    }
}

fn result_tab_class(selected: ResultTab, tab: ResultTab) -> &'static str {
    if selected == tab {
        "result-tab is-active"
    } else {
        "result-tab"
    }
}

fn export_option_class(is_active: bool) -> &'static str {
    if is_active {
        "export-option is-active"
    } else {
        "export-option"
    }
}

fn export_panel(
    export_detail: ExportDetail,
    export_format: ExportFormat,
    export_entries_disabled: bool,
    on_select_export_detail: EventHandler<ExportDetail>,
    on_select_export_format: EventHandler<ExportFormat>,
    on_copy_export: EventHandler<()>,
    on_download_export: EventHandler<()>,
) -> Element {
    rsx! {
        div { class: "export-panel",
            p { class: "panel-copy",
                "Choose how much detail to export, then copy the table or download it as a file. Failed entries are omitted."
            }

            div { class: "export-control",
                p { class: "export-label", "Detail" }
                div { class: "export-options",
                    {export_detail_option(
                        ExportDetail::Summary,
                        export_detail,
                        on_select_export_detail,
                    )}
                    {export_detail_option(
                        ExportDetail::Complete,
                        export_detail,
                        on_select_export_detail,
                    )}
                }
            }

            div { class: "export-control",
                p { class: "export-label", "Format" }
                div { class: "export-options",
                    {export_format_option(ExportFormat::Csv, export_format, on_select_export_format)}
                    {export_format_option(ExportFormat::Tsv, export_format, on_select_export_format)}
                    {export_format_option(ExportFormat::Json, export_format, on_select_export_format)}
                }
            }

            if export_entries_disabled {
                p { class: "copy-note",
                    "No successful classifications are available to export."
                }
            }

            div { class: "export-actions",
                button {
                    class: "export-action",
                    disabled: export_entries_disabled,
                    onclick: move |_| on_copy_export.call(()),
                    {app_icon(IconKind::Copy)}
                    span { "Copy" }
                }
                button {
                    class: "export-action",
                    disabled: export_entries_disabled,
                    onclick: move |_| on_download_export.call(()),
                    {app_icon(IconKind::Download)}
                    span { "Download" }
                }
            }
        }
    }
}

fn export_detail_option(
    detail: ExportDetail,
    selected_detail: ExportDetail,
    on_select_export_detail: EventHandler<ExportDetail>,
) -> Element {
    rsx! {
        button {
            class: export_option_class(detail == selected_detail),
            aria_pressed: "{detail == selected_detail}",
            onclick: move |_| on_select_export_detail.call(detail),
            span { class: "export-option-title", "{detail.label()}" }
            span { class: "export-option-copy", "{detail.description()}" }
        }
    }
}

fn export_format_option(
    format: ExportFormat,
    selected_format: ExportFormat,
    on_select_export_format: EventHandler<ExportFormat>,
) -> Element {
    rsx! {
        button {
            class: export_option_class(format == selected_format),
            aria_pressed: "{format == selected_format}",
            onclick: move |_| on_select_export_format.call(format),
            span { class: "export-option-title", "{format.label()}" }
            span { class: "export-option-copy", "{format.description()}" }
        }
    }
}

fn fa_icon(view_box: &'static str, path: &'static str) -> Element {
    rsx! {
        svg {
            class: "app-icon app-icon-fill",
            view_box: "{view_box}",
            path { d: "{path}" }
        }
    }
}

fn fa_spin_icon(view_box: &'static str, path: &'static str) -> Element {
    rsx! {
        svg {
            class: "app-icon app-icon-fill is-spin",
            view_box: "{view_box}",
            path { d: "{path}" }
        }
    }
}

fn repository_icon() -> Element {
    fa_icon("0 0 496 512", GITHUB_ICON_PATH)
}

fn mini_icon() -> Element {
    fa_icon("0 0 512 512", GAUGE_HIGH_ICON_PATH)
}

fn faithful_icon() -> Element {
    fa_icon("0 0 640 512", SCALE_BALANCED_ICON_PATH)
}

fn classification_icon() -> Element {
    fa_icon("0 0 512 512", LIST_CHECK_ICON_PATH)
}

fn export_icon() -> Element {
    fa_icon("0 0 576 512", FILE_EXPORT_ICON_PATH)
}

fn copy_icon() -> Element {
    fa_icon("0 0 448 512", COPY_ICON_PATH)
}

fn download_icon() -> Element {
    fa_icon("0 0 512 512", DOWNLOAD_ICON_PATH)
}

fn arrow_left_icon() -> Element {
    fa_icon("0 0 320 512", CHEVRON_LEFT_ICON_PATH)
}

fn arrow_right_icon() -> Element {
    fa_icon("0 0 320 512", CHEVRON_RIGHT_ICON_PATH)
}

fn spinner_icon() -> Element {
    fa_spin_icon("0 0 512 512", SPINNER_ICON_PATH)
}

fn error_icon() -> Element {
    fa_icon("0 0 512 512", CIRCLE_EXCLAMATION_ICON_PATH)
}

fn warning_icon() -> Element {
    fa_icon("0 0 512 512", TRIANGLE_EXCLAMATION_ICON_PATH)
}

fn ban_icon() -> Element {
    fa_icon("0 0 512 512", BAN_ICON_PATH)
}

fn report_icon() -> Element {
    fa_icon("0 0 448 512", FLAG_ICON_PATH)
}

fn pathway_icon() -> Element {
    fa_icon("0 0 512 512", ROUTE_ICON_PATH)
}

fn superclass_icon() -> Element {
    fa_icon("0 0 576 512", SITEMAP_ICON_PATH)
}

fn class_icon() -> Element {
    fa_icon("0 0 448 512", TAG_ICON_PATH)
}

fn render_result_state(state: &BatchState, active_entry: Option<&BatchEntry>) -> Element {
    match state {
        BatchState::Loading(progress) => render_loading_state(progress),
        BatchState::Fatal(error) => render_fatal_state(error),
        BatchState::Ready(_) => {
            if let Some(entry) = active_entry {
                if let Some(error) = entry.error.as_deref() {
                    render_invalid_smiles_state(error)
                } else if entry_has_no_labels(entry) {
                    render_unclassified_state()
                } else {
                    render_labeled_result(entry)
                }
            } else {
                empty_result_state()
            }
        }
        BatchState::Empty => empty_result_state(),
    }
}

fn render_loading_state(progress: &LoadingState) -> Element {
    rsx! {
        div { class: "loading-card",
                div { class: "loading-head",
                    div { class: "state-icon",
                    {app_icon(IconKind::Spinner)}
                }
                p { class: "empty-title", "{progress.label}" }
            }
            progress {
                class: "loading-progress",
                aria_label: "Classifier loading progress",
                aria_valuetext: "{progress.completed} of {progress.total} complete",
                max: "{progress.total.max(1)}",
                value: "{progress.completed}",
            }
            p { class: "loading-meta",
                "{progress.completed} / {progress.total} complete"
            }
            if let Some(eta_seconds) = progress.eta_seconds {
                p { class: "loading-meta",
                    "ETA {format_eta(eta_seconds)}"
                }
            }
            if progress.suggest_native {
                p { class: "panel-copy loading-note",
                    "This browser run is estimated to take over an hour. For large-scale batches, use the native CLI instead."
                }
            }
        }
    }
}

fn render_fatal_state(error: &str) -> Element {
    rsx! {
        div { class: "empty-state error-state",
            div { class: "state-icon error-icon",
                {app_icon(IconKind::Error)}
            }
            p { class: "empty-title", "Classifier could not run" }
            p { class: "panel-copy",
                "The worker could not load or execute the q4 classifier."
            }
            p { class: "copy-note", "{error}" }
        }
    }
}

fn render_invalid_smiles_state(error: &str) -> Element {
    rsx! {
        div { class: "empty-state error-state",
            div { class: "state-icon error-icon",
                {app_icon(IconKind::Warning)}
            }
            p { class: "empty-title", "Invalid SMILES" }
            p { class: "panel-copy",
                "This line could not be parsed as SMILES, so no classification was produced."
            }
            p { class: "copy-note", "{error}" }
        }
    }
}

fn render_unclassified_state() -> Element {
    rsx! {
        div { class: "empty-state",
            div { class: "state-icon",
                {app_icon(IconKind::Ban)}
            }
            p { class: "empty-title", "No classification for this case" }
            p { class: "panel-copy",
                "NPClassifier did not assign any pathway, superclass, or class label to this structure."
            }
            p { class: "copy-note",
                "This can happen for structures outside the model's domain or when no score clears the decision thresholds."
            }
        }
    }
}

fn render_labeled_result(entry: &BatchEntry) -> Element {
    let overview = WebOverview::load();
    let visible_pathways = visible_scored_labels(
        &entry.labels.pathways,
        &entry.pathway_scores,
        VISIBLE_PATHWAY_LIMIT,
    );
    let visible_superclasses = visible_scored_labels(
        &entry.labels.superclasses,
        &entry.superclass_scores,
        VISIBLE_SUPERCLASS_LIMIT,
    );
    let visible_classes = visible_scored_labels(
        &entry.labels.classes,
        &entry.class_scores,
        VISIBLE_CLASS_LIMIT,
    );

    rsx! {
        {label_group(
            IconKind::Pathway,
            &visible_pathways,
            GroupKind::Pathway,
            overview,
        )}
        if !entry.labels.superclasses.is_empty() {
            {label_group(
                IconKind::Superclass,
                &visible_superclasses,
                GroupKind::Superclass,
                overview,
            )}
        }
        if !entry.labels.classes.is_empty() {
            {label_group(
                IconKind::Class,
                &visible_classes,
                GroupKind::Class,
                overview,
            )}
        }
    }
}

fn result_body_class(state: &BatchState, active_entry: Option<&BatchEntry>) -> &'static str {
    match state {
        BatchState::Loading(_) => "result-body is-loading",
        BatchState::Fatal(_) => "result-body is-compact",
        BatchState::Ready(_) => {
            if let Some(entry) = active_entry {
                if entry.error.is_some() || entry_has_no_labels(entry) {
                    "result-body is-compact"
                } else {
                    "result-body is-ready"
                }
            } else {
                "result-body"
            }
        }
        BatchState::Empty => "result-body",
    }
}

fn entry_has_no_labels(entry: &BatchEntry) -> bool {
    entry.labels.pathways.is_empty()
        && entry.labels.superclasses.is_empty()
        && entry.labels.classes.is_empty()
}

fn empty_result_state() -> Element {
    rsx! {}
}

fn format_eta(total_seconds: u64) -> String {
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    match (hours, minutes, seconds) {
        (0, 0, seconds) => format!("{seconds}s"),
        (0, minutes, seconds) => format!("{minutes}m {seconds:02}s"),
        (hours, minutes, _) => format!("{hours}h {minutes:02}m"),
    }
}

fn label_group(
    icon_class: IconKind,
    labels: &[VisibleLabel],
    group_kind: GroupKind,
    overview: &WebOverview,
) -> Element {
    let title = group_title(group_kind, labels.len());
    rsx! {
        section { class: "section-card",
            div { class: "section-head",
                h3 { class: "title-with-icon",
                    {app_icon(icon_class)}
                    span { "{title}" }
                }
            }

            div { class: "predicted-row",
                if labels.is_empty() {
                    span { class: "tone-chip muted-chip", "None" }
                } else {
                    {labels.iter().map(|label| {
                        let tone = overview.tone_for_name(group_kind, &label.name);
                        rsx! {
                            span {
                                class: "tone-chip",
                                style: "{tone}",
                                span { class: "chip-label", "{label.name}" }
                                span { class: "chip-score", "{format_score(label.score)}" }
                            }
                        }
                    })}
                }
            }
        }
    }
}

fn group_title(group_kind: GroupKind, visible_label_count: usize) -> &'static str {
    match (group_kind, visible_label_count == 1) {
        (GroupKind::Pathway, true) => "Pathway",
        (GroupKind::Pathway, false) => "Pathways",
        (GroupKind::Superclass, true) => "Superclass",
        (GroupKind::Superclass, false) => "Superclasses",
        (GroupKind::Class, true) => "Class",
        (GroupKind::Class, false) => "Classes",
    }
}

fn model_toggle_icon(model: WebModelVariant) -> IconKind {
    match model {
        WebModelVariant::MiniShared => IconKind::Mini,
        WebModelVariant::Full => IconKind::Faithful,
    }
}

#[cfg(test)]
mod tests {
    use npclassifier_core::{FingerprintGenerator, PredictionLabels, WebBatchEntry};

    use super::{entry_has_no_labels, format_eta, group_title};
    use crate::presentation::GroupKind;

    #[test]
    fn counted_morgan_rejects_invalid_smiles() {
        let generator = npclassifier_core::CountedMorganGenerator::new();
        let result = generator.generate("CCCCeeeeeedsdsdsààùàùàùàùàù");

        assert!(result.is_err());
    }

    #[test]
    fn no_label_entries_are_treated_as_unclassified() {
        let entry = WebBatchEntry {
            smiles: String::from("C1=C(C(C=C(C1O)Cl)Cl)Cl"),
            error: None,
            labels: PredictionLabels::new(Vec::new(), Vec::new(), Vec::new(), None),
            pathway_scores: Vec::new(),
            superclass_scores: Vec::new(),
            class_scores: Vec::new(),
        };

        assert!(entry_has_no_labels(&entry));
    }

    #[test]
    fn eta_formatting_is_compact_and_readable() {
        assert_eq!(format_eta(42), "42s");
        assert_eq!(format_eta(125), "2m 05s");
        assert_eq!(format_eta(3_726), "1h 02m");
    }

    #[test]
    fn group_titles_match_visible_label_count() {
        assert_eq!(group_title(GroupKind::Pathway, 0), "Pathways");
        assert_eq!(group_title(GroupKind::Pathway, 1), "Pathway");
        assert_eq!(group_title(GroupKind::Pathway, 2), "Pathways");
        assert_eq!(group_title(GroupKind::Superclass, 1), "Superclass");
        assert_eq!(group_title(GroupKind::Superclass, 2), "Superclasses");
        assert_eq!(group_title(GroupKind::Class, 1), "Class");
        assert_eq!(group_title(GroupKind::Class, 2), "Classes");
    }
}

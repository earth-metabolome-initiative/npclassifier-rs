use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    time::Duration,
};

use dioxus::prelude::*;
#[cfg(target_arch = "wasm32")]
use js_sys::Date;
#[cfg(target_arch = "wasm32")]
use npclassifier_core::WebWorkerResponse;
use npclassifier_core::{WebBatchEntry as BatchEntry, WebModelVariant, WebWorkerRequest};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
#[cfg(target_arch = "wasm32")]
use web_sys::{ErrorEvent, MessageEvent, Worker, WorkerOptions, WorkerType};

const INITIAL_MODEL_LOAD_TOTAL: usize = 6;
#[cfg(target_arch = "wasm32")]
const CLASSIFIER_WORKER_SCRIPT: &str = "/generated/classifier-worker.js";
#[cfg(target_arch = "wasm32")]
const LOADING_DELAY_MS: u64 = 600;
pub const MAX_WEB_INPUT_BYTES: usize = 1_048_576;
pub const MAX_WEB_SMILES_LINES: usize = 10_000;

#[cfg(target_arch = "wasm32")]
type RequestStartedAt = f64;
#[cfg(not(target_arch = "wasm32"))]
type RequestStartedAt = std::time::Instant;

#[derive(Clone, PartialEq)]
pub struct BatchClassification {
    entries: Vec<BatchEntry>,
}

impl BatchClassification {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn entries(&self) -> &[BatchEntry] {
        &self.entries
    }

    pub fn selected(&self, index: usize) -> Option<&BatchEntry> {
        self.entries.get(index)
    }
}

#[derive(Clone, PartialEq)]
pub struct LoadingState {
    pub label: String,
    pub completed: usize,
    pub total: usize,
    pub eta_seconds: Option<u64>,
    pub suggest_native: bool,
}

#[derive(Clone, PartialEq)]
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub enum BatchState {
    Empty,
    Loading(LoadingState),
    Ready(Rc<BatchClassification>),
    Fatal(String),
}

#[derive(Clone)]
struct LoadingControls {
    request_inflight: Rc<Cell<Option<u64>>>,
    loading_visible: Rc<Cell<bool>>,
    loading_timeout_id: Rc<Cell<Option<i32>>>,
    pending_loading_state: Rc<RefCell<Option<LoadingState>>>,
    request_started_at: Rc<RefCell<Option<RequestStartedAt>>>,
}

impl LoadingControls {
    fn clear_request_timing(&self) {
        self.request_started_at.borrow_mut().take();
    }

    fn reset_loading_state(&self) {
        self.request_inflight.set(None);
        self.loading_visible.set(false);
        self.clear_request_timing();
        self.pending_loading_state.borrow_mut().take();
        clear_loading_timeout(&self.loading_timeout_id);
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn request_started_at(&self) -> Option<RequestStartedAt> {
        self.request_started_at.borrow().as_ref().copied()
    }

    fn mark_request_started(&self) {
        self.request_started_at
            .borrow_mut()
            .replace(request_now_timestamp());
    }
}

#[derive(Clone)]
struct ClassifierRuntime {
    batch_state: Signal<BatchState>,
    selected_index: Signal<usize>,
    worker_client: Result<Rc<ClassifierWorker>, String>,
    request_token: Rc<Cell<u64>>,
    loading: LoadingControls,
}

#[derive(Clone)]
pub struct ClassifierHandle {
    batch_input: Signal<String>,
    selected_model: Signal<WebModelVariant>,
    input_notice: Signal<Option<String>>,
    runtime: Rc<ClassifierRuntime>,
}

pub struct ClassifierView {
    pub state: BatchState,
    pub active_entry: Option<BatchEntry>,
    pub entry_count: usize,
    pub active_index: usize,
    pub input_notice: Option<String>,
}

impl ClassifierHandle {
    pub fn current_input(&self) -> String {
        self.batch_input.read().clone()
    }

    pub fn current_model(&self) -> WebModelVariant {
        *self.selected_model.read()
    }

    pub fn view(&self) -> ClassifierView {
        let state = self.runtime.batch_state.read().clone();
        let ready_batch = match &state {
            BatchState::Ready(batch) => Some(batch.clone()),
            _ => None,
        };
        let entry_count = ready_batch.as_ref().map_or(0, |batch| batch.len());
        let active_index = if entry_count == 0 {
            0
        } else {
            (*self.runtime.selected_index.read()).min(entry_count - 1)
        };
        let active_entry = ready_batch
            .as_ref()
            .and_then(|batch| batch.selected(active_index))
            .cloned();

        ClassifierView {
            state,
            active_entry,
            entry_count,
            active_index,
            input_notice: self.input_notice.read().clone(),
        }
    }

    pub fn export_entries(&self) -> Vec<BatchEntry> {
        match &*self.runtime.batch_state.read() {
            BatchState::Ready(batch) => batch.entries().to_vec(),
            _ => Vec::new(),
        }
    }

    pub fn has_export_entries(&self) -> bool {
        matches!(
            &*self.runtime.batch_state.read(),
            BatchState::Ready(batch) if batch.entries().iter().any(|entry| entry.error.is_none())
        )
    }

    pub fn handle_input(&self, value: &str) {
        let mut input_notice = self.input_notice;
        match validate_web_input(value) {
            Ok(()) => input_notice.set(None),
            Err(message) => {
                input_notice.set(Some(message));
                if value.len() <= MAX_WEB_INPUT_BYTES {
                    let mut batch_input = self.batch_input;
                    batch_input.set(value.to_string());
                }
                self.runtime.clear_classification();
                return;
            }
        }

        let mut batch_input = self.batch_input;
        batch_input.set(value.to_string());
        self.classify(value, self.current_model(), false);
    }

    pub fn select_model(&self, model: WebModelVariant) {
        if self.current_model() == model {
            return;
        }
        let mut selected_model = self.selected_model;
        selected_model.set(model);
        let input = self.current_input();
        self.classify(&input, model, false);
    }

    pub fn classify_current(&self, force_loading_immediately: bool) {
        let input = self.current_input();
        self.classify(&input, self.current_model(), force_loading_immediately);
    }

    pub fn select_previous(&self) {
        let view = self.view();
        if view.entry_count <= 1 {
            return;
        }
        let next_index = if view.active_index == 0 {
            view.entry_count - 1
        } else {
            view.active_index - 1
        };
        let mut selected_index = self.runtime.selected_index;
        selected_index.set(next_index);
    }

    pub fn select_next(&self) {
        let view = self.view();
        if view.entry_count <= 1 {
            return;
        }
        let next_index = if view.active_index + 1 >= view.entry_count {
            0
        } else {
            view.active_index + 1
        };
        let mut selected_index = self.runtime.selected_index;
        selected_index.set(next_index);
    }

    fn classify(
        &self,
        input: &str,
        model_variant: WebModelVariant,
        force_loading_immediately: bool,
    ) {
        let mut input_notice = self.input_notice;
        if let Err(message) = validate_web_input(input) {
            input_notice.set(Some(message));
            self.runtime.clear_classification();
            return;
        }
        input_notice.set(None);
        self.runtime
            .schedule_classification(input, model_variant, force_loading_immediately);
    }
}

impl ClassifierRuntime {
    fn clear_classification(&self) {
        let token = next_request_token(&self.request_token);
        self.loading.reset_loading_state();
        let mut batch_state = self.batch_state;
        let mut selected_index = self.selected_index;
        selected_index.set(0);
        batch_state.set(BatchState::Empty);
        let _ignored =
            send_worker_request(&self.worker_client, &WebWorkerRequest::Cancel { token });
    }

    fn schedule_classification(
        &self,
        input: &str,
        model_variant: WebModelVariant,
        force_loading_immediately: bool,
    ) {
        let mut batch_state = self.batch_state;
        let mut selected_index = self.selected_index;
        let token = next_request_token(&self.request_token);
        self.loading.request_inflight.set(Some(token));
        self.loading.loading_visible.set(false);
        self.loading.pending_loading_state.borrow_mut().take();
        self.loading.mark_request_started();
        clear_loading_timeout(&self.loading.loading_timeout_id);
        selected_index.set(0);

        let lines = parse_batch_smiles(input);
        if lines.is_empty() {
            self.loading.reset_loading_state();
            batch_state.set(BatchState::Empty);
            let _ignored =
                send_worker_request(&self.worker_client, &WebWorkerRequest::Cancel { token });
            return;
        }

        let show_loading_immediately =
            force_loading_immediately || !matches!((self.batch_state)(), BatchState::Ready(_));
        if show_loading_immediately {
            self.loading.loading_visible.set(true);
            batch_state.set(BatchState::Loading(LoadingState {
                label: format!("Starting {}", model_variant.loading_name()),
                completed: 0,
                total: INITIAL_MODEL_LOAD_TOTAL,
                eta_seconds: None,
                suggest_native: false,
            }));
        } else {
            schedule_loading_timeout(batch_state, &self.loading, token, lines.len());
        }

        let request = WebWorkerRequest::Classify {
            token,
            model: model_variant,
            lines,
        };
        if let Err(message) = send_worker_request(&self.worker_client, &request) {
            self.loading.request_inflight.set(None);
            batch_state.set(BatchState::Fatal(message));
        }
    }
}

pub fn use_classifier(initial_input: impl FnOnce() -> String + 'static) -> ClassifierHandle {
    let batch_input = use_signal(initial_input);
    let selected_model = use_signal(WebModelVariant::default);
    let input_notice = use_signal(|| None::<String>);
    let batch_state = use_signal(|| BatchState::Empty);
    let selected_index = use_signal(|| 0usize);
    let request_token = use_hook(|| Rc::new(Cell::new(0_u64)));
    let request_inflight = use_hook(|| Rc::new(Cell::new(None::<u64>)));
    let loading_visible = use_hook(|| Rc::new(Cell::new(false)));
    let loading_timeout_id = use_hook(|| Rc::new(Cell::new(None::<i32>)));
    let pending_loading_state = use_hook(|| Rc::new(RefCell::new(None::<LoadingState>)));
    let request_started_at = use_hook(|| Rc::new(RefCell::new(None::<RequestStartedAt>)));
    let loading = LoadingControls {
        request_inflight: request_inflight.clone(),
        loading_visible: loading_visible.clone(),
        loading_timeout_id: loading_timeout_id.clone(),
        pending_loading_state: pending_loading_state.clone(),
        request_started_at: request_started_at.clone(),
    };
    let worker_client = use_hook({
        let request_token = request_token.clone();
        let loading = loading.clone();
        move || create_worker_client(batch_state, request_token, loading.clone())
    });
    let runtime = Rc::new(ClassifierRuntime {
        batch_state,
        selected_index,
        worker_client: worker_client.clone(),
        request_token,
        loading,
    });
    let classifier = ClassifierHandle {
        batch_input,
        selected_model,
        input_notice,
        runtime,
    };

    let startup_classifier = classifier.clone();
    use_hook(move || {
        if startup_classifier.current_input().trim().is_empty() {
            return;
        }
        startup_classifier.classify_current(true);
    });

    classifier
}

#[cfg(target_arch = "wasm32")]
struct ClassifierWorker {
    worker: Worker,
    ready: Rc<Cell<bool>>,
    pending_request: Rc<RefCell<Option<WebWorkerRequest>>>,
    onmessage: Closure<dyn FnMut(MessageEvent)>,
    onerror: Closure<dyn FnMut(ErrorEvent)>,
}

#[cfg(target_arch = "wasm32")]
impl ClassifierWorker {
    fn new(
        batch_state: Signal<BatchState>,
        request_token: Rc<Cell<u64>>,
        loading: LoadingControls,
    ) -> Result<Self, String> {
        let worker = Self::create_worker()?;
        let ready = Rc::new(Cell::new(false));
        let pending_request = Rc::new(RefCell::new(None::<WebWorkerRequest>));
        let onmessage = Self::install_onmessage(
            &worker,
            batch_state,
            request_token,
            loading.clone(),
            ready.clone(),
            pending_request.clone(),
        );
        let onerror = Self::install_onerror(&worker, batch_state, loading);

        Ok(Self {
            worker,
            ready,
            pending_request,
            onmessage,
            onerror,
        })
    }

    fn create_worker() -> Result<Worker, String> {
        let options = WorkerOptions::new();
        options.set_type(WorkerType::Module);
        Worker::new_with_options(CLASSIFIER_WORKER_SCRIPT, &options)
            .map_err(|error| format!("failed to start worker: {}", js_error_text(&error)))
    }

    fn install_onmessage(
        worker: &Worker,
        mut batch_state: Signal<BatchState>,
        request_token: Rc<Cell<u64>>,
        loading: LoadingControls,
        ready: Rc<Cell<bool>>,
        pending_request: Rc<RefCell<Option<WebWorkerRequest>>>,
    ) -> Closure<dyn FnMut(MessageEvent)> {
        let onmessage_worker = worker.clone();
        let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
            let response = match serde_wasm_bindgen::from_value::<WebWorkerResponse>(event.data()) {
                Ok(response) => response,
                Err(error) => {
                    loading.reset_loading_state();
                    batch_state.set(BatchState::Fatal(format!(
                        "failed to decode worker response: {error}"
                    )));
                    return;
                }
            };

            if matches!(response, WebWorkerResponse::Ready) {
                ready.set(true);
                Self::flush_pending_request(&pending_request, &onmessage_worker);
                return;
            }

            if response.token() != request_token.get() {
                return;
            }

            Self::handle_worker_response(response, &loading, &mut batch_state);
        }) as Box<dyn FnMut(MessageEvent)>);
        worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage
    }

    fn install_onerror(
        worker: &Worker,
        mut batch_state: Signal<BatchState>,
        loading: LoadingControls,
    ) -> Closure<dyn FnMut(ErrorEvent)> {
        let onerror = Closure::wrap(Box::new(move |event: ErrorEvent| {
            loading.reset_loading_state();
            batch_state.set(BatchState::Fatal(format!(
                "classifier worker crashed: {}",
                event.message()
            )));
        }) as Box<dyn FnMut(ErrorEvent)>);
        worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror
    }

    fn flush_pending_request(pending_request: &RefCell<Option<WebWorkerRequest>>, worker: &Worker) {
        if let Some(request) = pending_request.borrow_mut().take()
            && let Ok(payload) = serde_wasm_bindgen::to_value(&request)
        {
            let _ = worker.post_message(&payload);
        }
    }

    fn handle_worker_response(
        response: WebWorkerResponse,
        loading: &LoadingControls,
        batch_state: &mut Signal<BatchState>,
    ) {
        match response {
            WebWorkerResponse::Progress {
                label,
                completed,
                total,
                ..
            } => Self::handle_progress(label, completed, total, loading, batch_state),
            WebWorkerResponse::Complete { entries, .. } => {
                loading.reset_loading_state();
                batch_state.set(BatchState::Ready(Rc::new(BatchClassification { entries })));
            }
            WebWorkerResponse::Fatal { message, .. } => {
                loading.reset_loading_state();
                batch_state.set(BatchState::Fatal(message));
            }
            WebWorkerResponse::Ready => unreachable!("ready messages return early"),
        }
    }

    fn handle_progress(
        label: String,
        completed: usize,
        total: usize,
        loading: &LoadingControls,
        batch_state: &mut Signal<BatchState>,
    ) {
        let eta_seconds =
            estimate_loading_eta_seconds(&label, completed, total, loading.request_started_at());
        let next_loading_state = LoadingState {
            label,
            completed,
            total,
            eta_seconds,
            suggest_native: should_suggest_native_run(eta_seconds),
        };
        if loading.loading_visible.get() {
            clear_loading_timeout(&loading.loading_timeout_id);
            batch_state.set(BatchState::Loading(next_loading_state));
        } else {
            loading
                .pending_loading_state
                .borrow_mut()
                .replace(next_loading_state);
        }
    }

    fn post(&self, message: &WebWorkerRequest) -> Result<(), String> {
        if !self.ready.get() {
            self.pending_request.replace(Some(message.clone()));
            return Ok(());
        }
        let payload = serde_wasm_bindgen::to_value(message)
            .map_err(|error| format!("failed to encode worker request: {error}"))?;
        self.worker
            .post_message(&payload)
            .map_err(|error| format!("failed to post worker request: {}", js_error_text(&error)))
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for ClassifierWorker {
    fn drop(&mut self) {
        self.worker.set_onmessage(None);
        self.worker.set_onerror(None);
        self.worker.terminate();
        let _ = &self.onmessage;
        let _ = &self.onerror;
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct ClassifierWorker;

#[cfg(not(target_arch = "wasm32"))]
impl ClassifierWorker {
    fn new(
        _batch_state: Signal<BatchState>,
        _request_token: Rc<Cell<u64>>,
        _loading: LoadingControls,
    ) -> Result<Self, String> {
        Err(String::from(
            "worker classification is only available in the browser build",
        ))
    }

    fn post(&self, _message: &WebWorkerRequest) -> Result<(), String> {
        let _ = self;
        Err(String::from(
            "worker classification is only available in the browser build",
        ))
    }
}

fn create_worker_client(
    batch_state: Signal<BatchState>,
    request_token: Rc<Cell<u64>>,
    loading: LoadingControls,
) -> Result<Rc<ClassifierWorker>, String> {
    ClassifierWorker::new(batch_state, request_token, loading).map(Rc::new)
}

fn next_request_token(request_token: &Cell<u64>) -> u64 {
    let next = request_token.get().wrapping_add(1).max(1);
    request_token.set(next);
    next
}

fn send_worker_request(
    worker_client: &Result<Rc<ClassifierWorker>, String>,
    request: &WebWorkerRequest,
) -> Result<(), String> {
    match worker_client {
        Ok(worker_client) => worker_client.post(request),
        Err(message) => Err(message.clone()),
    }
}

#[cfg(target_arch = "wasm32")]
fn clear_loading_timeout(loading_timeout_id: &Cell<Option<i32>>) {
    if let Some(timeout_id) = loading_timeout_id.take()
        && let Some(window) = web_sys::window()
    {
        window.clear_timeout_with_handle(timeout_id);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn clear_loading_timeout(_loading_timeout_id: &Cell<Option<i32>>) {}

#[cfg(target_arch = "wasm32")]
fn schedule_loading_timeout(
    mut batch_state: Signal<BatchState>,
    loading: &LoadingControls,
    token: u64,
    total: usize,
) {
    let callback_loading_timeout_id = loading.loading_timeout_id.clone();
    let callback_loading = loading.clone();
    let callback = Closure::once_into_js(move || {
        callback_loading_timeout_id.set(None);
        if callback_loading.request_inflight.get() == Some(token) {
            callback_loading.loading_visible.set(true);
            let next_loading_state = callback_loading
                .pending_loading_state
                .borrow_mut()
                .take()
                .unwrap_or_else(|| LoadingState {
                    label: String::from("Starting classifier"),
                    completed: 0,
                    total: total.max(1),
                    eta_seconds: None,
                    suggest_native: false,
                });
            batch_state.set(BatchState::Loading(next_loading_state));
        }
    });
    if let Some(window) = web_sys::window()
        && let Ok(timeout_id) = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            callback.unchecked_ref(),
            i32::try_from(LOADING_DELAY_MS).unwrap_or(i32::MAX),
        )
    {
        loading.loading_timeout_id.set(Some(timeout_id));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn schedule_loading_timeout(
    _batch_state: Signal<BatchState>,
    _loading: &LoadingControls,
    _token: u64,
    _total: usize,
) {
}

#[cfg(target_arch = "wasm32")]
fn js_error_text(error: &JsValue) -> String {
    error
        .as_string()
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| format!("{error:?}"))
}

fn parse_batch_smiles(input: &str) -> Vec<String> {
    input
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect()
}

fn validate_web_input(input: &str) -> Result<(), String> {
    if input.len() > MAX_WEB_INPUT_BYTES {
        return Err(format!(
            "The web UI accepts at most {} MiB of SMILES input. For larger batches, use `cargo npc`.",
            MAX_WEB_INPUT_BYTES / (1024 * 1024)
        ));
    }

    let smiles_count = count_non_empty_smiles_lines(input, MAX_WEB_SMILES_LINES + 1);
    if smiles_count > MAX_WEB_SMILES_LINES {
        return Err(format!(
            "This batch contains {smiles_count} SMILES. The web UI accepts at most {MAX_WEB_SMILES_LINES} SMILES per run. For larger batches, use `cargo npc`."
        ));
    }

    Ok(())
}

fn count_non_empty_smiles_lines(input: &str, limit: usize) -> usize {
    let mut count = 0;
    for line in input.lines() {
        if !line.trim().is_empty() {
            count += 1;
            if count >= limit {
                break;
            }
        }
    }
    count
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn estimate_loading_eta_seconds(
    label: &str,
    completed: usize,
    total: usize,
    request_started_at: Option<RequestStartedAt>,
) -> Option<u64> {
    if !label.starts_with("Classifying ") || completed == 0 || completed >= total {
        return None;
    }
    let elapsed = elapsed_since_request_started(request_started_at?)?;
    if elapsed.is_zero() {
        return None;
    }

    estimate_loading_eta_from_elapsed(completed, total, elapsed)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
const NATIVE_HINT_THRESHOLD_SECONDS: u64 = 60 * 60;

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn should_suggest_native_run(eta_seconds: Option<u64>) -> bool {
    eta_seconds.is_some_and(|seconds| seconds > NATIVE_HINT_THRESHOLD_SECONDS)
}

fn estimate_loading_eta_from_elapsed(
    completed: usize,
    total: usize,
    elapsed: Duration,
) -> Option<u64> {
    if completed == 0 || completed >= total || elapsed.is_zero() {
        return None;
    }

    let completed_units = u128::from(u64::try_from(completed).ok()?);
    let remaining_units = u128::from(u64::try_from(total.saturating_sub(completed)).ok()?);
    let elapsed_ms = elapsed.as_millis();
    let eta_ms = (elapsed_ms.saturating_mul(remaining_units)).div_ceil(completed_units);
    let eta_seconds = eta_ms.div_ceil(1_000).max(1);
    u64::try_from(eta_seconds).ok()
}

#[cfg(target_arch = "wasm32")]
fn request_now_timestamp() -> RequestStartedAt {
    web_sys::window()
        .and_then(|window| window.performance())
        .map_or_else(Date::now, |performance| performance.now())
}

#[cfg(not(target_arch = "wasm32"))]
fn request_now_timestamp() -> RequestStartedAt {
    std::time::Instant::now()
}

#[cfg(target_arch = "wasm32")]
fn elapsed_since_request_started(request_started_at: RequestStartedAt) -> Option<Duration> {
    let elapsed_ms = request_now_timestamp() - request_started_at;
    if !elapsed_ms.is_finite() || elapsed_ms <= 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(elapsed_ms / 1_000.0))
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::unnecessary_wraps)]
fn elapsed_since_request_started(request_started_at: RequestStartedAt) -> Option<Duration> {
    Some(request_started_at.elapsed())
}

#[cfg(test)]
mod tests {
    use std::{
        rc::Rc,
        time::{Duration, Instant},
    };

    use super::{
        BatchClassification, BatchState, MAX_WEB_SMILES_LINES, count_non_empty_smiles_lines,
        estimate_loading_eta_seconds, parse_batch_smiles, should_suggest_native_run,
        validate_web_input,
    };

    #[test]
    fn parse_batch_smiles_keeps_one_smiles_per_non_empty_line() {
        let contents = "CCO\n\nC1=CC=CC=C1  \n  \nCC(=O)O\n";
        let parsed = parse_batch_smiles(contents);
        assert_eq!(parsed, vec!["CCO", "C1=CC=CC=C1", "CC(=O)O"]);
    }

    #[test]
    fn count_non_empty_smiles_lines_stops_at_limit() {
        let count = count_non_empty_smiles_lines("CCO\nCCN\nCCC\n", 2);
        assert_eq!(count, 2);
    }

    #[test]
    fn validate_web_input_rejects_batches_above_line_cap() {
        let input = std::iter::repeat_n("CCO", MAX_WEB_SMILES_LINES + 1)
            .collect::<Vec<_>>()
            .join("\n");

        let error = validate_web_input(&input).expect_err("input should exceed web line cap");
        assert!(error.contains("web UI accepts at most"));
        assert!(error.contains("cargo npc"));
    }

    #[test]
    fn ready_state_is_constructible() {
        let state = BatchState::Ready(Rc::new(BatchClassification {
            entries: Vec::new(),
        }));
        assert!(matches!(state, BatchState::Ready(_)));
    }

    #[test]
    fn eta_is_only_reported_for_classification_progress() {
        let eta =
            estimate_loading_eta_seconds("Loading model", 1, 10, Some(started_at_seconds_ago(4)));

        assert_eq!(eta, None);
    }

    #[test]
    fn eta_is_estimated_from_elapsed_time() {
        let eta = estimate_loading_eta_seconds(
            "Classifying 10 SMILES",
            2,
            10,
            Some(started_at_seconds_ago(2)),
        );

        assert_eq!(eta, Some(8));
    }

    #[test]
    fn native_hint_is_only_shown_for_very_long_runs() {
        assert!(!should_suggest_native_run(Some(3_600)));
        assert!(should_suggest_native_run(Some(3_601)));
    }

    #[test]
    fn embedded_ontology_matches_model_head_widths() {
        use npclassifier_core::{EmbeddedOntology, ModelHead};

        let ontology = EmbeddedOntology::load().expect("embedded ontology should load");

        assert_eq!(ModelHead::Pathway.output_width(), ontology.pathway_count());
        assert_eq!(
            ModelHead::Superclass.output_width(),
            ontology.superclass_count()
        );
        assert_eq!(ModelHead::Class.output_width(), ontology.class_count());
    }

    fn started_at_seconds_ago(seconds: u64) -> Instant {
        Instant::now()
            .checked_sub(Duration::from_secs(seconds))
            .expect("test duration should fit in Instant range")
    }
}

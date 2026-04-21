//! Dedicated wasm worker entrypoint for browser-side `NPClassifier` inference.

#[cfg(target_arch = "wasm32")]
mod app {
    use gloo_timers::future::TimeoutFuture;
    use js_sys::global;
    use npclassifier_core::{
        ClassificationThresholds, CountedMorganGenerator, EmbeddedOntology, Ontology,
        PackedModelSet, PackedModelVariant, WebModelVariant, WebWorkerRequest, WebWorkerResponse,
        classify_web_entry_with_thresholds,
    };
    use std::{
        cell::{Cell, RefCell},
        rc::Rc,
    };
    use wasm_bindgen::{JsCast, JsValue, closure::Closure, prelude::wasm_bindgen};
    use wasm_bindgen_futures::{JsFuture, spawn_local};
    use web_sys::{DedicatedWorkerGlobalScope, MessageEvent, Response};

    const MODEL_LOAD_TOTAL: usize = 6;
    thread_local! {
        static ACTIVE_TOKEN: Cell<u64> = const { Cell::new(0) };
        static RUNTIMES: RefCell<Vec<(WebModelVariant, Rc<ModelRuntime>)>> = const { RefCell::new(Vec::new()) };
    }

    struct ModelRuntime {
        ontology: Ontology,
        generator: CountedMorganGenerator,
        model: PackedModelSet,
        thresholds: ClassificationThresholds,
    }

    /// Starts the dedicated classifier worker message loop.
    ///
    #[wasm_bindgen(start)]
    pub fn start() {
        let scope = worker_scope();
        let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
            let request = match serde_wasm_bindgen::from_value::<WebWorkerRequest>(event.data()) {
                Ok(request) => request,
                Err(error) => {
                    let _ = post_response(&WebWorkerResponse::Fatal {
                        token: ACTIVE_TOKEN.with(Cell::get),
                        message: format!("invalid worker request: {error}"),
                    });
                    return;
                }
            };

            match request {
                WebWorkerRequest::Cancel { token } => ACTIVE_TOKEN.with(|active| active.set(token)),
                WebWorkerRequest::Classify {
                    token,
                    model,
                    lines,
                } => {
                    ACTIVE_TOKEN.with(|active| active.set(token));
                    spawn_local(async move {
                        let total = lines.len();
                        let mut entries = Vec::with_capacity(total);

                        for smiles in lines {
                            if is_stale(token) {
                                return;
                            }

                            let runtime = match ensure_runtime(token, model).await {
                                Ok(runtime) => runtime,
                                Err(message) => {
                                    let _ =
                                        post_response(&WebWorkerResponse::Fatal { token, message });
                                    return;
                                }
                            };
                            let entry = classify_web_entry_with_thresholds(
                                &smiles,
                                &runtime.ontology,
                                &runtime.generator,
                                &runtime.model,
                                runtime.thresholds,
                            );
                            entries.push(entry);
                            let completed = entries.len();
                            if post_response(&WebWorkerResponse::Progress {
                                token,
                                label: format!("Classifying {total} SMILES"),
                                completed,
                                total,
                            })
                            .is_err()
                            {
                                return;
                            }
                            TimeoutFuture::new(0).await;
                        }

                        if is_stale(token) {
                            return;
                        }

                        let _ = post_response(&WebWorkerResponse::Complete { token, entries });
                    });
                }
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        scope.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        let _ = post_response(&WebWorkerResponse::Ready);
        onmessage.forget();
    }

    async fn ensure_runtime(
        token: u64,
        model_variant: WebModelVariant,
    ) -> Result<Rc<ModelRuntime>, String> {
        if let Some(runtime) = RUNTIMES.with(|slot| {
            slot.borrow()
                .iter()
                .find(|(variant, _)| *variant == model_variant)
                .map(|(_, runtime)| runtime.clone())
        }) {
            return Ok(runtime);
        }

        post_load_progress(
            token,
            &format!("Loading {} thresholds", model_variant.loading_name()),
            1,
        )
        .map_err(|error| js_error(&error))?;
        let thresholds =
            fetch_json::<ClassificationThresholds>(&model_thresholds_url(model_variant)).await?;
        let shared_archive = if model_variant.has_shared_archive() {
            fetch_optional_bytes(&model_shared_url(model_variant)).await?
        } else {
            None
        };
        post_load_progress(
            token,
            &format!("Loading {} pathway weights", model_variant.loading_name()),
            2,
        )
        .map_err(|error| js_error(&error))?;
        let pathway_archive = fetch_bytes(&model_pathway_url(model_variant)).await?;
        post_load_progress(
            token,
            &format!(
                "Loading {} superclass weights",
                model_variant.loading_name()
            ),
            3,
        )
        .map_err(|error| js_error(&error))?;
        let superclass_archive = fetch_bytes(&model_superclass_url(model_variant)).await?;
        post_load_progress(
            token,
            &format!("Loading {} class weights", model_variant.loading_name()),
            4,
        )
        .map_err(|error| js_error(&error))?;
        let class_archive = fetch_bytes(&model_class_url(model_variant)).await?;
        post_load_progress(
            token,
            &format!("Decoding {}", model_variant.loading_name()),
            5,
        )
        .map_err(|error| js_error(&error))?;

        let runtime = Rc::new(ModelRuntime {
            ontology: EmbeddedOntology::load().map_err(|error| error.to_string())?,
            generator: CountedMorganGenerator::new(),
            model: PackedModelSet::from_archives_with_shared(
                shared_archive.as_deref(),
                pathway_archive.as_slice(),
                superclass_archive.as_slice(),
                class_archive.as_slice(),
                PackedModelVariant::Q4Kernel,
            )
            .map_err(|error| error.to_string())?,
            thresholds,
        });

        post_load_progress(
            token,
            &format!("{} ready", model_variant.display_name()),
            MODEL_LOAD_TOTAL,
        )
        .map_err(|error| js_error(&error))?;
        RUNTIMES.with(|slot| {
            let mut runtimes = slot.borrow_mut();
            if let Some((_, cached_runtime)) = runtimes
                .iter_mut()
                .find(|(variant, _)| *variant == model_variant)
            {
                *cached_runtime = runtime.clone();
            } else {
                runtimes.push((model_variant, runtime.clone()));
            }
        });
        Ok(runtime)
    }

    fn model_thresholds_url(model_variant: WebModelVariant) -> String {
        format!("/models/{}/thresholds.json", model_variant.slug())
    }

    fn model_shared_url(model_variant: WebModelVariant) -> String {
        format!(
            "/models/{}/shared/shared.q4-kernel.npz",
            model_variant.slug()
        )
    }

    fn model_pathway_url(model_variant: WebModelVariant) -> String {
        format!(
            "/models/{}/pathway/pathway.q4-kernel.npz",
            model_variant.slug()
        )
    }

    fn model_superclass_url(model_variant: WebModelVariant) -> String {
        format!(
            "/models/{}/superclass/superclass.q4-kernel.npz",
            model_variant.slug()
        )
    }

    fn model_class_url(model_variant: WebModelVariant) -> String {
        format!("/models/{}/class/class.q4-kernel.npz", model_variant.slug())
    }

    fn post_load_progress(token: u64, label: &str, completed: usize) -> Result<(), JsValue> {
        post_response(&WebWorkerResponse::Progress {
            token,
            label: String::from(label),
            completed,
            total: MODEL_LOAD_TOTAL,
        })
    }

    async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
        let response = fetch_response(url).await?;
        let buffer = JsFuture::from(response.array_buffer().map_err(|error| js_error(&error))?)
            .await
            .map_err(|error| js_error(&error))?;
        Ok(js_sys::Uint8Array::new(&buffer).to_vec())
    }

    async fn fetch_optional_bytes(url: &str) -> Result<Option<Vec<u8>>, String> {
        let Some(response) = fetch_optional_response(url).await? else {
            return Ok(None);
        };
        let buffer = JsFuture::from(response.array_buffer().map_err(|error| js_error(&error))?)
            .await
            .map_err(|error| js_error(&error))?;
        Ok(Some(js_sys::Uint8Array::new(&buffer).to_vec()))
    }

    async fn fetch_json<T>(url: &str) -> Result<T, String>
    where
        T: serde::de::DeserializeOwned,
    {
        let response = fetch_response(url).await?;
        let text = JsFuture::from(response.text().map_err(|error| js_error(&error))?)
            .await
            .map_err(|error| js_error(&error))?
            .as_string()
            .ok_or_else(|| format!("response for {url} was not text"))?;
        serde_json::from_str(&text).map_err(|error| format!("invalid JSON at {url}: {error}"))
    }

    async fn fetch_response(url: &str) -> Result<Response, String> {
        let response = JsFuture::from(worker_scope().fetch_with_str(url))
            .await
            .map_err(|error| js_error(&error))?
            .dyn_into::<Response>()
            .map_err(|_| format!("fetch for {url} did not return a response"))?;
        if !response.ok() {
            return Err(format!("failed to fetch {url}: HTTP {}", response.status()));
        }
        Ok(response)
    }

    async fn fetch_optional_response(url: &str) -> Result<Option<Response>, String> {
        let response = JsFuture::from(worker_scope().fetch_with_str(url))
            .await
            .map_err(|error| js_error(&error))?
            .dyn_into::<Response>()
            .map_err(|_| format!("fetch for {url} did not return a response"))?;
        if response.status() == 404 {
            return Ok(None);
        }
        if !response.ok() {
            return Err(format!("failed to fetch {url}: HTTP {}", response.status()));
        }
        Ok(Some(response))
    }

    fn is_stale(token: u64) -> bool {
        ACTIVE_TOKEN.with(|active| active.get() != token)
    }

    fn post_response(response: &WebWorkerResponse) -> Result<(), JsValue> {
        let payload = serde_wasm_bindgen::to_value(response)
            .map_err(|error| JsValue::from_str(&format!("invalid worker response: {error}")))?;
        worker_scope().post_message(&payload)
    }

    fn worker_scope() -> DedicatedWorkerGlobalScope {
        global().unchecked_into::<DedicatedWorkerGlobalScope>()
    }

    fn js_error(error: &JsValue) -> String {
        error
            .as_string()
            .filter(|message| !message.is_empty())
            .unwrap_or_else(|| format!("{error:?}"))
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod app {}

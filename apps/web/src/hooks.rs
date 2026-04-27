use std::{cell::Cell, rc::Rc};

use dioxus::prelude::*;
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, closure::Closure};
#[cfg(target_arch = "wasm32")]
use web_sys::{HtmlTextAreaElement, KeyboardEvent};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransientMessage {
    pub id: u64,
    pub text: String,
}

#[derive(Clone, Copy)]
pub struct TransientMessageHandle {
    message: Signal<Option<TransientMessage>>,
    next_id: Signal<u64>,
}

impl TransientMessageHandle {
    pub fn current(self) -> Option<TransientMessage> {
        (self.message)()
    }

    pub fn clear(mut self) {
        self.message.set(None);
    }

    pub fn show(mut self, text: String) {
        let next_id = (self.next_id)().wrapping_add(1);
        self.next_id.set(next_id);
        self.message
            .set(Some(TransientMessage { id: next_id, text }));
    }
}

pub fn use_transient_message(clear_ms: i32) -> TransientMessageHandle {
    let message = use_signal(|| None::<TransientMessage>);
    let next_id = use_signal(|| 0_u64);
    let timeout_id = use_hook(|| Rc::new(Cell::new(None::<i32>)));

    {
        let message_signal = message;
        let timeout_id = timeout_id.clone();
        use_effect(move || {
            let has_message = message_signal.read().is_some();
            schedule_message_clear(has_message, message_signal, &timeout_id, clear_ms);
        });
    }

    {
        let timeout_id = timeout_id.clone();
        dioxus::core::use_drop(move || clear_message_timeout(&timeout_id));
    }

    TransientMessageHandle { message, next_id }
}

pub fn use_entry_keyboard_navigation(
    entry_count: usize,
    on_previous: impl Fn() + 'static,
    on_next: impl Fn() + 'static,
) {
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (entry_count, &on_previous, &on_next);

    #[cfg(target_arch = "wasm32")]
    {
        let entry_count_ref = use_hook(|| Rc::new(Cell::new(0usize)));
        entry_count_ref.set(entry_count);

        let previous_action = use_hook(|| Rc::new(RefCell::new(Box::new(|| {}) as Box<dyn Fn()>)));
        let next_action = use_hook(|| Rc::new(RefCell::new(Box::new(|| {}) as Box<dyn Fn()>)));
        *previous_action.borrow_mut() = Box::new(on_previous);
        *next_action.borrow_mut() = Box::new(on_next);

        let listener =
            use_hook(|| Rc::new(RefCell::new(None::<Closure<dyn FnMut(KeyboardEvent)>>)));

        {
            let entry_count_ref = entry_count_ref.clone();
            let previous_action = previous_action.clone();
            let next_action = next_action.clone();
            let listener = listener.clone();
            use_effect(move || {
                if listener.borrow().is_some() {
                    return;
                }

                let Some(window) = web_sys::window() else {
                    return;
                };
                let Some(document) = window.document() else {
                    return;
                };
                let listener_entry_count = entry_count_ref.clone();
                let listener_previous_action = previous_action.clone();
                let listener_next_action = next_action.clone();

                let closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
                    if listener_entry_count.get() <= 1 {
                        return;
                    }
                    if event.alt_key() || event.ctrl_key() || event.meta_key() {
                        return;
                    }
                    if document
                        .active_element()
                        .and_then(|element| element.dyn_into::<HtmlTextAreaElement>().ok())
                        .is_some()
                    {
                        return;
                    }

                    match event.key().as_str() {
                        "ArrowLeft" | "ArrowUp" => {
                            event.prevent_default();
                            let callback = listener_previous_action.borrow();
                            (*callback)();
                        }
                        "ArrowRight" | "ArrowDown" => {
                            event.prevent_default();
                            let callback = listener_next_action.borrow();
                            (*callback)();
                        }
                        _ => {}
                    }
                }) as Box<dyn FnMut(_)>);

                let _ = window
                    .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref());
                *listener.borrow_mut() = Some(closure);
            });
        }

        {
            let listener = listener.clone();
            dioxus::core::use_drop(move || {
                let Some(window) = web_sys::window() else {
                    return;
                };
                if let Some(closure) = listener.borrow_mut().take() {
                    let _ = window.remove_event_listener_with_callback(
                        "keydown",
                        closure.as_ref().unchecked_ref(),
                    );
                }
            });
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn schedule_message_clear(
    has_message: bool,
    mut message_signal: Signal<Option<TransientMessage>>,
    timeout_id: &Rc<Cell<Option<i32>>>,
    clear_ms: i32,
) {
    clear_message_timeout(timeout_id);
    if !has_message {
        return;
    }

    let callback_timeout_id = timeout_id.clone();
    let callback = Closure::once_into_js(move || {
        callback_timeout_id.set(None);
        message_signal.set(None);
    });
    if let Some(window) = web_sys::window()
        && let Ok(timeout_id_value) = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            callback.unchecked_ref(),
            clear_ms,
        )
    {
        timeout_id.set(Some(timeout_id_value));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn schedule_message_clear(
    _has_message: bool,
    _message_signal: Signal<Option<TransientMessage>>,
    timeout_id: &Rc<Cell<Option<i32>>>,
    _clear_ms: i32,
) {
    clear_message_timeout(timeout_id);
}

#[cfg(target_arch = "wasm32")]
fn clear_message_timeout(timeout_id: &Cell<Option<i32>>) {
    if let Some(timeout_id) = timeout_id.take()
        && let Some(window) = web_sys::window()
    {
        window.clear_timeout_with_handle(timeout_id);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn clear_message_timeout(_timeout_id: &Cell<Option<i32>>) {}

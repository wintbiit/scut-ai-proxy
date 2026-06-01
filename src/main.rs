mod chat3;
mod config;
mod openai;
mod planner;
mod reasoning;
mod sse;

use async_stream::stream;
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{get, post},
};
use chat3::{Chat3Client, completion_response};
use config::Config;
use futures_util::StreamExt;
use openai::{
    ChatChoice, ChatChunkChoice, ChatChunkDelta, ChatChunkToolCall, ChatChunkToolCallFunction,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    ResponseToolCall,
};
use serde_json::{Value, json};
use std::time::Instant;
use std::{convert::Infallible, net::SocketAddr, sync::Arc};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
struct AppState {
    chat3: Chat3Client,
    config: Config,
}

#[tokio::main]
async fn main() -> anyhow_free::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Config::from_env();
    let addr: SocketAddr = config
        .bind_addr
        .parse()
        .expect("BIND_ADDR must be a valid socket address");
    let chat3 = Chat3Client::new(config.chat3_base_url.clone(), config.request_timeout)?;
    let state = Arc::new(AppState { chat3, config });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    tracing::info!("listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

async fn models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let auth = match authorization(&headers) {
        Ok(auth) => auth,
        Err(response) => return response,
    };

    let started = Instant::now();
    match state.chat3.models(auth).await {
        Ok(value) => with_proxy_headers(
            Json(value).into_response(),
            "models",
            started.elapsed().as_millis(),
        ),
        Err(error) => upstream_error(error),
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let auth = match authorization(&headers) {
        Ok(auth) => auth.to_string(),
        Err(response) => return response,
    };

    if planner::should_plan(&request)
        && planner::tool_result_count(&request) < state.config.planner_max_tool_rounds
    {
        return plan_tool_calls(state, &auth, request).await;
    }

    if request.stream {
        stream_chat(state, auth, request).await
    } else {
        collect_chat(state, &auth, request).await
    }
}

async fn collect_chat(
    state: Arc<AppState>,
    auth: &str,
    request: ChatCompletionRequest,
) -> Response {
    let requested_model = request.model.clone();
    let started = Instant::now();
    match state.chat3.chat_collect(auth, request).await {
        Ok(collected) => {
            let content = reasoning::clean_reasoning(&collected.raw_content);
            with_proxy_headers(
                Json(completion_response(
                    if collected.model.is_empty() {
                        requested_model
                    } else {
                        collected.model
                    },
                    content,
                    collected.id,
                    collected.created,
                ))
                .into_response(),
                "chat_collect",
                started.elapsed().as_millis(),
            )
        }
        Err(error) => upstream_error(error),
    }
}

async fn stream_chat(
    state: Arc<AppState>,
    auth: String,
    request: ChatCompletionRequest,
) -> Response {
    let stream_result = state.chat3.chat_stream(&auth, request.clone()).await;
    let upstream = match stream_result {
        Ok(stream) => stream,
        Err(error) => return upstream_error(error),
    };

    let model = request.model.clone();
    let events = stream! {
        let mut parser = sse::SseParser::default();
        futures_util::pin_mut!(upstream);
        while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(bytes) => {
                    for event in parser.push(&bytes) {
                        if event.is_done() {
                            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                            continue;
                        }
                        if let Some(content) = event_content(&event.data) {
                            let cleaned = reasoning::clean_reasoning(&content);
                            if !cleaned.is_empty() {
                                let chunk = chunk_response(&model, cleaned, None);
                                yield Ok(Event::default().json_data(chunk).expect("chunk json serializes"));
                            }
                        }
                    }
                }
                Err(error) => {
                    let err = json!({"error":{"message": error.to_string(), "type":"upstream_error", "code": null}});
                    yield Ok(Event::default().json_data(err).expect("error json serializes"));
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
        for event in parser.finish() {
            if let Some(content) = event_content(&event.data) {
                let cleaned = reasoning::clean_reasoning(&content);
                if !cleaned.is_empty() {
                    let chunk = chunk_response(&model, cleaned, None);
                    yield Ok(Event::default().json_data(chunk).expect("chunk json serializes"));
                }
            }
        }
    };

    Sse::new(events).into_response()
}

async fn plan_tool_calls(
    state: Arc<AppState>,
    auth: &str,
    request: ChatCompletionRequest,
) -> Response {
    let Some(tools) = request.tools.clone() else {
        return openai::error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "tools are required for tool planning",
            Some("missing_tools".to_string()),
        );
    };

    let started = Instant::now();
    let mut repair: Option<String> = None;
    let mut last_error = None;
    for attempt in 0..=state.config.planner_repair_attempts {
        let planner_request = planner::planner_request(&request, repair.as_deref());
        let collected = match state.chat3.chat_collect(auth, planner_request).await {
            Ok(collected) => collected,
            Err(error) => return upstream_error(error),
        };
        let raw = reasoning::clean_reasoning(&collected.raw_content);
        match planner::parse_and_validate(&raw, &tools, &request.tool_choice) {
            Ok(decision) => {
                if planner::tool_result_count(&request) > 0
                    && is_unrequested_pod_metrics_call(&request, &decision)
                {
                    tracing::info!(
                        "tool planner requested pod metrics without a resource-usage query; generating final chat response"
                    );
                    return final_chat_after_tools(state, auth, request).await;
                }
                if matches!(decision, planner::PlannerDecision::Final { .. })
                    && planner::tool_result_count(&request) > 0
                {
                    tracing::info!(
                        "tool planner determined no more tools are needed; generating final chat response"
                    );
                    return final_chat_after_tools(state, auth, request).await;
                }
                if matches!(decision, planner::PlannerDecision::Final { .. })
                    && let Some(required_decision) =
                        planner::required_tool_decision(&request, &tools)
                {
                    tracing::warn!(
                        "tool planner returned final answer for a live-state request; forcing tool call"
                    );
                    return with_proxy_headers(
                        planner_decision_response(
                            request.model.clone(),
                            required_decision,
                            request.stream,
                        ),
                        planner_mode("tool_planner_required", request.stream),
                        started.elapsed().as_millis(),
                    );
                }
                return with_proxy_headers(
                    planner_decision_response(request.model.clone(), decision, request.stream),
                    planner_mode("tool_planner", request.stream),
                    started.elapsed().as_millis(),
                );
            }
            Err(error) => {
                let message = error.to_string();
                tracing::warn!(attempt, error = %message, "tool planner failed");
                repair = Some(message.clone());
                last_error = Some(message);
            }
        }
    }

    if let Some(decision) = planner::fallback_decision(&request, &tools) {
        tracing::warn!(
            error = %last_error.as_deref().unwrap_or("unknown error"),
            "tool planner falling back to deterministic tool selection"
        );
        return with_proxy_headers(
            planner_decision_response(request.model.clone(), decision, request.stream),
            planner_mode("tool_planner_fallback", request.stream),
            started.elapsed().as_millis(),
        );
    }

    if planner::tool_result_count(&request) > 0 {
        tracing::warn!(
            error = %last_error.as_deref().unwrap_or("unknown error"),
            "tool planner failed after tool results; falling back to final chat response"
        );
        return final_chat_after_tools(state, auth, request).await;
    }

    openai::error_response(
        StatusCode::BAD_GATEWAY,
        "upstream_error",
        format!(
            "tool planner failed: {}",
            last_error.unwrap_or_else(|| "unknown error".to_string())
        ),
        Some("tool_planner_failed".to_string()),
    )
}

async fn final_chat_after_tools(
    state: Arc<AppState>,
    auth: &str,
    request: ChatCompletionRequest,
) -> Response {
    let stream_response = request.stream;
    let requested_model = request.model.clone();
    let started = Instant::now();
    match state.chat3.chat_collect(auth, request).await {
        Ok(collected) => {
            let content = reasoning::clean_reasoning(&collected.raw_content);
            let model = if collected.model.is_empty() {
                requested_model
            } else {
                collected.model
            };
            let id = if collected.id.is_empty() {
                format!("chatcmpl-{}", chrono::Utc::now().timestamp_millis())
            } else {
                collected.id
            };
            let created = if collected.created == 0 {
                chrono::Utc::now().timestamp()
            } else {
                collected.created
            };
            let response = if stream_response {
                final_text_stream_response(id, created, model, content)
            } else {
                Json(completion_response(model, content, id, created)).into_response()
            };
            with_proxy_headers(
                response,
                planner_mode("tool_planner_final_chat", stream_response),
                started.elapsed().as_millis(),
            )
        }
        Err(error) => upstream_error(error),
    }
}

fn is_unrequested_pod_metrics_call(
    request: &ChatCompletionRequest,
    decision: &planner::PlannerDecision,
) -> bool {
    if user_asked_for_resource_usage(request) {
        return false;
    }
    matches!(
        decision,
        planner::PlannerDecision::ToolCalls { calls }
            if calls.iter().any(|call| call.name == "pods_top")
    )
}

fn user_asked_for_resource_usage(request: &ChatCompletionRequest) -> bool {
    let Some(text) = request.messages.iter().rev().find_map(|message| {
        if message.role != "user" {
            return None;
        }
        let content = message.content.as_ref()?;
        match content {
            Value::String(text) => Some(text.to_lowercase()),
            other => Some(other.to_string().to_lowercase()),
        }
    }) else {
        return false;
    };

    [
        "cpu",
        "memory",
        "top",
        "内存",
        "资源",
        "占用",
        "负载",
        "用量",
        "使用率",
    ]
    .iter()
    .any(|keyword| text.contains(keyword))
}

fn planner_mode(base: &'static str, stream_response: bool) -> &'static str {
    match (base, stream_response) {
        ("tool_planner", true) => "tool_planner_stream",
        ("tool_planner_required", true) => "tool_planner_required_stream",
        ("tool_planner_fallback", true) => "tool_planner_fallback_stream",
        _ => base,
    }
}

fn planner_decision_response(
    model: String,
    decision: planner::PlannerDecision,
    stream_response: bool,
) -> Response {
    let created = chrono::Utc::now().timestamp();
    let id = format!("chatcmpl-{}", chrono::Utc::now().timestamp_millis());

    match decision {
        planner::PlannerDecision::ToolCalls { .. } => {
            let tool_calls = planner::response_tool_calls(decision).unwrap_or_default();
            if stream_response {
                return tool_calls_stream_response(id, created, model, tool_calls);
            }
            Json(ChatCompletionResponse {
                id,
                object: "chat.completion".to_string(),
                created,
                model,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: None,
                        name: None,
                        tool_call_id: None,
                        tool_calls: Some(tool_calls),
                    },
                    finish_reason: "tool_calls".to_string(),
                }],
            })
            .into_response()
        }
        planner::PlannerDecision::Final { content } => {
            if stream_response {
                return final_text_stream_response(id, created, model, content);
            }
            Json(completion_response(model, content, id, created)).into_response()
        }
    }
}

fn tool_calls_stream_response(
    id: String,
    created: i64,
    model: String,
    tool_calls: Vec<ResponseToolCall>,
) -> Response {
    let events = stream! {
        for (index, tool_call) in tool_calls.into_iter().enumerate() {
            let chunk = tool_call_chunk_response(&id, created, &model, index as u32, tool_call);
            yield Ok::<Event, Infallible>(Event::default().json_data(chunk).expect("tool call chunk serializes"));
        }
        let done = finish_chunk_response(&id, created, &model, "tool_calls");
        yield Ok(Event::default().json_data(done).expect("finish chunk serializes"));
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(events).into_response()
}

fn final_text_stream_response(
    id: String,
    created: i64,
    model: String,
    content: String,
) -> Response {
    let events = stream! {
        if !content.is_empty() {
            let chunk = content_chunk_response(&id, created, &model, content);
            yield Ok::<Event, Infallible>(Event::default().json_data(chunk).expect("content chunk serializes"));
        }
        let done = finish_chunk_response(&id, created, &model, "stop");
        yield Ok(Event::default().json_data(done).expect("finish chunk serializes"));
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(events).into_response()
}

fn authorization(headers: &HeaderMap) -> Result<&str, Response> {
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return Err(openai::error_response(
            StatusCode::UNAUTHORIZED,
            "invalid_request_error",
            "missing Authorization header",
            Some("missing_authorization".to_string()),
        ));
    };
    let Ok(value) = value.to_str() else {
        return Err(openai::error_response(
            StatusCode::UNAUTHORIZED,
            "invalid_request_error",
            "invalid Authorization header",
            Some("invalid_authorization".to_string()),
        ));
    };
    if !value.starts_with("Bearer ") {
        return Err(openai::error_response(
            StatusCode::UNAUTHORIZED,
            "invalid_request_error",
            "Authorization must use Bearer scheme",
            Some("invalid_authorization_scheme".to_string()),
        ));
    }
    Ok(value)
}

fn upstream_error(error: chat3::Chat3Error) -> Response {
    let (status, message): (StatusCode, String) = error.into();
    let public_status = if status.is_server_error()
        || status == StatusCode::UNAUTHORIZED
        || status == StatusCode::FORBIDDEN
    {
        status
    } else {
        StatusCode::BAD_GATEWAY
    };
    openai::error_response(
        public_status,
        "upstream_error",
        message,
        Some("chat3_error".to_string()),
    )
}

fn with_proxy_headers(mut response: Response, mode: &'static str, elapsed_ms: u128) -> Response {
    let headers = response.headers_mut();
    headers.insert("x-scut-proxy-mode", HeaderValue::from_static(mode));
    if let Ok(value) = HeaderValue::from_str(&elapsed_ms.to_string()) {
        headers.insert("x-scut-proxy-upstream-ms", value);
    }
    response
}

fn event_content(data: &str) -> Option<String> {
    let value: Value = serde_json::from_str(data).ok()?;
    value
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn chunk_response(
    model: &str,
    content: String,
    finish_reason: Option<String>,
) -> ChatCompletionChunk {
    let id = format!("chatcmpl-{}", chrono::Utc::now().timestamp_millis());
    let created = chrono::Utc::now().timestamp();
    content_chunk_with_finish_response(&id, created, model, content, finish_reason)
}

fn content_chunk_response(
    id: &str,
    created: i64,
    model: &str,
    content: String,
) -> ChatCompletionChunk {
    content_chunk_with_finish_response(id, created, model, content, None)
}

fn content_chunk_with_finish_response(
    id: &str,
    created: i64,
    model: &str,
    content: String,
    finish_reason: Option<String>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatChunkDelta {
                role: None,
                content: Some(content),
                tool_calls: None,
            },
            finish_reason,
        }],
    }
}

fn tool_call_chunk_response(
    id: &str,
    created: i64,
    model: &str,
    index: u32,
    tool_call: ResponseToolCall,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatChunkDelta {
                role: Some("assistant".to_string()),
                content: None,
                tool_calls: Some(vec![ChatChunkToolCall {
                    index,
                    id: Some(tool_call.id),
                    kind: Some(tool_call.kind),
                    function: ChatChunkToolCallFunction {
                        name: Some(tool_call.function.name),
                        arguments: Some(tool_call.function.arguments),
                    },
                }]),
            },
            finish_reason: None,
        }],
    }
}

fn finish_chunk_response(
    id: &str,
    created: i64,
    model: &str,
    finish_reason: &str,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatChunkDelta {
                role: None,
                content: None,
                tool_calls: None,
            },
            finish_reason: Some(finish_reason.to_string()),
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openai::{ResponseToolCall, ResponseToolCallFunction};

    #[test]
    fn serializes_openai_streaming_tool_call_chunk() {
        let chunk = tool_call_chunk_response(
            "chatcmpl-test",
            123,
            "test-model",
            0,
            ResponseToolCall {
                id: "call_000000".to_string(),
                kind: "function".to_string(),
                function: ResponseToolCallFunction {
                    name: "pods_list_in_namespace".to_string(),
                    arguments: r#"{"namespace":"store"}"#.to_string(),
                },
            },
        );

        let value = serde_json::to_value(chunk).unwrap();

        assert_eq!(value["object"], "chat.completion.chunk");
        assert_eq!(
            value["choices"][0]["finish_reason"],
            serde_json::Value::Null
        );
        assert_eq!(value["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            value["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "pods_list_in_namespace"
        );
        assert_eq!(
            value["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            r#"{"namespace":"store"}"#
        );
        assert!(value["choices"][0]["delta"].get("content").is_none());
    }

    #[test]
    fn serializes_tool_call_finish_chunk() {
        let chunk = finish_chunk_response("chatcmpl-test", 123, "test-model", "tool_calls");
        let value = serde_json::to_value(chunk).unwrap();

        assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
        assert!(value["choices"][0]["delta"].get("tool_calls").is_none());
        assert!(value["choices"][0]["delta"].get("content").is_none());
    }

    #[test]
    fn marks_streaming_planner_mode() {
        assert_eq!(planner_mode("tool_planner", true), "tool_planner_stream");
        assert_eq!(planner_mode("tool_planner", false), "tool_planner");
        assert_eq!(
            planner_mode("tool_planner_fallback", true),
            "tool_planner_fallback_stream"
        );
    }
}

mod anyhow_free {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
}

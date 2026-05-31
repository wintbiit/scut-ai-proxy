use crate::{
    openai::{ChatCompletionRequest, ChatCompletionResponse},
    sse::SseEvent,
};
use bytes::Bytes;
use futures_util::{StreamExt, stream::BoxStream};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::{Value, json};
use thiserror::Error;

const MAX_TOOL_RESULT_CHARS: usize = 8_000;

#[derive(Clone)]
pub struct Chat3Client {
    http: reqwest::Client,
    base_url: String,
}

impl Chat3Client {
    pub fn new(base_url: String, timeout: std::time::Duration) -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .tcp_nodelay(true)
            .pool_max_idle_per_host(16)
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .build()?;
        Ok(Self { http, base_url })
    }

    pub async fn models(&self, auth: &str) -> Result<Value, Chat3Error> {
        let response = self
            .http
            .get(format!("{}/models", self.base_url))
            .headers(auth_headers(auth)?)
            .send()
            .await?;
        json_or_error(response).await
    }

    pub async fn chat_stream(
        &self,
        auth: &str,
        mut request: ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<Bytes, reqwest::Error>>, Chat3Error> {
        request.stream = true;
        normalize_tool_messages_for_upstream(&mut request);
        let response = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .headers(auth_headers(auth)?)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Chat3Error::UpstreamStatus {
                status: response.status().as_u16(),
                body: response.text().await.unwrap_or_default(),
            });
        }

        Ok(response.bytes_stream().boxed())
    }

    pub async fn chat_collect(
        &self,
        auth: &str,
        request: ChatCompletionRequest,
    ) -> Result<CollectedChat, Chat3Error> {
        let stream = self.chat_stream(auth, request).await?;
        futures_util::pin_mut!(stream);

        let mut parser = crate::sse::SseParser::default();
        let mut raw_content = String::new();
        let mut id = String::new();
        let mut model = String::new();
        let mut created = 0_i64;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            for event in parser.push(&chunk) {
                if event.is_done() {
                    continue;
                }
                if let Some(delta) = parse_delta(&event) {
                    if id.is_empty() {
                        id = delta.id.unwrap_or_default();
                    }
                    if model.is_empty() {
                        model = delta.model.unwrap_or_default();
                    }
                    if created == 0 {
                        created = delta.created.unwrap_or_default();
                    }
                    if let Some(content) = delta.content {
                        raw_content.push_str(&content);
                    }
                }
            }
        }

        for event in parser.finish() {
            if let Some(delta) = parse_delta(&event) {
                if let Some(content) = delta.content {
                    raw_content.push_str(&content);
                }
            }
        }

        Ok(CollectedChat {
            id,
            model,
            created,
            raw_content,
        })
    }
}

fn normalize_tool_messages_for_upstream(request: &mut ChatCompletionRequest) {
    let mut saw_tool_protocol = false;
    for message in &mut request.messages {
        match message.role.as_str() {
            "tool" => {
                saw_tool_protocol = true;
                let name = message
                    .name
                    .as_deref()
                    .or(message.tool_call_id.as_deref())
                    .unwrap_or("tool");
                let content = message
                    .content
                    .as_ref()
                    .map(stringify_message_content)
                    .unwrap_or_default();
                let content = truncate_for_upstream(&content, MAX_TOOL_RESULT_CHARS);
                message.role = "user".to_string();
                message.content = Some(Value::String(format!(
                    "Tool result from {name}:\n{content}"
                )));
                message.name = None;
                message.tool_call_id = None;
                message.tool_calls = None;
            }
            "assistant" => {
                if let Some(tool_calls) = message.tool_calls.take() {
                    saw_tool_protocol = true;
                    let existing = message
                        .content
                        .as_ref()
                        .map(stringify_message_content)
                        .unwrap_or_default();
                    let calls = tool_calls
                        .iter()
                        .map(|call| format!("{}({})", call.function.name, call.function.arguments))
                        .collect::<Vec<_>>()
                        .join("\n");
                    let content = if existing.trim().is_empty() {
                        format!("Tool calls requested:\n{calls}")
                    } else {
                        format!("{existing}\n\nTool calls requested:\n{calls}")
                    };
                    message.content = Some(Value::String(content));
                }
            }
            _ => {}
        }
    }
    if saw_tool_protocol {
        request.tools = None;
        request.tool_choice = None;
    }
}

fn stringify_message_content(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn truncate_for_upstream(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }

    let cutoff = content
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or(content.len());
    format!(
        "{}\n\n[scut-ai-proxy truncated tool result: original_chars={}, kept_chars={}]",
        &content[..cutoff],
        content.chars().count(),
        max_chars
    )
}

#[derive(Debug)]
pub struct CollectedChat {
    pub id: String,
    pub model: String,
    pub created: i64,
    pub raw_content: String,
}

#[derive(Debug)]
struct DeltaEvent {
    id: Option<String>,
    model: Option<String>,
    created: Option<i64>,
    content: Option<String>,
}

fn parse_delta(event: &SseEvent) -> Option<DeltaEvent> {
    let value: Value = serde_json::from_str(&event.data).ok()?;
    let content = value
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    Some(DeltaEvent {
        id: value
            .get("id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        model: value
            .get("model")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        created: value.get("created").and_then(Value::as_i64),
        content,
    })
}

fn auth_headers(auth: &str) -> Result<HeaderMap, Chat3Error> {
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(auth).map_err(|_| Chat3Error::InvalidAuthHeader)?,
    );
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(headers)
}

async fn json_or_error(response: reqwest::Response) -> Result<Value, Chat3Error> {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(Chat3Error::UpstreamStatus {
            status: status.as_u16(),
            body,
        });
    }
    serde_json::from_str(&body).map_err(|source| Chat3Error::InvalidJson { source, body })
}

#[derive(Debug, Error)]
pub enum Chat3Error {
    #[error("invalid authorization header")]
    InvalidAuthHeader,
    #[error("upstream request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("upstream returned status {status}: {body}")]
    UpstreamStatus { status: u16, body: String },
    #[error("upstream returned invalid json: {source}; body={body}")]
    InvalidJson {
        source: serde_json::Error,
        body: String,
    },
}

impl From<Chat3Error> for (http::StatusCode, String) {
    fn from(error: Chat3Error) -> Self {
        match error {
            Chat3Error::UpstreamStatus { status, body } => (
                http::StatusCode::from_u16(status).unwrap_or(http::StatusCode::BAD_GATEWAY),
                body,
            ),
            Chat3Error::InvalidAuthHeader => (
                http::StatusCode::UNAUTHORIZED,
                "invalid authorization header".to_string(),
            ),
            other => (http::StatusCode::BAD_GATEWAY, other.to_string()),
        }
    }
}

pub fn completion_response(
    model: String,
    content: String,
    id: String,
    created: i64,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: if id.is_empty() {
            format!("chatcmpl-{}", chrono::Utc::now().timestamp_millis())
        } else {
            id
        },
        object: "chat.completion".to_string(),
        created: if created == 0 {
            chrono::Utc::now().timestamp()
        } else {
            created
        },
        model,
        choices: vec![crate::openai::ChatChoice {
            index: 0,
            message: crate::openai::ChatMessage {
                role: "assistant".to_string(),
                content: Some(json!(content)),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            finish_reason: "stop".to_string(),
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{ChatMessage, ResponseToolCall, ResponseToolCallFunction};

    #[test]
    fn converts_tool_role_messages_to_user_text_for_chat3() {
        let mut request = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage {
                role: "tool".to_string(),
                content: Some(Value::String("{\"ok\":true}".to_string())),
                name: Some("query_prometheus".to_string()),
                tool_call_id: Some("call_1".to_string()),
                tool_calls: None,
            }],
            stream: false,
            temperature: None,
            top_p: None,
            max_tokens: None,
            tools: None,
            tool_choice: None,
        };

        normalize_tool_messages_for_upstream(&mut request);

        assert_eq!(request.messages[0].role, "user");
        assert_eq!(request.messages[0].name, None);
        assert_eq!(request.messages[0].tool_call_id, None);
        assert!(
            request.messages[0]
                .content
                .as_ref()
                .and_then(Value::as_str)
                .unwrap()
                .contains("Tool result from query_prometheus")
        );
    }

    #[test]
    fn truncates_large_tool_results_for_chat3() {
        let large = "x".repeat(MAX_TOOL_RESULT_CHARS + 100);
        let mut request = ChatCompletionRequest {
            model: "chat3".to_string(),
            messages: vec![ChatMessage {
                role: "tool".to_string(),
                content: Some(Value::String(large)),
                name: Some("pods_list_in_namespace".to_string()),
                tool_call_id: Some("call_1".to_string()),
                tool_calls: None,
            }],
            stream: false,
            temperature: None,
            top_p: None,
            max_tokens: None,
            tools: None,
            tool_choice: None,
        };

        normalize_tool_messages_for_upstream(&mut request);

        let content = request.messages[0]
            .content
            .as_ref()
            .and_then(Value::as_str)
            .unwrap();
        assert!(content.contains("Tool result from pods_list_in_namespace"));
        assert!(content.contains("truncated tool result"));
        assert!(content.len() < MAX_TOOL_RESULT_CHARS + 300);
        assert!(request.tools.is_none());
        assert!(request.tool_choice.is_none());
    }

    #[test]
    fn converts_assistant_tool_calls_to_text_for_chat3() {
        let mut request = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage {
                role: "assistant".to_string(),
                content: None,
                name: None,
                tool_call_id: None,
                tool_calls: Some(vec![ResponseToolCall {
                    id: "call_1".to_string(),
                    kind: "function".to_string(),
                    function: ResponseToolCallFunction {
                        name: "nodes_top".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
            }],
            stream: false,
            temperature: None,
            top_p: None,
            max_tokens: None,
            tools: None,
            tool_choice: None,
        };

        normalize_tool_messages_for_upstream(&mut request);

        assert!(request.messages[0].tool_calls.is_none());
        assert_eq!(
            request.messages[0].content.as_ref().and_then(Value::as_str),
            Some("Tool calls requested:\nnodes_top({})")
        );
    }
}

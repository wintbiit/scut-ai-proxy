use crate::{
    openai::{ChatCompletionRequest, ChatCompletionResponse},
    sse::SseEvent,
};
use bytes::Bytes;
use futures_util::{StreamExt, stream::BoxStream};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::{Value, json};
use thiserror::Error;

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

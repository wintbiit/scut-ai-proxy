use crate::openai::{
    ChatCompletionRequest, ChatMessage, ResponseToolCall, ResponseToolCallFunction, Tool,
    ToolChoice,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum PlannerDecision {
    ToolCalls { calls: Vec<PlannerCall> },
    Final { content: String },
}

#[derive(Debug, Deserialize)]
pub struct PlannerCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Error)]
pub enum PlannerError {
    #[error("planner output is not valid json: {0}")]
    InvalidJson(serde_json::Error),
    #[error("requested tool is not available: {0}")]
    UnknownTool(String),
    #[error("tool arguments for {0} must be a json object")]
    ArgumentsNotObject(String),
    #[error("tool {tool} missing required argument {argument}")]
    MissingRequired { tool: String, argument: String },
    #[error("tool {tool} argument {argument} must be {expected}")]
    TypeMismatch {
        tool: String,
        argument: String,
        expected: String,
    },
    #[error("tool_choice requires a tool call")]
    RequiredToolCall,
    #[error("tool_choice requires tool {0}")]
    SpecificToolRequired(String),
    #[error(
        "planner returned a final answer that promises tool use instead of returning tool_calls"
    )]
    PrematureFinal,
}

pub fn should_plan(request: &ChatCompletionRequest) -> bool {
    request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && !matches!(
            request.tool_choice.as_ref(),
            Some(ToolChoice::String(choice)) if choice == "none"
        )
}

pub fn planner_request(
    original: &ChatCompletionRequest,
    repair: Option<&str>,
) -> ChatCompletionRequest {
    let tools = original.tools.clone().unwrap_or_default();
    let mut messages = Vec::new();
    messages.push(ChatMessage::system(planner_prompt(
        &tools,
        &original.tool_choice,
        repair,
    )));
    messages.extend(original.messages.clone());

    ChatCompletionRequest {
        model: original.model.clone(),
        messages,
        stream: false,
        temperature: Some(0.0),
        top_p: Some(1.0),
        max_tokens: Some(512),
        tools: None,
        tool_choice: None,
    }
}

fn planner_prompt(tools: &[Tool], choice: &Option<ToolChoice>, repair: Option<&str>) -> String {
    let tool_specs = serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".to_string());
    let choice = serde_json::to_string(choice).unwrap_or_else(|_| "null".to_string());
    let repair_note = repair
        .map(|err| {
            format!("\nYour previous output was invalid: {err}\nReturn only corrected JSON.")
        })
        .unwrap_or_default();
    format!(
        r#"You are a tool-call planner for an OpenAI-compatible API gateway.
Return only a single JSON object. Do not use markdown. Do not explain.

Available tools:
{tool_specs}

tool_choice:
{choice}

If a tool should be called, return:
{{"type":"tool_calls","calls":[{{"name":"tool_name","arguments":{{}}}}]}}

If no tool is needed, return:
{{"type":"final","content":"your final answer"}}

Use only tool names from the available tools. Arguments must be a JSON object.
For tool_choice "auto", call a tool whenever the user asks to check, inspect, query,
search, list, diagnose, troubleshoot, verify current state, or use external/live
information. Do not answer that you will call a tool later; return tool_calls now.
Return "final" only when the user's request can be fully answered without any tool.
{repair_note}"#
    )
}

pub fn parse_and_validate(
    raw: &str,
    tools: &[Tool],
    choice: &Option<ToolChoice>,
) -> Result<PlannerDecision, PlannerError> {
    let decision: PlannerDecision = match serde_json::from_str(extract_json(raw)) {
        Ok(decision) => decision,
        Err(error) => {
            if let Some(decision) =
                parse_xml_style_tool_call(raw, tools).or_else(|| infer_named_tool_call(raw, tools))
            {
                decision
            } else {
                return Err(PlannerError::InvalidJson(error));
            }
        }
    };
    validate_decision(&decision, tools, choice)?;
    Ok(decision)
}

fn extract_json(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        &trimmed[start..=end]
    } else {
        trimmed
    }
}

fn validate_decision(
    decision: &PlannerDecision,
    tools: &[Tool],
    choice: &Option<ToolChoice>,
) -> Result<(), PlannerError> {
    let map = tools
        .iter()
        .map(|tool| (tool.function.name.clone(), tool))
        .collect::<HashMap<_, _>>();

    match choice {
        Some(ToolChoice::String(choice)) if choice == "required" => {
            if !matches!(decision, PlannerDecision::ToolCalls { calls } if !calls.is_empty()) {
                return Err(PlannerError::RequiredToolCall);
            }
        }
        Some(ToolChoice::Object(object)) => {
            if !matches!(decision, PlannerDecision::ToolCalls { calls } if calls.iter().any(|call| call.name == object.function.name))
            {
                return Err(PlannerError::SpecificToolRequired(
                    object.function.name.clone(),
                ));
            }
        }
        _ => {}
    }

    if let PlannerDecision::Final { content } = decision {
        if looks_like_deferred_tool_use(content, tools) {
            return Err(PlannerError::PrematureFinal);
        }
    }

    if let PlannerDecision::ToolCalls { calls } = decision {
        for call in calls {
            let Some(tool) = map.get(&call.name) else {
                return Err(PlannerError::UnknownTool(call.name.clone()));
            };
            validate_arguments(tool, &call.arguments)?;
        }
    }
    Ok(())
}

fn looks_like_deferred_tool_use(content: &str, tools: &[Tool]) -> bool {
    let normalized = content.to_lowercase();
    let deferred_markers = [
        "i will call",
        "i'll call",
        "i will use",
        "i'll use",
        "i will query",
        "i'll query",
        "i will check",
        "i'll check",
        "need to call",
        "need to use",
        "need to query",
        "calling tool",
        "call tool",
        "use tool",
        "query tool",
        "tool:",
        "tool：",
        "调用工具",
        "调用 ",
        "使用工具",
        "查询工具",
        "我将",
        "我会",
        "需要先",
        "请稍等",
        "正在查询",
        "将查询",
        "先查询",
    ];

    let promises_tool = deferred_markers
        .iter()
        .any(|marker| normalized.contains(marker));
    if !promises_tool {
        return false;
    }

    tools.iter().any(|tool| {
        let name = tool.function.name.to_lowercase();
        !name.is_empty() && normalized.contains(&name)
    }) || normalized.contains("tool")
        || normalized.contains("工具")
}

fn parse_xml_style_tool_call(raw: &str, tools: &[Tool]) -> Option<PlannerDecision> {
    for tool in tools {
        let name = &tool.function.name;
        let open = format!("<{name}");
        let Some(start) = raw.find(&open) else {
            continue;
        };
        let after_open = &raw[start + open.len()..];
        let Some(end) = after_open.find('>') else {
            continue;
        };
        let tag_body = after_open[..end].trim().trim_end_matches('/').trim();
        let mut arguments = parse_xml_attrs(tag_body);
        normalize_common_arguments(name, &mut arguments);
        return Some(PlannerDecision::ToolCalls {
            calls: vec![PlannerCall {
                name: name.clone(),
                arguments: Value::Object(arguments),
            }],
        });
    }
    None
}

fn parse_xml_attrs(input: &str) -> serde_json::Map<String, Value> {
    let mut attrs = serde_json::Map::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let key_start = i;
        while i < bytes.len()
            && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b'-' | b'.'))
        {
            i += 1;
        }
        if key_start == i {
            break;
        }
        let key = &input[key_start..i];
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            continue;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let quote = bytes[i];
        let value = if quote == b'"' || quote == b'\'' {
            i += 1;
            let value_start = i;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            let value = &input[value_start..i];
            if i < bytes.len() {
                i += 1;
            }
            value
        } else {
            let value_start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            &input[value_start..i]
        };
        attrs.insert(key.to_string(), coerce_attr_value(value));
    }
    attrs
}

fn coerce_attr_value(value: &str) -> Value {
    match value {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        _ => value
            .parse::<i64>()
            .map(|number| Value::Number(number.into()))
            .unwrap_or_else(|_| Value::String(value.to_string())),
    }
}

fn normalize_common_arguments(tool_name: &str, arguments: &mut serde_json::Map<String, Value>) {
    if tool_name == "resources_list" || tool_name == "resources_get" {
        if !arguments.contains_key("apiVersion") {
            let group = arguments
                .remove("group")
                .and_then(|v| v.as_str().map(ToOwned::to_owned));
            let version = arguments
                .remove("version")
                .and_then(|v| v.as_str().map(ToOwned::to_owned));
            if let Some(version) = version {
                let api_version = match group.as_deref() {
                    Some("") | Some("core") | None => version,
                    Some(group) => format!("{group}/{version}"),
                };
                arguments.insert("apiVersion".to_string(), Value::String(api_version));
            }
        }
        if !arguments.contains_key("kind") {
            if let Some(resource) = arguments
                .remove("resource")
                .and_then(|v| v.as_str().map(ToOwned::to_owned))
            {
                let kind = match resource.as_str() {
                    "deployments" | "deployment" => "Deployment",
                    "services" | "service" => "Service",
                    "pods" | "pod" => "Pod",
                    "ingresses" | "ingress" => "Ingress",
                    "statefulsets" | "statefulset" => "StatefulSet",
                    "daemonsets" | "daemonset" => "DaemonSet",
                    other => other,
                };
                arguments.insert("kind".to_string(), Value::String(kind.to_string()));
            }
        }
        arguments.remove("all_namespaces");
    }
}

fn infer_named_tool_call(raw: &str, tools: &[Tool]) -> Option<PlannerDecision> {
    let normalized = raw.to_lowercase();
    if !looks_like_deferred_tool_use(raw, tools) {
        return None;
    }
    tools.iter().find_map(|tool| {
        let name = tool.function.name.to_lowercase();
        if !name.is_empty() && normalized.contains(&name) {
            Some(PlannerDecision::ToolCalls {
                calls: vec![PlannerCall {
                    name: tool.function.name.clone(),
                    arguments: Value::Object(serde_json::Map::new()),
                }],
            })
        } else {
            None
        }
    })
}

fn validate_arguments(tool: &Tool, arguments: &Value) -> Result<(), PlannerError> {
    let Some(args) = arguments.as_object() else {
        return Err(PlannerError::ArgumentsNotObject(tool.function.name.clone()));
    };
    let params = &tool.function.parameters;
    if let Some(required) = params.get("required").and_then(Value::as_array) {
        for required_arg in required.iter().filter_map(Value::as_str) {
            if !args.contains_key(required_arg) {
                return Err(PlannerError::MissingRequired {
                    tool: tool.function.name.clone(),
                    argument: required_arg.to_string(),
                });
            }
        }
    }
    if let Some(properties) = params.get("properties").and_then(Value::as_object) {
        for (name, spec) in properties {
            let Some(value) = args.get(name) else {
                continue;
            };
            if let Some(expected) = spec.get("type").and_then(Value::as_str) {
                let ok = match expected {
                    "string" => value.is_string(),
                    "number" => value.is_number(),
                    "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
                    "boolean" => value.is_boolean(),
                    "object" => value.is_object(),
                    "array" => value.is_array(),
                    _ => true,
                };
                if !ok {
                    return Err(PlannerError::TypeMismatch {
                        tool: tool.function.name.clone(),
                        argument: name.clone(),
                        expected: expected.to_string(),
                    });
                }
            }
        }
    }
    Ok(())
}

pub fn response_tool_calls(decision: PlannerDecision) -> Option<Vec<ResponseToolCall>> {
    match decision {
        PlannerDecision::ToolCalls { calls } => Some(
            calls
                .into_iter()
                .enumerate()
                .map(|(idx, call)| ResponseToolCall {
                    id: format!("call_{idx:06}"),
                    kind: "function".to_string(),
                    function: ResponseToolCallFunction {
                        name: call.name,
                        arguments: serde_json::to_string(&call.arguments)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                })
                .collect(),
        ),
        PlannerDecision::Final { .. } => None,
    }
}

#[cfg(test)]
fn tool_schema(name: &str) -> Tool {
    Tool {
        kind: "function".to_string(),
        function: crate::openai::ToolFunction {
            name: name.to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_tool_call() {
        let tools = vec![tool_schema("get_weather")];
        let decision = parse_and_validate(
            r#"{"type":"tool_calls","calls":[{"name":"get_weather","arguments":{"city":"广州"}}]}"#,
            &tools,
            &Some(ToolChoice::String("auto".to_string())),
        )
        .unwrap();
        assert!(matches!(decision, PlannerDecision::ToolCalls { .. }));
    }

    #[test]
    fn rejects_missing_required() {
        let tools = vec![tool_schema("get_weather")];
        let err = parse_and_validate(
            r#"{"type":"tool_calls","calls":[{"name":"get_weather","arguments":{}}]}"#,
            &tools,
            &None,
        )
        .unwrap_err();
        assert!(matches!(err, PlannerError::MissingRequired { .. }));
    }

    #[test]
    fn respects_none_choice() {
        let req = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![],
            stream: false,
            temperature: None,
            top_p: None,
            max_tokens: None,
            tools: Some(vec![tool_schema("x")]),
            tool_choice: Some(ToolChoice::String("none".to_string())),
        };
        assert!(!should_plan(&req));
    }

    #[test]
    fn rejects_final_answer_that_defers_tool_call() {
        let tools = vec![tool_schema("nodes_top")];
        let err = parse_and_validate(
            r#"{"type":"final","content":"请稍等，我将调用工具 nodes_top 获取节点 CPU 信息。"}"#,
            &tools,
            &Some(ToolChoice::String("auto".to_string())),
        )
        .unwrap_err();
        assert!(matches!(err, PlannerError::PrematureFinal));
    }

    #[test]
    fn allows_final_answer_without_tool_promise() {
        let tools = vec![tool_schema("nodes_top")];
        let decision = parse_and_validate(
            r#"{"type":"final","content":"你好，我可以帮助你分析集群状态。"}"#,
            &tools,
            &Some(ToolChoice::String("auto".to_string())),
        )
        .unwrap();
        assert!(matches!(decision, PlannerDecision::Final { .. }));
    }

    #[test]
    fn parses_xml_style_tool_call() {
        let tools = vec![Tool {
            kind: "function".to_string(),
            function: crate::openai::ToolFunction {
                name: "resources_list".to_string(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "apiVersion": {"type": "string"},
                        "kind": {"type": "string"}
                    },
                    "required": ["apiVersion", "kind"]
                }),
            },
        }];
        let decision = parse_and_validate(
            r#"<resources_list group="apps" version="v1" resource="deployments" all_namespaces="true"></resources_list>"#,
            &tools,
            &Some(ToolChoice::String("auto".to_string())),
        )
        .unwrap();
        match decision {
            PlannerDecision::ToolCalls { calls } => {
                assert_eq!(calls[0].name, "resources_list");
                assert_eq!(calls[0].arguments["apiVersion"], "apps/v1");
                assert_eq!(calls[0].arguments["kind"], "Deployment");
            }
            _ => panic!("expected tool call"),
        }
    }

    #[test]
    fn infers_named_tool_call_from_deferred_text() {
        let tools = vec![Tool {
            kind: "function".to_string(),
            function: crate::openai::ToolFunction {
                name: "nodes_top".to_string(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        }];
        let decision = parse_and_validate(
            "结论：正在查询节点 CPU 使用情况，将调用 nodes_top 工具。",
            &tools,
            &Some(ToolChoice::String("auto".to_string())),
        )
        .unwrap();
        assert!(matches!(decision, PlannerDecision::ToolCalls { .. }));
    }
}

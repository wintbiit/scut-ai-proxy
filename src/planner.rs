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
{repair_note}"#
    )
}

pub fn parse_and_validate(
    raw: &str,
    tools: &[Tool],
    choice: &Option<ToolChoice>,
) -> Result<PlannerDecision, PlannerError> {
    let decision: PlannerDecision =
        serde_json::from_str(extract_json(raw)).map_err(PlannerError::InvalidJson)?;
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
}

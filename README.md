# SCUT AI Proxy

OpenAI-compatible gateway for `https://chat3.scut.edu.cn/api`.

## Run

```powershell
cargo run
```

Environment variables:

- `BIND_ADDR`, default `127.0.0.1:3000`
- `CHAT3_BASE_URL`, default `https://chat3.scut.edu.cn/api`
- `REQUEST_TIMEOUT_SECS`, default `120`
- `PLANNER_REPAIR_ATTEMPTS`, default `1`

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

The gateway does not manage API keys. Pass a valid chat3 token directly:

```powershell
$env:CHAT3_TOKEN = "..."
curl.exe http://127.0.0.1:3000/v1/models `
  -H "Authorization: Bearer $env:CHAT3_TOKEN"
```

Chat example:

```powershell
curl.exe http://127.0.0.1:3000/v1/chat/completions `
  -H "Authorization: Bearer $env:CHAT3_TOKEN" `
  -H "Content-Type: application/json" `
  -d '{
    "model": "deepseek_r1_32b_hnlg_released_version.DeepSeek-R1-32B",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

Tool calling is emulated. If `tools` are present and `tool_choice` is not `none`, the gateway asks chat3 to plan tool calls, validates the result, and returns OpenAI-style `tool_calls`. It does not execute tools.

## Performance Notes

The gateway adds response headers on non-streaming requests:

- `x-scut-proxy-mode`: `models`, `chat_collect`, or `tool_planner`
- `x-scut-proxy-upstream-ms`: elapsed upstream/proxy time in milliseconds

Important limits:

- Non-streaming chat is implemented by collecting chat3 streaming output, so it finishes only after the upstream model finishes.
- Emulated tool calling requires at least one extra LLM request for planning.
- DeepSeek-R1 style reasoning may delay visible output when reasoning is stripped.
- Use `cargo build --release` for deployment.

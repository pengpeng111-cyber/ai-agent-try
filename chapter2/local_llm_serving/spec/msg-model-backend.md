# msg_model 后端支持修改计划

> 目标文件：`chapter2/local_llm_serving/config.py`、`agent.py`、`main.py`
> 状态：已实施
> 日期：2026-09-09

## 1. 背景与需求

本地部署了一个 OpenAI 兼容的大模型服务（MSG_Model），希望通过现有的 tool-calling 框架（`ToolCallingAgent`）调用它，且不影响既有的 vLLM / Ollama 后端。

| # | 需求 |
|---|------|
| 1 | 保留原有 vLLM 配置不变 |
| 2 | 在 backend 中新增独立选项 `msg_model`，连接本地部署的大模型 API |
| 3 | vLLM 后端的模型名也改为从 config 读取（不再硬编码在 agent 里） |

**本地模型连接信息**

- `api_url`: `http://109.105.111.17:8000/api/msg_qwen3`
- `model_name`: `MSG_Model`
- `api_key`: `samsung`

**关键发现（探测确认）**

- 该端点前面有 **Kong API 网关**（`Server: kong/3.9.0`），后端为 **FastAPI**，实际由 **vLLM 0.26.0** 提供服务（`system_fingerprint: vllm-0.26.0-...`）。
- 它**不是**标准 OpenAI 路径（非 `/v1/chat/completions`）；直接请求 `/api/msg_qwen3` 返回 FastAPI 404。
- 正确端点为 `http://109.105.111.17:8000/api/msg_qwen3/chat/completions` —— 即用户给的 `api_url` 是 **base_url**，`openai` 客户端会自动追加 `/chat/completions`。
- 响应为标准 OpenAI 格式（`choices[0].message.content`），因此可直接复用 `VLLMToolAgent`（其底层即 `openai.OpenAI` 客户端）。

**关键决策（已确认）**

- 不新增 Agent 类：`msg_model` 后端复用 `VLLMToolAgent`，仅 `api_base` / `api_key` / `model` 不同。
- 配置完全隔离：vLLM 与 MSG_Model 各自独立配置项，互不影响。
- 后端标识用 `msg_model`（snake_case，与 `vllm` / `ollama` 一致），支持环境变量覆盖。

## 2. 现状分析

- `main.py` 的 `ToolCallingAgent` 通过 `_detect_best_backend()` / `_initialize_backend()` 分发到 `_init_vllm()` / `_init_ollama()`；argparse `--backend` 选项原为 `["vllm", "ollama", "auto"]`。
- `agent.py` 的 `VLLMToolAgent` 基于 `openai.OpenAI(base_url=..., api_key=...)`，原先模型名在 `chat()` / `chat_stream()` 中**硬编码**为 `"Qwen/Qwen3-0.6B"`。
- `VLLMToolAgent.__init__` 原签名 `__init__(self, api_base, api_key)`，不支持自定义 model。

## 3. 方案设计

### 3.1 config.py 配置

保留原 vLLM 配置，新增 MSG_Model 独立配置；vLLM 模型名改为集中配置：

```python
# vLLM（保留）
OPENAI_API_BASE = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
OPENAI_API_KEY = "EMPTY"  # vLLM doesn't require a real key
OPENAI_API_MODEL_NAME = os.getenv("OPENAI_API_MODEL_NAME", "Qwen/Qwen3-0.6B")

# 新增：MSG_Model 本地后端（OpenAI 兼容）
# base_url 给到 /api/msg_qwen3，OpenAI 客户端会自动追加 /chat/completions
MSG_MODEL_API_BASE = os.getenv("MSG_MODEL_API_BASE", "http://109.105.111.17:8000/api/msg_qwen3")
MSG_MODEL_API_KEY = os.getenv("MSG_MODEL_API_KEY", "samsung")
MSG_MODEL_NAME = os.getenv("MSG_MODEL_NAME", "MSG_Model")
```

### 3.2 agent.py：模型名可配置

`VLLMToolAgent.__init__` 新增 `model` 参数（默认 `MODEL_NAME`）；`chat()` / `chat_stream()` 中硬编码的 `"Qwen/Qwen3-0.6B"` 改为 `self.model`：

```python
def __init__(self, api_base: str = OPENAI_API_BASE, api_key: str = OPENAI_API_KEY,
             model: str = MODEL_NAME):
    self.client = OpenAI(api_key=api_key, base_url=api_base)
    self.model = model
    ...
```

### 3.3 main.py：新增 msg_model 后端

1. `_initialize_backend()` 增加分支：

```python
if self.backend_type == "vllm":
    self._init_vllm()
elif self.backend_type == "msg_model":
    self._init_msg_model()
else:
    self._init_ollama()
```

2. 新增 `_init_msg_model()`（复用 `VLLMToolAgent`）：

```python
def _init_msg_model(self):
    """Initialize MSG_Model local backend (OpenAI-compatible endpoint)"""
    from agent import VLLMToolAgent
    from config import MSG_MODEL_API_BASE, MSG_MODEL_API_KEY, MSG_MODEL_NAME
    self.agent = VLLMToolAgent(
        api_base=MSG_MODEL_API_BASE,
        api_key=MSG_MODEL_API_KEY,
        model=MSG_MODEL_NAME
    )
    logger.info("✅ MSG_Model agent initialized")
```

3. argparse `--backend` 选项加入 `msg_model`：`["vllm", "ollama", "msg_model", "auto"]`。
4. `_init_vllm()` 中 `VLLMToolAgent(...)` 增加 `model=OPENAI_API_MODEL_NAME`（vLLM 模型名去硬编码）。

## 4. 文件改动清单

| 文件 | 改动 |
|------|------|
| `config.py` | 新增 `OPENAI_API_MODEL_NAME`；新增 `MSG_MODEL_API_BASE` / `MSG_MODEL_API_KEY` / `MSG_MODEL_NAME`（保留原 vLLM 配置） |
| `agent.py` | `__init__` 新增 `model` 参数（默认 `MODEL_NAME`）；`chat()` / `chat_stream()` 硬编码模型名改为 `self.model`；import 引入 `MODEL_NAME` |
| `main.py` | 新增 `_init_msg_model()`；`_initialize_backend()` 增加 `msg_model` 分支；`_init_vllm()` 传入 `model=OPENAI_API_MODEL_NAME`；argparse `--backend` 加入 `msg_model`；更新 `__init__` docstring |

## 5. 边界情况

- 环境变量覆盖：`MSG_MODEL_API_BASE` / `MSG_MODEL_API_KEY` / `MSG_MODEL_NAME` / `OPENAI_API_MODEL_NAME` 均可覆盖默认值。
- `msg_model` 分支不触发本地 vLLM server 的健康检查与自动拉起（该逻辑仅存在于 `_init_vllm`）。
- 三后端配置隔离：修改 MSG_Model 配置不影响 `vllm` / `ollama`，反之亦然。
- `msg_model` 依赖远端服务可用性；服务不可达时由底层 `openai` 客户端抛出连接异常，`_init_msg_model` 不做静默回退（与显式指定后端的语义一致）。

## 6. 验证计划

- 语法自检：`python -m py_compile config.py agent.py main.py` ✅
- 端点探测：`POST http://109.105.111.17:8000/api/msg_qwen3/chat/completions` 返回 HTTP 200，标准 OpenAI 响应格式 ✅
- E2E 实测：`ToolCallingAgent(backend="msg_model").chat("1+1等于几？")` → 模型返回 `2`；`backend_type='msg_model'`、底层模型 `MSG_Model`，对话历史正常 ✅

## 7. 使用方式

```bash
# 命令行指定 msg_model 后端（交互式）
python main.py --backend msg_model --mode interactive

# 或指定单任务
python main.py --backend msg_model --mode single --task "查询温哥华天气"
```

```python
# 代码中使用
from main import ToolCallingAgent
agent = ToolCallingAgent(backend="msg_model")
print(agent.chat("你好", use_tools=True))
```

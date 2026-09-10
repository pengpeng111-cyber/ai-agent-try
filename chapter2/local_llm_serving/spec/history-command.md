# /history 命令实现说明

> 目标文件：`chapter2/local_llm_serving/main.py`
> 状态：已实施
> 日期：2026-09-10

## 1. 背景与需求

交互式对话中需要一个命令直接查看**当前内存里**的对话历史，用于：

| # | 需求 |
|---|------|
| 1 | 实时查看当前会话的完整消息列表（`conversation_history`） |
| 2 | 调试 ReAct 循环：确认 `tool_calls` 是否正确写入 `assistant` 消息、工具结果格式是否符合后端要求 |
| 3 | 不产生任何副作用（只读），不影响当前对话状态 |

## 2. 实现方案

### 2.1 命令处理块（`interactive_mode`，第 553-562 行）

```python
elif user_input.lower() == "/history":
    history = agent.agent.conversation_history
    if not history:
        print("\n📭 Conversation history is empty")
        continue
    print(f"\n📚 Conversation History ({len(history)} messages):")
    print("-" * 60)
    print(history)
    print("-" * 60)
    continue
```

### 2.2 帮助文本

两处提示均加入 `/history    - Show conversation history`：

- 初始欢迎界面命令列表（第 457 行）
- `/help` 命令输出（第 542 行）

两处均位于 `/sample <n>` 与 `/stream` 之间。

### 2.3 数据访问路径

```
agent (ToolCallingAgent, main.py 包装类)
  └── .agent (底层后端 agent: VLLMToolAgent / OllamaNativeAgent / ...)
        └── .conversation_history (List[Dict[str, Any]])  ← 直接读取公共属性
```

`conversation_history` 是底层 agent 的公共属性，直接读取即可，无需经包装层新增 getter。

## 3. 输出说明

### 3.1 空历史

```
📭 Conversation history is empty
```

### 3.2 非空历史

直接 `print(history)` 输出 Python 列表的 `repr`（单引号风格，未做逐条格式化）。示例（含一次工具调用）：

```
📚 Conversation History (4 messages):
------------------------------------------------------------
[{'role': 'user', 'content': '巴黎天气怎么样？'},
 {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'}}]},
 {'role': 'user', 'content': '<tool_result>\n{"temp": 22}\n</tool_result>', 'name': 'get_weather'},
 {'role': 'assistant', 'content': '巴黎当前气温 22°C。'}]
------------------------------------------------------------
```

> 历史条目结构因后端而异：vLLM/Qwen3 的工具结果为 `role="user"` + `<tool_result>` 标签 + `name` 字段；Ollama 原生为 `role="tool"`。直接 `print` 能完整呈现这些差异，便于排查格式问题。

## 4. 设计决策

- **直接 `print(history)` 输出原始结构**：完整展示 dict/JSON，最适合调试 `tool_calls` 与工具结果格式，实现最简。
- **只读命令**：不修改 `conversation_history`、不触发 LLM、不落盘，处理完 `continue` 跳过。
- **复用底层 agent 公共属性**：`ToolCallingAgent` 包装层直接读 `agent.agent.conversation_history`，与 `/new`、`/save`、自动持久化等命令读取方式一致，保持统一。

## 5. 与其它命令的关系

| 命令 | 查看对象 | 落盘 |
|------|----------|------|
| `/history` | **当前内存中的工作副本**（`conversation_history`） | 否 |
| `/sessions` | **已落盘**的会话列表（`sessions/*.json`），可切换 | 否（仅读取，切换时载入） |

`/history` 看的是「现在正在进行的对话」，`/sessions` 看的是「已经保存下来的对话」。二者与 session 持久化机制（`_persist_current` / 每轮自动持久化）配合：切到某 session 后继续对话，`/history` 可即时核对载入 + 新增后的上下文是否正确。

## 6. 已知局限与可选改进

- **局限**：直接 `print` 长内容会刷屏，未做截断，长会话输出可能较长。
- **可选改进**（当前未实现，仅作后续方向记录）：
  1. 逐条格式化输出（按 `[i] ROLE: preview`）；
  2. 内容超过 N 字符时截断（如 100 字符）；
  3. 追加标签提示，如 `[🔧 tools: get_weather]` / `[📦 tool_result: get_weather]`，提升可读性。

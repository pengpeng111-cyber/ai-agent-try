# Session 管理功能修改计划

> 目标文件：`chapter2/local_llm_serving/main.py`
> 状态：已实施
> 日期：2026-09-09

## 1. 背景与需求

为交互式对话增加会话（session）的保存、列表、切换、删除能力，并持久化到磁盘。

| # | 需求 |
|---|------|
| 1 | 新增 `/new` 命令：把当前 `conversation_history` 保存为新 session，并用 LLM 根据会话历史总结一个简短名称，用于展示与索引 |
| 2 | 增加 session 管理功能，管理所有已保存会话并持久化 |
| 3 | 新增 `/sessions` 命令：列出所有 session，展示后提示输入序号，选择并切换 session（可随意来回切换） |
| 4 | 新增 `/delete` 命令：列出所有 session，输入序号删除所选 session |
| 5 | 新增 `/save` 命令：保存当前会话为新 session，但**不重置**对话，保存后可继续当前会话（checkpoint 语义，区别于 `/new`） |

**关键决策（已确认）**
- `/new` 采用「保存并重置」：存档当前会话为新 session 后，重置当前会话、开始新对话（语义最贴合 `new`）。
- 不设独立的 `/load` 命令：切换能力合并进 `/sessions`——列出后输入序号即切换。
- `/delete` 采用交互式：先列出所有 session，再输入序号删除所选项（与 `/sessions` 一致的交互模式）。
- `/save` 采用「保存不重置」：存档当前会话为新 session 但保留对话，保存后可继续当前会话（checkpoint 语义）；`current_session_id` 指向新 session。

## 2. 现状分析

- `VLLMToolAgent`（`agent.py`）与 `OllamaNativeAgent`（`ollama_native.py`）均以 `self.conversation_history = []` 存储历史。
- `conversation_history` **不含 system prompt**（system prompt 在 `chat()` 中动态生成并拼接），仅含 `user` / `assistant` / `tool` 消息，均为纯 dict，可直接 JSON 序列化；恢复时直接赋值回 `agent.conversation_history` 即可。
- 访问入口：`agent.agent.conversation_history`；发送：`agent.chat(msg, use_tools=, stream=)`；重置：`agent.reset_conversation()`。
- 持久化惯例：项目已有 `logs/`、`runs/` 目录，`config.py` 集中管理路径；新增 `sessions/` 目录符合惯例。

## 3. 方案设计

### 3.1 数据格式

每个 session 存为一个 JSON 文件：`sessions/<session_id>.json`

```json
{
  "id": "a1b2c3d4",
  "name": "温哥华天气查询",
  "created_at": "2026-09-09T10:00:00",
  "updated_at": "2026-09-09T10:05:00",
  "backend": "ollama",
  "message_count": 10,
  "history": [ {"role": "user", "content": "..."} ]
}
```

### 3.2 SessionManager（新增 `session_manager.py`）

纯持久化组件，不耦合 LLM 逻辑，便于单元测试。

- `__init__(sessions_dir=None)`：默认取 `config.SESSIONS_DIR`，自动创建目录
- `create(history, name, backend="") -> str`：生成 id、写文件，返回 id
- `list() -> list[dict]`：扫描目录返回摘要，按 `updated_at` 降序
- `get(session_id) -> dict` / `load_history(session_id) -> list`
- `delete(session_id) -> bool`

### 3.3 LLM 命名：`ToolCallingAgent._summarize_name(history) -> str`

- 取前若干条 `user` / `assistant` 纯文本拼成 transcript（每条截断）。
- **快照-调用-恢复**：临时把 `conversation_history` 置空 → `chat(prompt, use_tools=False, stream=False)` → 恢复，避免污染当前会话；统一适配 vllm / ollama / msg_model。
- **fallback**：LLM 调用失败或返回空 → 取首条 `user` 消息前若干字 → 再不行用时间戳。
- prompt 要求：不超过 12 字的中文短语，只输出短语本身（清洗引号/前缀）。

### 3.4 命令设计（`interactive_mode`）

| 命令 | 行为 |
|------|------|
| `/new` | 空会话则提示无内容可保存；否则 LLM 命名 → `SessionManager.create` → `agent.reset_conversation()`，并置 `current_session_id = None` |
| `/sessions` | 列出所有 session（编号 / 名称 / 消息数 / 时间），当前项标 `← 当前`；展示后阻塞等待序号输入：合法编号 → 切换；`q` / 空 → 取消；非法 → 提示 |
| `/delete` | 列出所有 session（编号 / 名称 / 消息数 / 时间），输入序号删除所选项（`q` 取消）；若删的是当前项则清 `current_session_id` |
| `/save` | 空会话则提示无内容可保存；**在会话中**（`current_session_id` 已设）→ `_persist_current` 就地更新当前 session（保留 id/名称、刷新 `updated_at`）；**不在会话中** → LLM 命名 + `create` 新建并置为 current；均**不** `reset` |

**切换（`/sessions` 内）**：把目标 session 的 `history` 载入 agent（先 `reset_conversation()` 再赋值 `agent.agent.conversation_history`），并设 `current_session_id`。若当前会话存在未保存内容，切换前给出覆盖提示（不强制阻断）。

**`/sessions` 交互示例**

```
👤 /sessions
📚 已保存的会话 (3 个):
   1. 温哥华天气查询  (10 条)  2026-09-09 10:05  ← 当前
   2. 金融分析        (24 条)  2026-09-09 09:30
   3. 时区协调        (8 条)   2026-09-08 16:20
   💡 输入序号切换会话，输入 q 取消
👤 2
   ⚠️ 当前会话有 5 条未保存消息，切换将被覆盖（如需保存请先 /new）
✅ 已切换到: 金融分析
```

### 3.5 状态跟踪

`interactive_mode` 内维护 `current_session_id`，标识当前 agent 会话所对应的已保存 session（可为 `None`）。`/new` 保存重置后置 `None`；`/sessions` 切换成功、`/save`（新建分支）时置为对应 id；`/reset` 与 `/delete`（删当前项）时清为 `None`。它是「活跃工作副本」：每轮对话与 `/save` 都通过 `_persist_current` 回写。

### 3.6 持久化与 upsert（更新会话）

**问题**：早期 `/save` 一律调用 `SessionManager.create`（每次生成新 id），导致「切换到某 session → 继续对话 → `/save`」时新建了一个 session，当前 session 的 context 未被更新。

**修复**：

1. `SessionManager.update(session_id, history, name=None, backend=None) -> bool`：就地更新已有 session，刷新 `updated_at` / `history` / `message_count`，可选更新 `name` / `backend`；**保留** `id` / `created_at`；session 不存在返回 `False`。
2. `main._persist_current(sm, agent, current_session_id, reason)`（模块级 helper）：无 `current_session_id` 或无历史 → 返回 `None`（不落盘）；`sm.update` 成功 → 打印并返回原 id；当前 session 文件丢失（如被 `/delete` 删除）→ fallback 为 `sm.create` 新建并告警。
3. 三处调用点：

| 位置 | 行为 |
|------|------|
| `/save` | 在会话中 → `_persist_current`（**更新**，保留名称）；否则 → `create` 新建并置为 current |
| 每轮对话后 | `_persist_current(..., "自动持久化")`，context 实时同步到当前 session |
| `/reset` | 清空对话时同步置 `current_session_id = None`（脱离会话） |

> `/new` 语义保持不变：总是 `create` 新 session + 重置（存档并开始新对话）；`/save` 用于「保存/更新当前会话」。

## 4. 文件改动清单

- **新增** `chapter2/local_llm_serving/session_manager.py` — `SessionManager` 类
- **新增** `chapter2/local_llm_serving/test_session_manager.py` — 12 个测试用例（SessionManager CRUD + `update` / `_summarize_name` / `interactive_mode` 端到端，含 `/save` upsert 与自动持久化），可独立运行或经 pytest 运行
- **修改** `chapter2/local_llm_serving/config.py` — 新增 `SESSIONS_DIR = Path(__file__).parent / "sessions"`
- **修改** `chapter2/local_llm_serving/main.py` — `ToolCallingAgent._summarize_name`；`interactive_mode` 新增 `/new`、`/sessions`、`/delete` 命令处理及 `current_session_id` 状态；更新两处帮助文本（`interactive_mode` 开头 + `/help`）

## 5. 边界情况

- 空会话执行 `/new` → 提示无内容可保存，不创建空 session。
- `/sessions` 尚无任何 session → 提示「尚未保存任何会话」。
- `/sessions` 输入非法 / 越界编号 → 提示错误，不切换。
- `/delete` 无 session / 输入非法或越界编号 → 提示错误，不删除；删除当前加载项 → 清 `current_session_id`。
- LLM 命名失败 / 返回空 → 走 fallback 名称，保证 `/new` 始终可用。
- session 文件损坏 / 缺字段 → `list` / `get` 容错跳过，不影响其他 session。
- `/save` 或自动持久化时当前 session 文件已丢失（如被 `/delete` 删除）→ `_persist_current` fallback 为新建 session 并告警，`current_session_id` 指向新 id。
- `/reset` 清空对话后 `current_session_id` 置 `None`，后续 `/save` 走新建分支。
- 自动持久化仅在 `current_session_id` 已设且有历史时触发；`/new` 之后（`current_session_id = None`）的对话不会自动落盘，直到再次 `/save` 或 `/sessions` 切换。

## 6. 验证计划

- 单元测试见 `chapter2/local_llm_serving/test_session_manager.py`（共 12 个用例，详见第 7 节）。
- 语法自检：`python -m py_compile config.py session_manager.py main.py`。
- 运行结果（venv）：`12/12 tests passed`。

## 7. 测试用例

测试文件：`chapter2/local_llm_serving/test_session_manager.py`

运行方式：
```bash
# 直接运行
python test_session_manager.py
# 或经 pytest（需已安装）
pytest test_session_manager.py
```

测试通过 mock 底层 agent（`object.__new__` 绕过后端初始化）与临时目录隔离，实现离线、无副作用运行。

| # | 测试函数 | 覆盖点 |
|---|----------|--------|
| 1 | `test_session_manager_crud` | create 生成 id 且字段正确；list 按 `updated_at` 降序；get / load_history 往返；delete 成功、重复删除返回 False；损坏文件被跳过；缺失 id 返回 None |
| 2 | `test_session_manager_empty_dir` | 空目录 `list()` 返回 `[]` |
| 3 | `test_summarize_name_no_pollution` | 快照-恢复：总结后 `conversation_history` 不被污染 |
| 4 | `test_summarize_name_prefix_strip` | 模型返回 "主题：xxx" 时剥离前缀 → "xxx" |
| 5 | `test_summarize_name_fallback_empty` | 模型返回空 → 回退到首条 user 消息 |
| 6 | `test_summarize_name_fallback_exception` | 模型抛异常 → 回退到首条 user 消息 |
| 7 | `test_summarize_name_empty_history` | 空历史 → 返回空字符串 |
| 8 | `test_interactive_mode_e2e` | 模拟 stdin 走通 `/new` → `/sessions`（切换 + 取消）→ `/delete`（交互式选序号）→ 空列表全流程 |
| 9 | `test_interactive_mode_save` | `/save` 保存后对话**不被重置**（历史保留）、会话写入磁盘、`/sessions` 标记 `← current` |
| 10 | `test_session_manager_update` | `update` 刷新 history/message_count/updated_at，保留 id/created_at；不传 name 保留原名；session 不存在返回 False |
| 11 | `test_interactive_mode_save_updates_current` | 切换到 session → 聊天 → `/save`：**同一 id 被更新**（磁盘仍 1 个文件、history 含新消息、name 保留），而非新建 |
| 12 | `test_interactive_mode_autosave` | 处于 session 中时，每轮对话自动持久化到该 session（无需手动 `/save`），且不新建 session |

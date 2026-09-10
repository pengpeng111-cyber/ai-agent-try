#!/usr/bin/env python3
"""
Tests for the session management feature.

Covers:
  - SessionManager CRUD (create/list/get/load_history/delete) + fault tolerance
  - ToolCallingAgent._summarize_name (snapshot/restore, prefix strip, fallbacks)
  - interactive_mode session commands end-to-end (mock agent + simulated stdin)

Run directly:
    python test_session_manager.py
or via pytest:
    pytest test_session_manager.py
"""
import io
import builtins
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import session_manager
import main


# ---------- helpers ----------

class _MockInner:
    """Minimal stand-in for the underlying agent (no real backend)."""

    def __init__(self):
        self.conversation_history = []
        self._resp = ""

    def chat(self, message, use_tools=True, stream=False, **kw):
        return self._resp

    def reset_conversation(self):
        self.conversation_history = []


def _make_agent():
    """Build a ToolCallingAgent without triggering backend initialization."""
    agent = object.__new__(main.ToolCallingAgent)
    agent.backend_type = "mock"
    agent.agent = _MockInner()
    return agent


# ---------- SessionManager ----------

def test_session_manager_crud():
    with tempfile.TemporaryDirectory() as tmp:
        sm = session_manager.SessionManager(sessions_dir=Path(tmp))
        assert sm.sessions_dir.exists(), "sessions dir should be auto-created"

        h1 = [{"role": "user", "content": "你好"},
              {"role": "assistant", "content": "在的"}]
        id1 = sm.create(history=h1, name="测试会话A", backend="ollama")
        assert isinstance(id1, str) and len(id1) == 8

        data1 = sm.get(id1)
        assert data1 is not None
        assert data1["name"] == "测试会话A"
        assert data1["history"] == h1
        assert data1["message_count"] == 2
        assert data1["backend"] == "ollama"

        id2 = sm.create(history=[{"role": "user", "content": "hi"}], name="SessionB")

        lst = sm.list()
        assert len(lst) == 2
        assert lst[0]["id"] == id2, "list must be sorted by updated_at desc"
        assert lst[0]["name"] == "SessionB"

        assert sm.load_history(id1) == h1

        assert sm.delete(id1) is True
        assert sm.get(id1) is None
        assert sm.delete(id1) is False, "deleting twice should return False"

        # Corrupt file must be tolerated and skipped.
        (Path(tmp) / "corrupt123.json").write_text("{not valid json", encoding="utf-8")
        assert sm.get("corrupt123") is None
        assert all(s["id"] != "corrupt123" for s in sm.list())

        assert sm.load_history("nonexistent") is None


def test_session_manager_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        sm = session_manager.SessionManager(sessions_dir=Path(tmp))
        assert sm.list() == []


def test_session_manager_update():
    """update() refreshes history/message_count/updated_at, preserves id/created_at."""
    with tempfile.TemporaryDirectory() as tmp:
        sm = session_manager.SessionManager(sessions_dir=Path(tmp))
        h1 = [{"role": "user", "content": "你好"}]
        sid = sm.create(history=h1, name="会话A", backend="ollama")
        created_at = sm.get(sid)["created_at"]

        h2 = h1 + [{"role": "assistant", "content": "在的"},
                   {"role": "user", "content": "谢谢"}]
        assert sm.update(sid, h2, backend="ollama") is True
        data = sm.get(sid)
        assert data["id"] == sid
        assert data["created_at"] == created_at, "created_at must be preserved"
        assert data["history"] == h2
        assert data["message_count"] == len(h2)
        assert data["name"] == "会话A", "name preserved when not provided"

        assert sm.update(sid, h2, name="会话A改名") is True
        assert sm.get(sid)["name"] == "会话A改名"

        assert sm.update("nope1234", h2) is False, "missing session -> False"



# ---------- _summarize_name ----------

def test_summarize_name_no_pollution():
    agent = _make_agent()
    agent.agent._resp = "温哥华天气查询"
    agent.agent.conversation_history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
    ]
    original = list(agent.agent.conversation_history)

    name = agent._summarize_name(agent.agent.conversation_history)
    assert name == "温哥华天气查询"
    assert agent.agent.conversation_history == original, "history was polluted!"


def test_summarize_name_prefix_strip():
    agent = _make_agent()
    agent.agent._resp = "主题：温哥华天气"
    name = agent._summarize_name([{"role": "user", "content": "x"}])
    assert name == "温哥华天气"


def test_summarize_name_fallback_empty():
    agent = _make_agent()
    agent.agent._resp = ""
    agent.agent.conversation_history = [{"role": "user", "content": "帮我查一下东京天气"}]
    name = agent._summarize_name(agent.agent.conversation_history)
    assert name == "帮我查一下东京天气"


def test_summarize_name_fallback_exception():
    agent = _make_agent()

    def _raise(*a, **k):
        raise RuntimeError("boom")

    agent.agent.chat = _raise
    agent.agent.conversation_history = [{"role": "user", "content": "帮我查一下东京天气"}]
    name = agent._summarize_name(agent.agent.conversation_history)
    assert name == "帮我查一下东京天气"


def test_summarize_name_empty_history():
    agent = _make_agent()
    assert agent._summarize_name([]) == ""


# ---------- interactive_mode end-to-end ----------

def test_interactive_mode_e2e():
    """Simulate the full /new, /sessions, /delete flow against a mock agent."""
    # Point SessionManager at a throwaway temp dir so real data is untouched.
    tmp = tempfile.mkdtemp()
    sm_cls = session_manager.SessionManager
    orig_init = sm_cls.__init__

    def patched_init(self, sessions_dir=None):
        orig_init(self, sessions_dir=Path(tmp))

    sm_cls.__init__ = patched_init
    try:
        agent = _make_agent()
        agent.backend_type = "ollama"
        agent.agent.conversation_history = [
            {"role": "user", "content": "帮我看看温哥华天气"}
        ]
        agent.agent._resp = "温哥华天气查询"

        inputs = ["/new", "/sessions", "1", "/sessions", "q",
                  "/delete", "1", "/sessions", "/exit"]
        it = iter(inputs)
        buf = io.StringIO()

        def fake_input(prompt=""):
            return next(it)

        with patch.object(builtins, "input", side_effect=fake_input):
            with redirect_stdout(buf):
                main.interactive_mode(agent, stream=False)
        out = buf.getvalue()

        for expected in [
            "Saved session: 温哥华天气查询",
            "Conversation reset - starting fresh",
            "Saved Sessions (1)",
            "Switched to session: 温哥华天气查询",
            "Cancelled",
            "Deleted session: 温哥华天气查询",
            "No saved sessions yet",
        ]:
            assert expected in out, f"missing: {expected!r}\n--- output ---\n{out}"
    finally:
        sm_cls.__init__ = orig_init


def test_interactive_mode_save():
    """/save archives the current session WITHOUT resetting the conversation."""
    tmp = tempfile.mkdtemp()
    sm_cls = session_manager.SessionManager
    orig_init = sm_cls.__init__

    def patched_init(self, sessions_dir=None):
        orig_init(self, sessions_dir=Path(tmp))

    sm_cls.__init__ = patched_init
    try:
        agent = _make_agent()
        agent.backend_type = "ollama"
        agent.agent.conversation_history = [
            {"role": "user", "content": "帮我看看温哥华天气"},
            {"role": "assistant", "content": "温哥华今天晴，22 度"},
        ]
        agent.agent._resp = "温哥华天气查询"

        inputs = ["/save", "/sessions", "q", "/exit"]
        it = iter(inputs)
        buf = io.StringIO()

        def fake_input(prompt=""):
            return next(it)

        with patch.object(builtins, "input", side_effect=fake_input):
            with redirect_stdout(buf):
                main.interactive_mode(agent, stream=False)
        out = buf.getvalue()

        for expected in [
            "Saved session: 温哥华天气查询",
            "Conversation kept - continue chatting",
            "Saved Sessions (1)",
            "← current",
        ]:
            assert expected in out, f"missing: {expected!r}\n--- output ---\n{out}"

        # /save must keep the conversation (unlike /new which resets).
        assert agent.agent.conversation_history == [
            {"role": "user", "content": "帮我看看温哥华天气"},
            {"role": "assistant", "content": "温哥华今天晴，22 度"},
        ], "conversation should be kept after /save"

        # Exactly one session file should be written to disk.
        files = list(Path(tmp).glob("*.json"))
        assert len(files) == 1, f"expected 1 session file, got {files}"
    finally:
        sm_cls.__init__ = orig_init



def test_interactive_mode_save_updates_current():
    """Switching into a session then /save must UPDATE it, not create a new one."""
    tmp = tempfile.mkdtemp()
    sm_cls = session_manager.SessionManager
    orig_init = sm_cls.__init__

    def patched_init(self, sessions_dir=None):
        orig_init(self, sessions_dir=Path(tmp))

    sm_cls.__init__ = patched_init
    try:
        sm_seed = session_manager.SessionManager(sessions_dir=Path(tmp))
        h0 = [
            {"role": "user", "content": "帮我看看温哥华天气"},
            {"role": "assistant", "content": "温哥华今天晴，22 度"},
        ]
        seed_id = sm_seed.create(history=h0, name="Session A", backend="ollama")

        agent = object.__new__(main.ToolCallingAgent)
        agent.backend_type = "ollama"
        inner = _MockInner()
        inner._resp = "好的"

        def chat_append(message, use_tools=True, stream=False, **kw):
            inner.conversation_history.append({"role": "user", "content": message})
            inner.conversation_history.append({"role": "assistant", "content": inner._resp})
            return inner._resp

        inner.chat = chat_append
        agent.agent = inner

        inputs = ["/sessions", "1", "继续", "/save", "/sessions", "q", "/exit"]
        it = iter(inputs)
        buf = io.StringIO()

        def fake_input(prompt=""):
            return next(it)

        with patch.object(builtins, "input", side_effect=fake_input):
            with redirect_stdout(buf):
                main.interactive_mode(agent, stream=False)
        out = buf.getvalue()

        files = list(Path(tmp).glob("*.json"))
        assert len(files) == 1, f"expected 1 session file, got {[f.name for f in files]}"
        data = sm_seed.get(seed_id)
        assert data is not None, "original session must still exist"
        assert data["history"] == h0 + [
            {"role": "user", "content": "继续"},
            {"role": "assistant", "content": "好的"},
        ], "conversation context should be saved into the current session"
        assert data["message_count"] == 4
        assert data["name"] == "Session A", "name must be preserved on update"
        assert "Updated current session: Session A" in out, out
    finally:
        sm_cls.__init__ = orig_init


def test_interactive_mode_autosave():
    """While inside a session, each chat turn auto-persists context to it."""
    tmp = tempfile.mkdtemp()
    sm_cls = session_manager.SessionManager
    orig_init = sm_cls.__init__

    def patched_init(self, sessions_dir=None):
        orig_init(self, sessions_dir=Path(tmp))

    sm_cls.__init__ = patched_init
    try:
        sm_seed = session_manager.SessionManager(sessions_dir=Path(tmp))
        h0 = [{"role": "user", "content": "初始问题"},
              {"role": "assistant", "content": "初始回答"}]
        seed_id = sm_seed.create(history=h0, name="AutoSave", backend="ollama")

        agent = object.__new__(main.ToolCallingAgent)
        agent.backend_type = "ollama"
        inner = _MockInner()
        inner._resp = "自动"

        def chat_append(message, use_tools=True, stream=False, **kw):
            inner.conversation_history.append({"role": "user", "content": message})
            inner.conversation_history.append({"role": "assistant", "content": inner._resp})
            return inner._resp

        inner.chat = chat_append
        agent.agent = inner

        inputs = ["/sessions", "1", "第一句", "第二句", "/exit"]
        it = iter(inputs)
        buf = io.StringIO()

        def fake_input(prompt=""):
            return next(it)

        with patch.object(builtins, "input", side_effect=fake_input):
            with redirect_stdout(buf):
                main.interactive_mode(agent, stream=False)
        out = buf.getvalue()

        files = list(Path(tmp).glob("*.json"))
        assert len(files) == 1, "autosave must not create new sessions"
        data = sm_seed.get(seed_id)
        assert data["history"] == h0 + [
            {"role": "user", "content": "第一句"},
            {"role": "assistant", "content": "自动"},
            {"role": "user", "content": "第二句"},
            {"role": "assistant", "content": "自动"},
        ], "each turn should auto-persist into the current session"
        assert "自动持久化" in out, out
    finally:
        sm_cls.__init__ = orig_init



# ---------- runner ----------

def _run_all():
    tests = [
        test_session_manager_crud,
        test_session_manager_empty_dir,
        test_summarize_name_no_pollution,
        test_summarize_name_prefix_strip,
        test_summarize_name_fallback_empty,
        test_summarize_name_fallback_exception,
        test_summarize_name_empty_history,
        test_interactive_mode_e2e,
        test_interactive_mode_save,
        test_session_manager_update,
        test_interactive_mode_save_updates_current,
        test_interactive_mode_autosave,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  ✅ {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed, len(tests)


if __name__ == "__main__":
    import sys
    print("Running session management tests...\n")
    passed, total = _run_all()
    sys.exit(0 if passed == total else 1)

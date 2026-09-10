"""
Session Manager for persisting conversation sessions to disk.

Each session is stored as a JSON file under the sessions directory:
    sessions/<session_id>.json

The manager is intentionally free of any LLM/agent logic so it can be
unit-tested in isolation.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages persistence of conversation sessions on disk."""

    def __init__(self, sessions_dir: Optional[Path] = None):
        # Default to the configured directory; fall back to a local one.
        if sessions_dir is None:
            try:
                from config import SESSIONS_DIR
                sessions_dir = SESSIONS_DIR
            except Exception:
                sessions_dir = Path(__file__).parent / "sessions"
        self.sessions_dir = Path(sessions_dir)
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create sessions dir {self.sessions_dir}: {e}")

    def _path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def create(self, history: List[Dict[str, Any]], name: str,
               backend: str = "") -> str:
        """Create and persist a new session. Returns the session id."""
        session_id = uuid.uuid4().hex[:8]
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        data = {
            "id": session_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "backend": backend,
            "message_count": len(history),
            "history": history,
        }
        self._write(session_id, data)
        return session_id

    def update(self, session_id: str, history: List[Dict[str, Any]],
               name: Optional[str] = None, backend: Optional[str] = None) -> bool:
        """Update an existing session in place.

        Preserves the session id and created_at; refreshes updated_at,
        history, message_count, and (optionally) name / backend. Returns
        False if the session does not exist on disk.
        """
        data = self._read(session_id)
        if data is None:
            return False
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        data["history"] = history
        data["message_count"] = len(history)
        data["updated_at"] = now
        if name is not None and name != "":
            data["name"] = name
        if backend is not None:
            data["backend"] = backend
        return self._write(session_id, data)

    def list(self) -> List[Dict[str, Any]]:
        """Return summary info for all sessions, most recently updated first."""
        sessions: List[Dict[str, Any]] = []
        if not self.sessions_dir.exists():
            return sessions
        for f in self.sessions_dir.glob("*.json"):
            data = self._read(f.stem)
            if data is None:
                continue
            sessions.append({
                "id": data.get("id", f.stem),
                "name": data.get("name", "(unnamed)"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "backend": data.get("backend", ""),
                "message_count": data.get(
                    "message_count", len(data.get("history", []))
                ),
            })
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the full session data (including history), or None."""
        return self._read(session_id)

    def load_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Return the stored conversation history, or None on failure."""
        data = self._read(session_id)
        if data is None:
            return None
        history = data.get("history")
        if not isinstance(history, list):
            return None
        return history

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if a file was removed."""
        path = self._path(session_id)
        try:
            if path.exists():
                path.unlink()
                return True
        except Exception as e:
            logger.warning(f"Failed to delete session {session_id}: {e}")
        return False

    def _write(self, session_id: str, data: Dict[str, Any]) -> bool:
        path = self._path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to write session {session_id}: {e}")
            return False

    def _read(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if not isinstance(data, dict):
                return None
            return data
        except Exception as e:
            logger.warning(f"Failed to read session {session_id}: {e}")
            return None

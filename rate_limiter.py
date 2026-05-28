# backend/session/manager.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.config import config
from backend.session.prompt_builder import DynamicPromptBuilder


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


class ChatSession:
    def __init__(self):
        self.messages: list[ChatMessage] = []
        self.summary: Optional[str] = None
        self.created_at: float = time.time()
        self.last_activity: float = time.time()

    def add(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))
        self.last_activity = time.time()
        self._maybe_summarize()

    def _maybe_summarize(self):
        if len(self.messages) > config.summarize_after and not self.summary:
            old_messages = self.messages[:len(self.messages) - config.max_chat_history // 2]
            self.summary = self._create_summary(old_messages)
            self.messages = self.messages[len(self.messages) - config.max_chat_history // 2:]

    def _create_summary(self, messages: list[ChatMessage]) -> str:
        text = " ".join([m.content for m in messages if m.role == "user"])
        return text[:300]

    def reset(self):
        self.messages = []
        self.summary = None
        self.last_activity = time.time()

    def get_length(self) -> int:
        return len(self.messages)

    def is_expired(self) -> bool:
        return time.time() - self.last_activity > config.session_timeout


class SessionManager:
    """Manage per-session chat state and build LLM messages using a dynamic prompt builder."""

    def __init__(self):
        self.sessions: dict[str, ChatSession] = {}
        self.lock = threading.Lock()
        self.prompt_builder = DynamicPromptBuilder()

    def get_session(self, session_id: str) -> ChatSession:
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = ChatSession()
            session = self.sessions[session_id]
            if session.is_expired():
                session.reset()
            return session

    def get_messages_for_llm(
        self,
        session_id: str,
        user_message: str,
        context_chunks: list[dict]
    ) -> list[dict]:
        session = self.get_session(session_id)
        # Build a dynamic, document-aware system prompt.
        system_content = self.prompt_builder.build(context_chunks or [])
        messages = [{"role": "system", "content": system_content}]

        if session.summary:
            messages.append({"role": "system", "content": f"[Riepilogo conversazione: {session.summary}]"})

        for msg in session.messages[-config.max_chat_history:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_message})
        return messages

    def add_message(self, session_id: str, role: str, content: str):
        session = self.get_session(session_id)
        session.add(role, content)

    def reset_session(self, session_id: str):
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].reset()

    def get_session_length(self, session_id: str) -> int:
        return self.get_session(session_id).get_length()

    def cleanup_expired(self):
        with self.lock:
            expired = [
                sid for sid, session in self.sessions.items()
                if session.is_expired()
            ]
            for sid in expired:
                del self.sessions[sid]


session_manager = SessionManager()
last_code_per_session: dict[str, str] = {}
last_query_per_session: dict[str, str] = {}

"""Chat parsing helpers for reuse in tests and the Streamlit UI."""

from __future__ import annotations

__all__ = ["parse_chat_history"]


def parse_chat_history(file_content: str):
    """Parse uploaded chat history content into structured messages."""
    try:
        from chat_history_analyzer import ChatHistoryParser

        return ChatHistoryParser.auto_detect_format(file_content)
    except Exception:
        from types import SimpleNamespace

        messages = []
        for line in file_content.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                role, content = line.split(":", 1)
            else:
                role, content = "user", line
            messages.append(SimpleNamespace(content=content.strip(), role=role.strip().lower()))
        return messages

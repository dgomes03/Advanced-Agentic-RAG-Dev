"""
Chat Storage Manager for RAG Framework
Handles file-based persistence of conversations
"""
import json
import os
from datetime import datetime
from pathlib import Path


class ChatStorageManager:
    """Manages chat storage with JSON file persistence."""

    def __init__(self, storage_dir: str = None):
        """
        Initialize the chat storage manager.

        Args:
            storage_dir: Directory to store chat files. Defaults to ./chats
        """
        if storage_dir is None:
            storage_dir = Path(__file__).parent / 'chats'
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_chat(self, title: str = None) -> dict:
        """
        Create a new chat with a unique ID.

        Args:
            title: Optional title for the chat

        Returns:
            Chat data dictionary
        """
        chat_id = f"chat_{int(datetime.now().timestamp() * 1000)}"
        now = datetime.now().isoformat()

        chat_data = {
            "id": chat_id,
            "title": title or "New Conversation",
            "created_at": now,
            "updated_at": now,
            "messages": []
        }

        self._save_chat_file(chat_id, chat_data)
        return chat_data

    def load_chat(self, chat_id: str) -> dict:
        """
        Load a chat by its ID.

        Args:
            chat_id: The chat ID to load

        Returns:
            Chat data dictionary or None if not found
        """
        file_path = self.storage_dir / f"{chat_id}.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading chat {chat_id}: {e}")
            return None

    def update_chat(self, chat_id: str, messages: list, title: str = None) -> dict:
        """
        Update a chat with new messages.

        Args:
            chat_id: The chat ID to update
            messages: List of message dictionaries with 'role' and 'content'
            title: Optional new title

        Returns:
            Updated chat data or None if not found
        """
        chat_data = self.load_chat(chat_id)
        if chat_data is None:
            return None

        chat_data["messages"] = messages
        chat_data["updated_at"] = datetime.now().isoformat()

        if title:
            chat_data["title"] = title
        elif messages and chat_data["title"] == "New Conversation":
            # Auto-generate title from first user message
            first_user_msg = next(
                (m for m in messages if m.get("role") == "user"),
                None
            )
            if first_user_msg:
                content = first_user_msg.get("content", "")
                # Truncate to first 50 chars
                chat_data["title"] = content[:50] + ("..." if len(content) > 50 else "")

        self._save_chat_file(chat_id, chat_data)
        return chat_data

    def delete_chat(self, chat_id: str) -> bool:
        """
        Delete a chat by its ID.

        Args:
            chat_id: The chat ID to delete

        Returns:
            True if deleted, False if not found
        """
        file_path = self.storage_dir / f"{chat_id}.json"
        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            return True
        except IOError as e:
            print(f"Error deleting chat {chat_id}: {e}")
            return False

    def list_chats(self) -> list:
        """
        List all chats sorted by updated date (newest first).

        Returns:
            List of chat summary dictionaries
        """
        chats = []

        for file_path in self.storage_dir.glob("chat_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                    # Return summary without full messages
                    summary = {
                        "id": chat_data.get("id"),
                        "title": chat_data.get("title", "Untitled"),
                        "created_at": chat_data.get("created_at"),
                        "updated_at": chat_data.get("updated_at"),
                        "message_count": len(chat_data.get("messages", [])),
                        "preview": self._get_preview(chat_data.get("messages", []))
                    }
                    chats.append(summary)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {file_path}: {e}")
                continue

        # Sort by updated_at, newest first
        chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return chats

    def _save_chat_file(self, chat_id: str, chat_data: dict) -> None:
        """Save chat data to a JSON file."""
        file_path = self.storage_dir / f"{chat_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)

    def _get_preview(self, messages: list, max_length: int = 100) -> str:
        """Get a preview of the last assistant message."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if len(content) > max_length:
                    return content[:max_length] + "..."
                return content
        return ""


# Global instance for the server
_chat_storage = None


def get_chat_storage() -> ChatStorageManager:
    """Get the global chat storage instance."""
    global _chat_storage
    if _chat_storage is None:
        _chat_storage = ChatStorageManager()
    return _chat_storage

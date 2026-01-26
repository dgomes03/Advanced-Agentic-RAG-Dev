"""
ConversationManager: Manages persistent conversation history for multi-turn chat.

This enables proper KV-cache utilization by maintaining a growing conversation
that the cache can prefix-match against, rather than rebuilding from scratch
each query.
"""


class ConversationManager:
    """Manages persistent conversation history for multi-turn chat."""

    SYSTEM_PROMPT = (
        "You are an assistant with document and web search tools. "
        "Use tools only when needed; if unclear, ask the user first. "
        "Base answers strictly on tool results. "
        "If insufficient information is found, inform the user. "
        "End informative responses by offering to elaborate. "
        "No sequential tool calls. Current year: 2026."
    )

    def __init__(self, system_prompt: str = None):
        """
        Initialize the conversation manager.

        Args:
            system_prompt: Optional custom system prompt. Uses default if None.
        """
        self._system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.conversation = [{"role": "system", "content": self._system_prompt}]

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.conversation.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Add an assistant response to the conversation."""
        self.conversation.append({"role": "assistant", "content": content})

    def add_tool_call(self, tool_calls: list):
        """
        Add a tool call to the conversation.

        Args:
            tool_calls: List of formatted tool call objects with id, function, type
        """
        self.conversation.append({
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls
        })

    def add_tool_result(self, tool_name: str, content: str, tool_call_id: str):
        """
        Add a tool result to the conversation.

        Args:
            tool_name: Name of the tool that was called
            content: The tool's output/result
            tool_call_id: ID matching the original tool call
        """
        self.conversation.append({
            "role": "tool",
            "name": tool_name,
            "content": content,
            "tool_call_id": tool_call_id
        })

    def get_conversation(self) -> list:
        """Return the full conversation history."""
        return self.conversation

    def clear(self):
        """Reset conversation to just the system prompt."""
        self.conversation = [{"role": "system", "content": self._system_prompt}]

    def get_turn_count(self) -> int:
        """Return the number of user messages in the conversation."""
        return sum(1 for msg in self.conversation if msg.get("role") == "user")

    def __len__(self) -> int:
        """Return total number of messages in conversation."""
        return len(self.conversation)

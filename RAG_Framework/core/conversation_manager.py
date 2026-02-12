"""
ConversationManager: Manages persistent conversation history for multi-turn chat.

This enables proper KV-cache utilization by maintaining a growing conversation
that the cache can prefix-match against, rather than rebuilding from scratch
each query.
"""

from datetime import datetime


class ConversationManager:
    """Manages persistent conversation history for multi-turn chat."""

    SYSTEM_PROMPT = (
        "You are an assistant with document, SQL database and web search tools.\n"
        "Use tools only when needed; if unclear, ask the user first.\n"
        "Base answers strictly on tool results.\n"
        "If insufficient information is found, inform the user.\n"
        "End informative responses by offering to elaborate.\n"
        "When searching the web, cite your sources.\n"
        "\nBe objective and brief in your responses."
        "\nIf a tool is not providing the information you need, try a different tool or ask the user for clarification instead of making assumptions."
    )

    REASONING_SYSTEM_PROMPT = (
        "# HOW YOU SHOULD THINK AND ANSWER\n\n"
        "First draft your thinking process (inner monologue) until you arrive at a response. "
        "Format your response using Markdown, and use LaTeX for any mathematical equations. "
        "Write both your thoughts and the response in the same language as the input.\n\n"
        "Your thinking process must follow the template below:\n"
        "[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. "
        "Be as casual and as long as you want until you are confident to generate the response "
        "to the user.[/THINK]\nHere, provide a self-contained response.\n\n"
        "For trivial greetings or one-word acknowledgments you can skip the [THINK] block.\n\n"
        "# YOUR ROLE AND CAPABILITIES\n\n"
        + SYSTEM_PROMPT
    )

    def __init__(self, system_prompt: str = None, reasoning_model: bool = False):
        """
        Initialize the conversation manager.

        Args:
            system_prompt: Optional custom system prompt. Uses default if None.
            reasoning_model: If True, use reasoning system prompt with [THINK] support.
        """
        self.reasoning_model = reasoning_model
        if system_prompt:
            base_prompt = system_prompt
        elif reasoning_model:
            base_prompt = self.REASONING_SYSTEM_PROMPT
        else:
            base_prompt = self.SYSTEM_PROMPT
        # Inject current date/time at the start of system prompt
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        self._system_prompt = f"Current date and time: {current_datetime}\n\n{base_prompt}"
        self.conversation = [{"role": "system", "content": self._system_prompt}]

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.conversation.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Add an assistant response to the conversation."""
        self.conversation.append({"role": "assistant", "content": content})

    def add_assistant_message_with_thinking(self, thinking: str, text: str):
        """Add an assistant response with structured thinking content.

        The chat template renders these as [THINK]...[/THINK] + plain text,
        maintaining KV-cache consistency across turns.
        """
        content = []
        if thinking:
            content.append({"type": "thinking", "thinking": thinking})
        content.append({"type": "text", "text": text})
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

    def load_from_messages(self, messages: list):
        """Restore conversation from saved messages (user/assistant only).
        Note: This loses tool call/result messages, which invalidates the KV-cache.
        Prefer load_from_state() when restoring alongside a saved cache.
        """
        self.clear()
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'user':
                self.add_user_message(content)
            elif role == 'assistant':
                self.add_assistant_message(content)

    def load_from_state(self, conversation: list):
        """Restore full conversation state including tool calls and results.
        This preserves the exact token sequence the KV-cache was built from.
        """
        self.conversation = conversation

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

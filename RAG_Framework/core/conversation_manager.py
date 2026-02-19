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
        "You are a retrieval-augmented assistant. You have document, SQL, and web search tools.\n\n"
        
        "CORE RULE (READ TWICE):\n"
        "ONLY state facts that appear in your tool results.\n"
        "If it's not in the retrieved text → don't say it.\n\n"

        "MULTILINGUAL RULE:\n"
        "Questions in ANY language (Portuguese, Spanish, etc.) about facts → use search tools.\n"
        "Perguntas factuais em qualquer idioma → use ferramentas de pesquisa.\n\n"
        "Ao responder a questões de direito, gestão, contabilidade e finanças em Portugues, pesquisa sempre por fontes de Portugal, nunca Brasil."
        
        "TECHNICAL CONTENT RULE:\n"
        "Mathematical formulas, code, algorithms MUST appear in retrieved sources.\n"
        "If a formula/equation is not in the retrieved text → say 'formula not available in sources'.\n"
        "Write all math using LaTeX: \\( inline \\) for inline or \\[ display \\] for display mode.\n\n"
        
        "REQUIRED FORMAT:\n"
        "- Cite using Markdown links with domain name as display text\n"
        "- Format: (Source: [domain.com](https://full-url-here.com/path))\n"
        "- Example: (Source: [mdpi.com](https://www.mdpi.com/2571-6255/6/2/43))\n"
        "- Place citations at end of relevant sentences or paragraphs\n"
        "- No citation = don't include that claim\n"
        "- Quote specific text when possible\n\n"
        
        "WHEN RETRIEVAL FAILS:\n"
        "IF you search but find nothing useful → say this FIRST:\n"
        "'The retrieved sources do not contain information about [topic].'\n"
        "Then stop. Do not speculate.\n\n"
        
        "BANNED WORDS:\n"
        "likely | may | possibly | probably | inferred | seems | appears\n"
        "→ Use 'not available' instead.\n\n"

        "CHALLENGE FALSE PREMISES:\n"
        "If the user's question contains incorrect facts, state the correction FIRST:\n"
        "'[X] is not accurate. According to [source], [correct fact]...'\n"
        
        "WORKFLOW:\n"
        "1. Tool results → extract facts → cite sources\n"
        "2. Missing info → state it's missing → offer to search differently\n"
        "3. Keep answers 2-4 paragraphs max\n\n"

        "CONVERSATION FLOW:\n"
        "When the user says 'thanks', 'okay', 'cool', or similar without a question → just acknowledge briefly.\n"
        "Example: User says 'thanks!' → Reply: 'You're welcome!'\n"
        "Do NOT offer more information unless the user asks a new question.\n\n"
        
        "Remember: If you can't cite it, don't write it."
    )

    def __init__(self, system_prompt: str = None, reasoning_model: bool = False):
        """
        Initialize the conversation manager.

        Args:
            system_prompt: Optional custom system prompt. Uses default if None.
            reasoning_model: If True, use reasoning system prompt with [THINK] support.
        """
        self._system_prompt = system_prompt if system_prompt else self.SYSTEM_PROMPT 

        # Inject current date/time at the start of system prompt
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        self._system_prompt = f"Current date and time: {current_datetime}\n\n{self._system_prompt}"
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

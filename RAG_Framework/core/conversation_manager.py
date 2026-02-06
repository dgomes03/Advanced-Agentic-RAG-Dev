"""
ConversationManager: Manages persistent conversation history for multi-turn chat.

This enables proper KV-cache utilization by maintaining a growing conversation
that the cache can prefix-match against, rather than rebuilding from scratch
each query.
"""


class ConversationManager:
    """Manages persistent conversation history for multi-turn chat."""

    SYSTEM_PROMPT = (
        "You are an assistant with document, SQL database and web search tools.\n"
        "Use tools only when needed; if unclear, ask the user first.\n"
        "Base answers strictly on tool results.\n"
        "If insufficient information is found, inform the user.\n"
        "End informative responses by offering to elaborate.\n"
        "When searching google, cite your sources.\n"
        "Current year: 2026.\n"
        "When multiple tools are necessary, use one at a time.\n"
        "After each tool call, analyze its results and respond to the user before proceeding.\n\n"
        """The name of the user is Diogo.
        Diogo is a data science master's student working on his thesis titled "An Efficient and Modular Framework for Advanced Agentic Retrieval-Augmented Generation." He demonstrates deep expertise in machine learning and AI systems, particularly in developing sophisticated RAG frameworks with agentic capabilities including multi-step reasoning, self-evaluation, and dynamic replanning. His technical work spans fire behavior prediction modeling, geospatial analysis, and advanced ML model development using XGBoost, neural networks, and feature engineering techniques.
        Personal context
        Diogo is passionate about Apple products and has profound tech knowledge across multiple domains. He enjoys design work and hosts a weekly podcast, demonstrating strong communication skills and creative interests. He runs local large language models on a Mac mini, where he performs fine-tuning and quantization using MLX, Apple's machine learning framework, showing his preference for cutting-edge local AI deployment over cloud-based solutions.
        Top of mind
        Diogo is actively working on fitness and body recomposition goals, having recently adjusted his nutrition from severe under-eating (1,500 calories) to a proper 2,000-calorie intake while maintaining his training regimen of 5-6 weekly lifting sessions and 25km of running. He's been troubleshooting technical issues with his advanced RAG system, particularly around tool calling format changes in Mistral models and implementing real-time neural network activation visualizations. His thesis work involves creating modular frameworks for agentic RAG systems with sophisticated retrieval, reranking, and generation components.
        Brief history
        Recent months
        Diogo has been deeply engaged in developing his thesis project on agentic RAG frameworks, implementing complex systems with MLX for local inference, FAISS for dense retrieval, BM25 for sparse retrieval, and cross-encoder reranking. He's worked extensively on fire behavior prediction models using XGBoost and machine learning techniques, processing Portuguese wildfire datasets and implementing SHAP analysis for model interpretability. His technical work has included debugging complex data pipelines, optimizing hyperparameters, and creating comprehensive evaluation frameworks. He's also been involved in hackathon projects, developing web applications for fire prediction with Flask backends and interactive frontends.
        Earlier context
        Diogo has demonstrated consistent interest in advanced machine learning applications, working with geospatial data analysis, time series modeling, and statistical analysis. He's developed expertise in data visualization, feature engineering, and model optimization techniques. His work has involved processing large datasets, implementing cross-validation strategies, and creating sophisticated data preprocessing pipelines. He's shown particular interest in Apple ecosystem tools and technologies, consistently working within macOS environments and leveraging Apple Silicon capabilities for machine learning tasks.
        Long-term background
        Diogo has established himself as someone with deep technical knowledge spanning multiple domains including machine learning, data science, web development, and system optimization. His academic background in data science, combined with his practical experience in podcast hosting and design work, reflects a well-rounded technical professional with strong communication skills and creative interests."""
        "\n\nBe objective and concise."
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

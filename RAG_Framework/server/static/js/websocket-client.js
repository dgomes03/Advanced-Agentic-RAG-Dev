/**
 * WebSocket Client for RAG Framework
 * Handles real-time communication with the server via Socket.IO
 */

class RAGWebSocketClient {
    constructor() {
        this.socket = null;
        this.currentMessageId = null;
        this.isProcessing = false;
        this.initialize();
    }

    /**
     * Initialize Socket.IO connection and set up event handlers
     */
    initialize() {
        // Connect to the server
        this.socket = io({
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: 5
        });

        // Set up event handlers
        this.setupEventHandlers();

        // Set up UI event handlers
        this.setupUIHandlers();
    }

    /**
     * Set up Socket.IO event handlers
     */
    setupEventHandlers() {
        // Connection events
        this.socket.on('connect', () => this.onConnect());
        this.socket.on('disconnect', () => this.onDisconnect());
        this.socket.on('connect_error', (error) => this.onConnectionError(error));

        // Message events
        this.socket.on('text_chunk', (data) => this.onTextChunk(data));

        // Tool call events
        this.socket.on('tool_call_start', (data) => this.onToolCallStart(data));
        this.socket.on('tool_call_result', (data) => this.onToolCallResult(data));

        // Reasoning events
        this.socket.on('reasoning_step', (data) => this.onReasoningStep(data));
        this.socket.on('reasoning_goal', (data) => this.onReasoningGoal(data));
        this.socket.on('reasoning_evaluation', (data) => this.onReasoningEvaluation(data));

        // Status events
        this.socket.on('status', (data) => this.onStatus(data));
        this.socket.on('done', (data) => this.onDone(data));
        this.socket.on('error', (data) => this.onError(data));
    }

    /**
     * Set up UI event handlers
     */
    setupUIHandlers() {
        const sendBtn = document.getElementById('send-btn');
        const userInput = document.getElementById('user-input');

        // Send button click
        sendBtn.addEventListener('click', () => this.sendMessage());

        // Enter key to send (Shift+Enter for new line)
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = Math.min(userInput.scrollHeight, 200) + 'px';
        });
    }

    /**
     * Send a message to the server
     */
    sendMessage() {
        const userInput = document.getElementById('user-input');
        const message = userInput.value.trim();

        if (!message || this.isProcessing) {
            return;
        }

        // Clear input
        userInput.value = '';
        userInput.style.height = 'auto';

        // Generate message ID
        this.currentMessageId = 'msg-' + Date.now();

        // Create user message in UI
        UIComponents.createUserMessage(message);

        // Remove welcome message if it exists
        const welcomeMessage = document.querySelector('.welcome');
        if (welcomeMessage) {
            welcomeMessage.style.animation = 'fadeOut 0.3s ease';
            setTimeout(() => welcomeMessage.remove(), 300);
        }

        // Create assistant message placeholder
        UIComponents.createAssistantMessage(this.currentMessageId);

        // Set processing state
        this.isProcessing = true;
        this.updateSendButton(false);

        // Send query to server
        this.socket.emit('query', {
            query: message,
            message_id: this.currentMessageId
        });

        console.log('Query sent:', message);
    }

    /**
     * Handle connection established
     */
    onConnect() {
        console.log('Connected to server');
        this.updateStatusIndicator('connected', 'Ready');
        this.updateSendButton(true);
    }

    /**
     * Handle disconnection
     */
    onDisconnect() {
        console.log('Disconnected from server');
        this.updateStatusIndicator('error', 'Disconnected');
        this.updateSendButton(false);
        this.isProcessing = false;
    }

    /**
     * Handle connection error
     */
    onConnectionError(error) {
        console.error('Connection error:', error);
        this.updateStatusIndicator('error', 'Connection Error');
    }

    /**
     * Handle text chunk received
     */
    onTextChunk(data) {
        if (this.currentMessageId) {
            const messageContent = document.querySelector(`[data-message-id="${this.currentMessageId}"]`);
            if (messageContent) {
                // Accumulate text in data attribute
                const currentText = messageContent.getAttribute('data-raw-text') || '';
                const newText = currentText + data.text;
                messageContent.setAttribute('data-raw-text', newText);

                // Render markdown in real-time
                const html = MarkdownRenderer.render(newText);
                messageContent.innerHTML = html;

                // Render math with KaTeX
                MarkdownRenderer.renderMath(messageContent);

                // Scroll to bottom
                UIComponents.scrollToBottom();
            }
        }
    }

    /**
     * Handle tool call start
     */
    onToolCallStart(data) {
        console.log('Tool call started:', data);
        UIComponents.createToolCallCard(this.currentMessageId, data);
    }

    /**
     * Handle tool call result
     */
    onToolCallResult(data) {
        console.log('Tool call result:', data);
        UIComponents.updateToolCallResult(data.tool_id, data.result, data.status);
    }

    /**
     * Handle reasoning step
     */
    onReasoningStep(data) {
        console.log('Reasoning step:', data);
        UIComponents.updateReasoningStep(this.currentMessageId, data);
    }

    /**
     * Handle reasoning goal update
     */
    onReasoningGoal(data) {
        console.log('Reasoning goal:', data);
        UIComponents.updateReasoningGoal(this.currentMessageId, data);
    }

    /**
     * Handle reasoning evaluation
     */
    onReasoningEvaluation(data) {
        console.log('Reasoning evaluation:', data);
        UIComponents.updateReasoningEvaluation(this.currentMessageId, data);
    }

    /**
     * Handle status update
     */
    onStatus(data) {
        console.log('Status:', data);
        const state = data.state || 'thinking';
        const message = data.message || 'Processing...';

        if (state === 'connected') {
            this.updateStatusIndicator('connected', message);
        } else if (state === 'thinking') {
            this.updateStatusIndicator('thinking', message);
        }
    }

    /**
     * Handle completion
     */
    onDone(data) {
        console.log('Query completed');

        // Final render is already done in real-time, just do final cleanup
        if (this.currentMessageId) {
            const messageContent = document.querySelector(`[data-message-id="${this.currentMessageId}"]`);
            if (messageContent) {
                // Get the accumulated text from data attribute
                const rawText = messageContent.getAttribute('data-raw-text') || messageContent.textContent;

                // Final render
                const html = MarkdownRenderer.render(rawText);
                messageContent.innerHTML = html;

                // Final math render
                MarkdownRenderer.renderMath(messageContent);

                // Clean up data attribute
                messageContent.removeAttribute('data-raw-text');
            }
        }

        // Reset state
        this.isProcessing = false;
        this.updateStatusIndicator('connected', 'Ready');
        this.updateSendButton(true);

        // Focus input
        document.getElementById('user-input').focus();
    }

    /**
     * Handle error
     */
    onError(data) {
        console.error('Error:', data);

        // Display error message
        if (this.currentMessageId) {
            const messageContent = document.querySelector(`[data-message-id="${this.currentMessageId}"]`);
            if (messageContent) {
                messageContent.innerHTML = `<p style="color: var(--error);">❌ Error: ${data.message}</p>`;
            }
        }

        // Reset state
        this.isProcessing = false;
        this.updateStatusIndicator('error', 'Error occurred');
        this.updateSendButton(true);
    }

    /**
     * Update status indicator
     */
    updateStatusIndicator(state, text) {
        const indicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');

        // Remove all state classes
        indicator.classList.remove('connecting', 'error', 'thinking');

        // Add new state class
        if (state !== 'connected') {
            indicator.classList.add(state);
        }

        // Update text
        statusText.textContent = text;
    }

    /**
     * Update send button state
     */
    updateSendButton(enabled) {
        const sendBtn = document.getElementById('send-btn');
        sendBtn.disabled = !enabled;
    }
}

// Initialize the WebSocket client when the page loads
let ragClient;
document.addEventListener('DOMContentLoaded', () => {
    ragClient = new RAGWebSocketClient();
    console.log('RAG WebSocket Client initialized');
});

// Add fadeOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; transform: scale(1); }
        to { opacity: 0; transform: scale(0.95); }
    }
`;
document.head.appendChild(style);

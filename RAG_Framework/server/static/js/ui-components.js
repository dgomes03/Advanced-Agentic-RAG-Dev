/**
 * UI Components for RAG Framework
 * Handles all UI rendering and manipulation
 */

class UIComponents {
    /**
     * Create a user message bubble
     */
    static createUserMessage(text) {
        const messagesArea = document.getElementById('messages');
        const template = document.getElementById('user-message-template');
        const clone = template.content.cloneNode(true);

        const messageContent = clone.querySelector('.message-content');
        messageContent.textContent = text;

        messagesArea.appendChild(clone);
        this.scrollToBottom();
    }

    /**
     * Create an assistant message placeholder
     */
    static createAssistantMessage(messageId) {
        const messagesArea = document.getElementById('messages');
        const template = document.getElementById('assistant-message-template');
        const clone = template.content.cloneNode(true);

        const messageContent = clone.querySelector('.message-content');
        messageContent.setAttribute('data-message-id', messageId);

        messagesArea.appendChild(clone);
        this.scrollToBottom();
    }

    /**
     * Append text to a message (streaming)
     */
    static appendTextToMessage(messageId, text) {
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        if (messageContent) {
            messageContent.textContent += text;
            this.scrollToBottom();
        }
    }

    /**
     * Create a tool call card
     */
    static createToolCallCard(messageId, toolData) {
        // Find the assistant message for this messageId
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageContent) {
            console.error('Message not found for tool call');
            return;
        }

        const message = messageContent.closest('.message');
        const toolContainer = message.querySelector('.tool-calls-container');

        // Clone template
        const template = document.getElementById('tool-call-card-template');
        const clone = template.content.cloneNode(true);

        const card = clone.querySelector('.tool-call-card');
        card.setAttribute('data-tool-id', toolData.tool_id);

        // Set tool icon
        const icon = this.getToolIcon(toolData.tool_name);
        card.querySelector('.tool-icon').textContent = icon;

        // Set tool name
        card.querySelector('.tool-name').textContent = toolData.tool_name;

        // Set status
        const status = card.querySelector('.tool-status');
        status.classList.add('running');
        status.textContent = 'Running...';

        // Set arguments
        const argsElement = card.querySelector('.tool-args');
        argsElement.textContent = JSON.stringify(toolData.arguments, null, 2);

        // Hide result initially
        const result = card.querySelector('.tool-result');
        result.classList.add('collapsed');

        // Set up toggle for result
        const toggle = card.querySelector('.tool-result-toggle');
        const resultHeader = card.querySelector('.tool-result-header');
        resultHeader.addEventListener('click', () => {
            result.classList.toggle('collapsed');
        });

        toolContainer.appendChild(clone);
        this.scrollToBottom();
    }

    /**
     * Update tool call result
     */
    static updateToolCallResult(toolId, result, status) {
        const card = document.querySelector(`[data-tool-id="${toolId}"]`);
        if (!card) {
            console.error('Tool card not found:', toolId);
            return;
        }

        // Update status
        const statusElement = card.querySelector('.tool-status');
        statusElement.classList.remove('running');
        statusElement.classList.add(status);
        statusElement.textContent = status === 'success' ? 'Completed' : 'Failed';

        // Update result content
        const resultContent = card.querySelector('.tool-result-content');
        resultContent.textContent = result;

        // Show result (not collapsed)
        const resultDiv = card.querySelector('.tool-result');
        resultDiv.classList.remove('collapsed');

        this.scrollToBottom();
    }

    /**
     * Update reasoning step
     */
    static updateReasoningStep(messageId, data) {
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageContent) return;

        const message = messageContent.closest('.message');
        const reasoningContainer = message.querySelector('.reasoning-container');

        // Show reasoning container
        reasoningContainer.style.display = 'block';

        // Check if reasoning panel exists
        let panel = reasoningContainer.querySelector('.reasoning-panel');
        if (!panel) {
            const template = document.getElementById('reasoning-panel-template');
            const clone = template.content.cloneNode(true);
            reasoningContainer.appendChild(clone);
            panel = reasoningContainer.querySelector('.reasoning-panel');

            // Set up toggle
            const toggle = panel.querySelector('.reasoning-toggle');
            const content = panel.querySelector('.reasoning-content');
            toggle.addEventListener('click', () => {
                panel.classList.toggle('collapsed');
            });
        }

        // Update phase
        const phase = panel.querySelector('.reasoning-phase');
        if (data.type === 'planning') {
            phase.textContent = 'Planning - Breaking down the query...';
        } else if (data.type === 'searching') {
            phase.textContent = `Step ${data.step}/${data.max_steps} - Searching for information...`;
        } else if (data.type === 'evaluating') {
            phase.textContent = 'Evaluating - Assessing completeness...';
        }

        this.scrollToBottom();
    }

    /**
     * Update reasoning goal
     */
    static updateReasoningGoal(messageId, data) {
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageContent) return;

        const message = messageContent.closest('.message');
        const panel = message.querySelector('.reasoning-panel');
        if (!panel) return;

        const goalsContainer = panel.querySelector('.reasoning-goals');
        const goalId = `goal-${data.goal_id || Math.random().toString(36).substr(2, 9)}`;

        // Check if goal exists
        let goalElement = goalsContainer.querySelector(`[data-goal-id="${goalId}"]`);
        if (!goalElement) {
            // Create new goal
            const template = document.getElementById('reasoning-goal-template');
            const clone = template.content.cloneNode(true);
            goalElement = clone.querySelector('.reasoning-goal');
            goalElement.setAttribute('data-goal-id', goalId);
            goalsContainer.appendChild(clone);
            goalElement = goalsContainer.querySelector(`[data-goal-id="${goalId}"]`);
        }

        // Update goal status class
        goalElement.className = 'reasoning-goal ' + (data.status || 'pending');

        // Update goal text
        const goalText = goalElement.querySelector('.goal-text');
        goalText.textContent = data.goal || data.description;

        // Update status icon
        const statusIcon = goalElement.querySelector('.goal-status-icon');
        if (data.status === 'completed') {
            statusIcon.textContent = '✓';
        } else if (data.status === 'in-progress') {
            statusIcon.textContent = '⋯';
        } else {
            statusIcon.textContent = '○';
        }

        // Update confidence
        if (data.confidence !== undefined) {
            const confidence = data.confidence * 100;
            const confidenceBar = goalElement.querySelector('.confidence-fill');
            confidenceBar.style.width = confidence + '%';

            const confidenceValue = goalElement.querySelector('.confidence-value');
            confidenceValue.textContent = Math.round(confidence) + '%';
        }

        // Update status badge
        const statusBadge = goalElement.querySelector('.goal-status-badge');
        if (data.status === 'completed') {
            statusBadge.textContent = 'Completed';
        } else if (data.status === 'in-progress') {
            statusBadge.textContent = 'In Progress';
        } else {
            statusBadge.textContent = 'Pending';
        }

        this.scrollToBottom();
    }

    /**
     * Update reasoning evaluation
     */
    static updateReasoningEvaluation(messageId, data) {
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageContent) return;

        const message = messageContent.closest('.message');
        const panel = message.querySelector('.reasoning-panel');
        if (!panel) return;

        const progressContainer = panel.querySelector('.reasoning-progress');

        let html = '<strong>Evaluation:</strong><br>';
        html += `Can answer: ${data.can_answer ? '✓ Yes' : '✗ Not yet'}<br>`;
        if (data.overall_confidence !== undefined) {
            const confidence = Math.round(data.overall_confidence * 100);
            html += `Overall confidence: ${confidence}%<br>`;
        }
        if (data.missing_aspects && data.missing_aspects.length > 0) {
            html += `Missing: ${data.missing_aspects.join(', ')}`;
        }

        progressContainer.innerHTML = html;
        this.scrollToBottom();
    }

    /**
     * Get tool icon based on tool name
     */
    static getToolIcon(toolName) {
        const icons = {
            'search_documents': '🔍',
            'retrieve_document_by_name': '📄',
            'list_available_documents': '📚',
            'search_wikipedia': '🌐',
            'google_custom_search': '🔎',
            'agentic_generator': '🧠'
        };
        return icons[toolName] || '🔧';
    }

    /**
     * Smooth scroll to bottom of messages
     */
    static scrollToBottom() {
        const messagesContainer = document.querySelector('.messages-container');
        if (messagesContainer) {
            messagesContainer.scrollTo({
                top: messagesContainer.scrollHeight,
                behavior: 'smooth'
            });
        }
    }

    /**
     * Escape HTML to prevent XSS
     */
    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIComponents;
}

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
        const toolContainer = message.querySelector('.tools-container');

        // Clone template
        const template = document.getElementById('tool-call-card-template');
        const clone = template.content.cloneNode(true);

        const card = clone.querySelector('.tool-card');
        card.setAttribute('data-tool-id', toolData.tool_id);
        card.setAttribute('data-tool-name', toolData.tool_name);

        // Set tool icon with SVG
        const iconElement = card.querySelector('.tool-icon');
        const icon = this.getToolIcon(toolData.tool_name);
        iconElement.innerHTML = icon;

        // Set tool name
        card.querySelector('.tool-name').textContent = this.formatToolName(toolData.tool_name);

        // Set status with loading spinner
        const status = card.querySelector('.tool-status');
        status.classList.add('running');
        status.innerHTML = '<div class="tool-loading"></div><span>Running...</span>';

        // Hide result initially
        const result = card.querySelector('.tool-result');
        result.classList.add('collapsed');

        // Set up toggle for result
        const toggle = card.querySelector('.tool-result-toggle');
        toggle.addEventListener('click', () => {
            result.classList.toggle('collapsed');
        });

        toolContainer.appendChild(clone);
        this.scrollToBottom();
    }

    /**
     * Format tool name for display
     */
    static formatToolName(toolName) {
        // Convert snake_case to Title Case
        return toolName
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
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

        // Update status - remove loading spinner
        const statusElement = card.querySelector('.tool-status');
        statusElement.classList.remove('running');
        statusElement.classList.add(status);

        if (status === 'success') {
            statusElement.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M2 7l3 3 7-7" stroke-linecap="round" stroke-linejoin="round"/>
            </svg><span>Completed</span>`;
        } else {
            statusElement.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M2 2l10 10M12 2L2 12" stroke-linecap="round"/>
            </svg><span>Failed</span>`;
        }

        // Update result content
        const resultContent = card.querySelector('.tool-result-content');
        resultContent.textContent = result;

        // Show result section (collapsed by default)
        const resultDiv = card.querySelector('.tool-result');
        resultDiv.classList.add('has-result', 'collapsed');

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
     * Get tool icon based on tool name (returns SVG)
     */
    static getToolIcon(toolName) {
        const icons = {
            'search_documents': `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="8" cy="8" r="6"/>
                <path d="M13 13l4 4" stroke-linecap="round"/>
            </svg>`,
            'retrieve_document_by_name': `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M5 3h8l4 4v10a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2z"/>
                <path d="M13 3v4h4"/>
            </svg>`,
            'list_available_documents': `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M7 6h8M7 10h8M7 14h8"/>
                <path d="M4 6h.01M4 10h.01M4 14h.01"/>
            </svg>`,
            'search_wikipedia': `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="10" cy="10" r="7"/>
                <path d="M6 10h8M10 6v8" stroke-linecap="round"/>
            </svg>`,
            'google_custom_search': `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="8" cy="8" r="6"/>
                <path d="M13 13l4 4" stroke-linecap="round"/>
            </svg>`,
            'agentic_generator': `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M10 3v14M7 7l3-3 3 3M7 13l3 3 3-3" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>`
        };
        return icons[toolName] || `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="10" cy="10" r="2"/>
            <circle cx="10" cy="10" r="7"/>
        </svg>`;
    }

    /**
     * Create a thinking panel for LRM reasoning model
     */
    static createThinkingPanel(messageId) {
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageContent) return;

        // If panel already exists (tool call continuation), reuse it
        const existing = document.querySelector(`[data-thinking-for="${messageId}"]`);
        if (existing) {
            existing.classList.remove('collapsed');
            const content = existing.querySelector('.thinking-panel-content');
            const prevText = content.getAttribute('data-thinking-text') || '';
            if (prevText) {
                content.setAttribute('data-thinking-text', prevText + '\n\n---\n\n');
            }
            const label = existing.querySelector('.thinking-panel-label');
            if (label && !label.querySelector('.thinking-spinner')) {
                const spinner = document.createElement('div');
                spinner.className = 'thinking-spinner';
                label.appendChild(spinner);
            }
            return;
        }

        const message = messageContent.closest('.message');

        // Create thinking panel
        const panel = document.createElement('div');
        panel.className = 'thinking-panel';
        panel.setAttribute('data-thinking-for', messageId);

        panel.innerHTML = `
            <div class="thinking-panel-header">
                <div class="thinking-panel-label">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M8 2C5.5 2 3.5 4 3.5 6.5c0 1.5.7 2.8 1.8 3.6.3.2.5.6.5 1v.9h4.4v-.9c0-.4.2-.8.5-1C11.8 9.3 12.5 8 12.5 6.5 12.5 4 10.5 2 8 2z" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M6 14h4M6.5 12h3" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
                    </svg>
                    <span>Thinking</span>
                    <div class="thinking-spinner"></div>
                </div>
                <svg class="thinking-chevron" width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="thinking-panel-content" data-thinking-text=""></div>
        `;

        // Toggle collapse on header click
        const header = panel.querySelector('.thinking-panel-header');
        header.addEventListener('click', () => {
            panel.classList.toggle('collapsed');
        });

        // Insert before message content
        message.insertBefore(panel, messageContent);
        this.scrollToBottom();
    }

    /**
     * Append text to a thinking panel
     */
    static appendThinkingText(messageId, text) {
        const panel = document.querySelector(`[data-thinking-for="${messageId}"]`);
        if (!panel) return;

        const content = panel.querySelector('.thinking-panel-content');
        const currentText = content.getAttribute('data-thinking-text') || '';
        const newText = currentText + text;
        content.setAttribute('data-thinking-text', newText);

        // Render as markdown
        const html = MarkdownRenderer.render(newText);
        content.innerHTML = html;
        MarkdownRenderer.renderMath(content);

        this.scrollToBottom();
    }

    /**
     * Finalize a thinking panel (remove spinner, collapse by default)
     */
    static finalizeThinkingPanel(messageId) {
        const panel = document.querySelector(`[data-thinking-for="${messageId}"]`);
        if (!panel) return;

        // Remove spinner
        const spinner = panel.querySelector('.thinking-spinner');
        if (spinner) spinner.remove();

        // Only collapse if the message area has content; if it's empty the
        // thinking panel is the only place the user can see the response.
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        const hasResponse = messageContent && (messageContent.getAttribute('data-raw-text') || '').trim();
        if (hasResponse) {
            panel.classList.add('collapsed');
        }
    }

    /**
     * Smooth scroll to bottom of messages
     */
    static scrollToBottom() {
        const messagesContainer = document.querySelector('.main-content');
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

    // ===== Waiting Messages =====

    static _waitingMessages = [
        'Searching...', 'Contemplating...', 'Brainstorming...', "Nerding...", "Main character moment...", 
        'Thinking...', 'Manifesting...', 'Questioning...', "Cooking...", "Slaying...", "Voguing...",
        'Becoming sentient...', 'Lowkey figuring it out...', 'Existential crisis mode...', 
        'Channeling big brain energy...', 'Reading the room...', 'Powering up...', 'Consulting the oracle...',
        'Spilling the tea...', 'Serving logic...', 'Flexing neural networks...', 'Hacking the mainframe...',
        'Manifesting excellence...', 'Entering the simulation...', 'A reestruturar tudo...', 
        'A pensar muito...', 'Consultando o universo...', 'A atingir o pico...',
        'A queimar neurónios...', 'A sentir a frequência...', 'Momento de génio...',
        'A meditar profundamente...', 'A conjurar ideias...', 'A sondar o terreno...', 'A improvisar...',
        'A desvendar mistérios...', 'A entrar na simulação...'
    ];
    static _waitingIntervals = {};

    /**
     * Start showing rotating waiting messages in the assistant bubble
     */
    static startWaitingMessages(messageId) {
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageContent) return;

        // Create the waiting message element
        const waitingEl = document.createElement('span');
        waitingEl.className = 'waiting-message';
        waitingEl.setAttribute('data-waiting-for', messageId);

        // Pick a random initial message
        const msgs = this._waitingMessages;
        waitingEl.textContent = msgs[Math.floor(Math.random() * msgs.length)];

        messageContent.appendChild(waitingEl);

        // Rotate messages every 5 seconds with dissolve animation
        const interval = setInterval(() => {
            waitingEl.classList.add('dissolve-out');

            setTimeout(() => {
                // Pick a different random message
                let next;
                do {
                    next = msgs[Math.floor(Math.random() * msgs.length)];
                } while (next === waitingEl.textContent && msgs.length > 1);

                waitingEl.textContent = next;
                waitingEl.classList.remove('dissolve-out');
            }, 500); // matches dissolveOut duration
        }, 5000);

        this._waitingIntervals[messageId] = interval;
    }

    /**
     * Instantly remove the waiting message (called when LLM starts streaming)
     */
    static stopWaitingMessages(messageId) {
        // Clear the rotation interval
        if (this._waitingIntervals[messageId]) {
            clearInterval(this._waitingIntervals[messageId]);
            delete this._waitingIntervals[messageId];
        }

        // Remove the element instantly
        const waitingEl = document.querySelector(`[data-waiting-for="${messageId}"]`);
        if (waitingEl) {
            waitingEl.remove();
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIComponents;
}

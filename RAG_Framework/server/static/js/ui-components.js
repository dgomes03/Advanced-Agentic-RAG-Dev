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
        // Advanced reasoning is represented by the reasoning panel itself — skip the tool card
        if (toolData.tool_name === 'activate_advanced_reasoning') return;

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
     * Get or create the reasoning panel for a message
     */
    static _getOrCreateReasoningPanel(messageId) {
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageContent) return null;

        const message = messageContent.closest('.message');
        const reasoningContainer = message.querySelector('.reasoning-container');
        reasoningContainer.style.display = 'block';

        let panel = reasoningContainer.querySelector('.reasoning-panel');
        if (!panel) {
            const template = document.getElementById('reasoning-panel-template');
            const clone = template.content.cloneNode(true);
            reasoningContainer.appendChild(clone);
            panel = reasoningContainer.querySelector('.reasoning-panel');

            const toggle = panel.querySelector('.reasoning-toggle');
            toggle.addEventListener('click', () => {
                panel.classList.toggle('collapsed');
            });
        }
        return panel;
    }

    /**
     * Create a goal element and append it to the goals container
     */
    static _createGoalElement(container, goalData, index) {
        const template = document.getElementById('reasoning-goal-template');
        const clone = template.content.cloneNode(true);
        const goalEl = clone.querySelector('.goal');

        goalEl.setAttribute('data-goal-index', index);
        goalEl.querySelector('.goal-text').textContent = goalData.description;
        goalEl.querySelector('.goal-priority-tag').textContent = `P${goalData.priority || 2}`;
        goalEl.querySelector('.goal-strategy-tag').textContent = goalData.strategy || 'hybrid';

        container.appendChild(clone);
    }

    /**
     * Build a single source card element from a source descriptor.
     * source = { tool_name, label, chars (number) | chars_text (string), result | preview }
     */
    static _createSourceItem(source) {
        const item = document.createElement('div');
        item.className = 'goal-source-item';
        item.setAttribute('data-tool-name', source.tool_name || '');

        const toolLabel = this.formatToolName(source.tool_name || 'search');
        const label = source.label || '';
        const charsText = source.chars_text
            || (source.chars !== undefined ? `· ${Number(source.chars).toLocaleString()} chars` : '');
        const content = source.result || source.preview || '';
        const isUrl = label.startsWith('http://') || label.startsWith('https://');

        const icon = this.getToolIcon(source.tool_name || '').replace(
            /width="\d+" height="\d+"/, 'width="13" height="13"'
        );

        const labelHtml = label
            ? (isUrl
                ? `<a class="goal-source-label" href="${this.escapeHtml(label)}" target="_blank" rel="noopener noreferrer">${this.escapeHtml(label)}</a>`
                : `<span class="goal-source-label">${this.escapeHtml(label)}</span>`)
            : '';

        item.innerHTML = `
            <div class="goal-source-header">
                <span class="goal-source-icon">${icon}</span>
                <span class="goal-source-tool-name">${this.escapeHtml(toolLabel)}</span>
                ${label ? '<span class="goal-source-sep">·</span>' : ''}
                ${labelHtml}
                ${charsText ? `<span class="goal-source-chars">${this.escapeHtml(charsText)}</span>` : ''}
                <svg class="goal-source-toggle" width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path d="M3 4.5l3 3 3-3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="goal-source-preview"></div>
        `;

        item.querySelector('.goal-source-preview').textContent = content;

        item.querySelector('.goal-source-header').addEventListener('click', (e) => {
            if (e.target.closest('a')) return; // don't toggle when clicking links
            item.classList.toggle('open');
        });

        return item;
    }

    /**
     * Update reasoning step — called when planning completes (step=0) or a new goal starts (step>0)
     */
    static updateReasoningStep(messageId, data) {
        const panel = this._getOrCreateReasoningPanel(messageId);
        if (!panel) return;

        // Update step counter badge
        const counter = panel.querySelector('.reasoning-step-counter');
        if (counter) {
            if (data.step === 0) {
                counter.textContent = 'Planning…';
            } else {
                counter.textContent = `Step ${data.step} / ${data.total_steps}`;
            }
        }

        const goalsContainer = panel.querySelector('.reasoning-goals');

        // Step 0: planning done — render all goals
        if (data.step === 0 && data.goals) {
            goalsContainer.innerHTML = '';
            data.goals.forEach((goal, i) => this._createGoalElement(goalsContainer, goal, i));
        } else if (data.step > 0 && data.goal_index !== undefined) {
            // Mark current goal as in-progress
            const goalEl = goalsContainer.querySelector(`[data-goal-index="${data.goal_index}"]`);
            if (goalEl) {
                goalEl.className = 'goal in-progress';
            }
        }

        this.scrollToBottom();
    }

    /**
     * Show source cards for a goal after retrieval completes.
     * Uses data.sources (array) when available; falls back to legacy single-tool fields.
     */
    static updateReasoningRetrieval(messageId, data) {
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageContent) return;

        const message = messageContent.closest('.message');
        const panel = message.querySelector('.reasoning-panel');
        if (!panel) return;

        const goalIndex = data.goal_index !== undefined ? data.goal_index : 0;
        const goalEl = panel.querySelector(`[data-goal-index="${goalIndex}"]`);
        if (!goalEl) return;

        const sourcesContainer = goalEl.querySelector('.goal-sources');
        if (!sourcesContainer) return;

        // Build sources list — prefer new `sources` field, fall back to legacy
        const sources = (data.sources && data.sources.length > 0)
            ? data.sources
            : [{
                tool_name: data.tool_name || 'search',
                label: '',
                chars: data.chars_retrieved || 0,
                result: data.preview || ''
            }];

        sources.forEach(source => {
            sourcesContainer.appendChild(this._createSourceItem(source));
        });

        this.scrollToBottom();
    }

    /**
     * Show evaluation scores for a goal
     */
    static updateReasoningEvaluation(messageId, data) {
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageContent) return;

        const message = messageContent.closest('.message');
        const panel = message.querySelector('.reasoning-panel');
        if (!panel) return;

        const goalIndex = data.goal_index !== undefined ? data.goal_index : 0;
        const goalEl = panel.querySelector(`[data-goal-index="${goalIndex}"]`);
        if (!goalEl) return;

        // Mark goal completed
        goalEl.className = 'goal completed';

        // Show scores section
        const scoresEl = goalEl.querySelector('.goal-scores');
        if (scoresEl) {
            scoresEl.style.display = 'block';

            const confidence = Math.round((data.confidence || 0) * 100);
            goalEl.querySelector('.confidence-fill').style.width = confidence + '%';
            goalEl.querySelector('.confidence-value').textContent = confidence + '%';

            const infoGain = Math.round((data.information_gain || 0) * 100);
            goalEl.querySelector('.info-gain-fill').style.width = infoGain + '%';
            goalEl.querySelector('.info-gain-value').textContent = infoGain + '%';

            // Quality flags
            const flagsEl = scoresEl.querySelector('.goal-flags');
            if (flagsEl) {
                flagsEl.innerHTML = '';
                const isSparse = data.sparse_results || (data.confidence < 0.7);
                if (isSparse) {
                    flagsEl.innerHTML += '<span class="goal-flag sparse">⚠ Sparse</span>';
                }
                if (data.contradictory_info) {
                    flagsEl.innerHTML += '<span class="goal-flag contradictory">⚡ Contradictory</span>';
                }
                if (!isSparse && !data.contradictory_info) {
                    flagsEl.innerHTML += '<span class="goal-flag ok">✓ Good quality</span>';
                }
            }

            // Evaluation reasoning
            if (data.reasoning) {
                const reasoningEl = scoresEl.querySelector('.goal-reasoning');
                if (reasoningEl) reasoningEl.textContent = `"${data.reasoning}"`;
            }
        }

        this.scrollToBottom();
    }

    /**
     * Add replanned goals to the panel
     */
    static updateReasoningReplan(messageId, data) {
        const messageContent = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageContent) return;

        const message = messageContent.closest('.message');
        const panel = message.querySelector('.reasoning-panel');
        if (!panel) return;

        const goalsContainer = panel.querySelector('.reasoning-goals');
        if (!goalsContainer || !data.new_goals || !data.new_goals.length) return;

        const existingCount = goalsContainer.querySelectorAll('.goal').length;
        data.new_goals.forEach((goalDesc, i) => {
            const goalData = typeof goalDesc === 'string'
                ? { description: goalDesc, priority: 2, strategy: 'hybrid' }
                : goalDesc;
            this._createGoalElement(goalsContainer, goalData, existingCount + i);
        });

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
     * Restore a saved reasoning panel when loading a past conversation
     */
    static restoreReasoningPanel(messageId, reasoningData) {
        const panel = this._getOrCreateReasoningPanel(messageId);
        if (!panel) return;

        const counter = panel.querySelector('.reasoning-step-counter');
        if (counter && reasoningData.step_counter) {
            counter.textContent = reasoningData.step_counter;
        }

        // Collapse by default when loading from history
        panel.classList.add('collapsed');

        const goalsContainer = panel.querySelector('.reasoning-goals');
        goalsContainer.innerHTML = '';

        (reasoningData.goals || []).forEach(g => {
            const template = document.getElementById('reasoning-goal-template');
            const clone = template.content.cloneNode(true);
            const goalEl = clone.querySelector('.goal');

            goalEl.setAttribute('data-goal-index', g.index);
            goalEl.className = g.status ? `goal ${g.status}` : 'goal';
            goalEl.querySelector('.goal-text').textContent = g.description;
            goalEl.querySelector('.goal-priority-tag').textContent = g.priority_tag;
            goalEl.querySelector('.goal-strategy-tag').textContent = g.strategy_tag;

            // Restore source cards — migrate old single-tool format if needed
            const sourcesContainer = goalEl.querySelector('.goal-sources');
            if (sourcesContainer) {
                const sources = g.sources
                    || (g.tool_name ? [{
                        tool_name: g.tool_name,
                        label: '',
                        chars_text: g.chars_retrieved || '',
                        preview: g.preview || ''
                    }] : []);
                sources.forEach(s => sourcesContainer.appendChild(this._createSourceItem(s)));
            }

            if (g.scores) {
                const scoresEl = goalEl.querySelector('.goal-scores');
                scoresEl.style.display = 'block';
                goalEl.querySelector('.confidence-fill').style.width = g.scores.confidence_width;
                goalEl.querySelector('.confidence-value').textContent = g.scores.confidence_text;
                goalEl.querySelector('.info-gain-fill').style.width = g.scores.info_gain_width;
                goalEl.querySelector('.info-gain-value').textContent = g.scores.info_gain_text;
                if (g.scores.flags_html) {
                    goalEl.querySelector('.goal-flags').innerHTML = g.scores.flags_html;
                }
                if (g.scores.reasoning) {
                    goalEl.querySelector('.goal-reasoning').textContent = g.scores.reasoning;
                }
            }

            goalsContainer.appendChild(clone);
        });
    }

    /**
     * Smooth scroll to bottom of messages — only if user is already near the bottom.
     * This prevents hijacking the scroll position when the user has scrolled up to read.
     */
    static scrollToBottom() {
        const messagesContainer = document.querySelector('.main-content');
        if (messagesContainer) {
            const { scrollTop, scrollHeight, clientHeight } = messagesContainer;
            const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
            if (distanceFromBottom < 150) {
                messagesContainer.scrollTo({
                    top: messagesContainer.scrollHeight,
                    behavior: 'smooth'
                });
            }
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
        'A desvendar mistérios...', 'A entrar na simulação...', "Skibidi... skibidi toilet...",
        "I'm still working, dont worry...", "I'm not frozen, dont worry...", "Aguente mais um pouco...",
        "Estou a ler imensa coisa, tenha paciência...", "I'm calculating Attention(Q,K,V)=softmax(QK⊤√dk)V...",
        "Reading..."
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

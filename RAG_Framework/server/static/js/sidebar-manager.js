/**
 * Sidebar Manager for RAG Framework
 * Handles conversation list, navigation, and persistence
 */

class SidebarManager {
    constructor() {
        this.currentChatId = null;
        this.conversations = [];
        this.searchQuery = '';
        this.isInitialized = false;
    }

    /**
     * Initialize the sidebar manager
     */
    initialize() {
        if (this.isInitialized) return;

        // Wait for ragClient to be available
        if (!window.ragClient) {
            setTimeout(() => this.initialize(), 100);
            return;
        }

        this.setupEventListeners();
        this.setupSocketListeners();
        this.loadConversations();
        this.isInitialized = true;

        console.log('Sidebar Manager initialized');
    }

    /**
     * Set up DOM event listeners
     */
    setupEventListeners() {
        // New chat button (sidebar)
        const newBtn = document.getElementById('sidebar-new-btn');
        if (newBtn) {
            newBtn.addEventListener('click', () => this.createNewChat());
        }

        // Search input
        const searchInput = document.getElementById('sidebar-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchQuery = e.target.value.toLowerCase();
                this.renderConversations();
            });
        }

        // Sidebar toggle (mobile)
        const toggleBtn = document.getElementById('sidebar-toggle');
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebar-overlay');

        if (toggleBtn && sidebar) {
            toggleBtn.addEventListener('click', () => this.toggleSidebar());
        }

        if (overlay) {
            overlay.addEventListener('click', () => this.closeSidebar());
        }

        // Sidebar resize handle
        this.setupResizeHandle();
    }

    /**
     * Set up sidebar resize functionality
     */
    setupResizeHandle() {
        const resizeHandle = document.getElementById('sidebar-resize-handle');
        const sidebar = document.getElementById('sidebar');

        if (!resizeHandle || !sidebar) return;

        let isResizing = false;
        let startX = 0;
        let startWidth = 0;

        const startResize = (e) => {
            isResizing = true;
            startX = e.clientX;
            startWidth = sidebar.offsetWidth;
            resizeHandle.classList.add('dragging');
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
            e.preventDefault();
        };

        const doResize = (e) => {
            if (!isResizing) return;

            const diff = e.clientX - startX;
            const newWidth = Math.min(Math.max(startWidth + diff, 200), 500);
            sidebar.style.width = newWidth + 'px';

            // Save to localStorage
            localStorage.setItem('sidebar-width', newWidth);
        };

        const stopResize = () => {
            if (!isResizing) return;
            isResizing = false;
            resizeHandle.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };

        resizeHandle.addEventListener('mousedown', startResize);
        document.addEventListener('mousemove', doResize);
        document.addEventListener('mouseup', stopResize);

        // Touch support for mobile
        resizeHandle.addEventListener('touchstart', (e) => {
            const touch = e.touches[0];
            startResize({ clientX: touch.clientX, preventDefault: () => {} });
        });

        document.addEventListener('touchmove', (e) => {
            if (!isResizing) return;
            const touch = e.touches[0];
            doResize({ clientX: touch.clientX });
        });

        document.addEventListener('touchend', stopResize);

        // Restore saved width
        const savedWidth = localStorage.getItem('sidebar-width');
        if (savedWidth) {
            const width = parseInt(savedWidth, 10);
            if (width >= 200 && width <= 500) {
                sidebar.style.width = width + 'px';
            }
        }
    }

    /**
     * Set up WebSocket event listeners
     */
    setupSocketListeners() {
        const socket = window.ragClient.socket;

        socket.on('chat_saved', (data) => {
            console.log('Chat saved:', data);
            if (data.is_new) {
                this.currentChatId = data.chat_id;
            }
            this.loadConversations();
        });

        socket.on('chat_loaded', (data) => {
            console.log('Chat loaded:', data);
            this.displayLoadedChat(data.chat);
        });
    }

    /**
     * Toggle sidebar visibility (mobile)
     */
    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebar-overlay');

        if (sidebar.classList.contains('open')) {
            this.closeSidebar();
        } else {
            sidebar.classList.add('open');
            overlay.classList.add('visible');
        }
    }

    /**
     * Close sidebar (mobile)
     */
    closeSidebar() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebar-overlay');

        sidebar.classList.remove('open');
        overlay.classList.remove('visible');
    }

    /**
     * Load all conversations from the API
     */
    async loadConversations() {
        try {
            const response = await fetch('/api/chats');
            if (response.ok) {
                this.conversations = await response.json();
                this.renderConversations();
            }
        } catch (error) {
            console.error('Failed to load conversations:', error);
        }
    }

    /**
     * Render conversations in the sidebar
     */
    renderConversations() {
        const container = document.getElementById('conversations-list');
        if (!container) return;

        // Filter conversations based on search query
        let filtered = this.conversations;
        if (this.searchQuery) {
            filtered = this.conversations.filter(conv =>
                conv.title.toLowerCase().includes(this.searchQuery) ||
                (conv.preview && conv.preview.toLowerCase().includes(this.searchQuery))
            );
        }

        if (filtered.length === 0) {
            container.innerHTML = '<div class="conversations-empty">No conversations yet</div>';
            return;
        }

        const template = document.getElementById('conversation-item-template');
        container.innerHTML = '';

        filtered.forEach(conv => {
            const clone = template.content.cloneNode(true);
            const item = clone.querySelector('.conversation-item');

            item.dataset.chatId = conv.id;
            item.querySelector('.conversation-title').textContent = conv.title || 'Untitled';
            item.querySelector('.conversation-preview').textContent = conv.preview || '';
            item.querySelector('.conversation-timestamp').textContent = this.formatTimestamp(conv.updated_at);

            if (conv.id === this.currentChatId) {
                item.classList.add('active');
            }

            // Click to load conversation
            item.addEventListener('click', (e) => {
                if (!e.target.closest('.conversation-delete')) {
                    this.loadChat(conv.id);
                    this.closeSidebar();
                }
            });

            // Delete button
            const deleteBtn = item.querySelector('.conversation-delete');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteChat(conv.id);
            });

            container.appendChild(clone);
        });
    }

    /**
     * Format timestamp for display
     */
    formatTimestamp(isoString) {
        if (!isoString) return '';

        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;

        return date.toLocaleDateString();
    }

    /**
     * Create a new chat
     */
    createNewChat() {
        // Abort any ongoing generation
        if (window.ragClient && ragClient.socket) {
            ragClient.socket.emit('abort_generation', {});
            ragClient.currentMessageId = null;
            ragClient.isProcessing = false;
            ragClient.socket.disconnect();

            setTimeout(() => {
                ragClient.socket.connect();
                ragClient.socket.once('connect', () => {
                    ragClient.socket.emit('clear_cache', {});
                    ragClient.updateSendButton(true);
                    ragClient.updateStatusIndicator('connected', 'Ready');
                });
            }, 100);
        }

        // Clear current chat ID
        this.currentChatId = null;

        // Clear UI and show welcome
        this.clearMessages();

        // Update active state in sidebar
        this.renderConversations();

        console.log('New conversation started');
    }

    /**
     * Load a chat by ID
     */
    async loadChat(chatId) {
        try {
            // Abort any ongoing generation first
            if (window.ragClient && ragClient.isProcessing) {
                ragClient.socket.emit('abort_generation', {});
                ragClient.currentMessageId = null;
                ragClient.isProcessing = false;
            }

            const response = await fetch(`/api/chats/${chatId}`);
            if (response.ok) {
                const chat = await response.json();
                this.currentChatId = chatId;
                this.displayLoadedChat(chat);
                this.renderConversations();

                // Restore server-side KV-cache and conversation manager
                if (window.ragClient) {
                    ragClient.socket.emit('restore_chat', {
                        chat_id: chatId,
                        messages: chat.messages
                    });
                }
            }
        } catch (error) {
            console.error('Failed to load chat:', error);
        }
    }

    /**
     * Display a loaded chat in the UI
     */
    displayLoadedChat(chat) {
        this.clearMessages();

        const messagesArea = document.getElementById('messages');

        chat.messages.forEach(msg => {
            if (msg.role === 'user') {
                UIComponents.createUserMessage(msg.content);
            } else if (msg.role === 'assistant') {
                const msgId = 'loaded-' + Date.now() + Math.random();
                UIComponents.createAssistantMessage(msgId);

                const messageContent = document.querySelector(`[data-message-id="${msgId}"]`);
                if (messageContent) {
                    messageContent.setAttribute('data-raw-text', msg.content);
                    const html = MarkdownRenderer.render(msg.content);
                    messageContent.innerHTML = html;
                    MarkdownRenderer.renderMath(messageContent);
                }
            }
        });

        // Scroll to bottom
        UIComponents.scrollToBottom();
    }

    /**
     * Delete a chat
     */
    async deleteChat(chatId) {
        if (!confirm('Delete this conversation?')) return;

        try {
            const response = await fetch(`/api/chats/${chatId}`, { method: 'DELETE' });
            if (response.ok) {
                // If deleting current chat, clear UI
                if (chatId === this.currentChatId) {
                    this.currentChatId = null;
                    this.clearMessages();
                }
                this.loadConversations();
            }
        } catch (error) {
            console.error('Failed to delete chat:', error);
        }
    }

    /**
     * Save the current chat
     */
    saveCurrentChat() {
        if (!window.ragClient) return;

        const messages = this.collectMessagesFromUI();
        if (messages.length === 0) return;

        window.ragClient.socket.emit('save_chat', {
            chat_id: this.currentChatId,
            messages: messages
        });
    }

    /**
     * Collect messages from the UI
     */
    collectMessagesFromUI() {
        const messages = [];
        const messageElements = document.querySelectorAll('.message');

        messageElements.forEach(el => {
            const content = el.querySelector('.message-content');
            if (!content) return;

            let text = '';
            if (el.classList.contains('message-user')) {
                text = content.textContent || '';
                if (text) {
                    messages.push({ role: 'user', content: text });
                }
            } else if (el.classList.contains('message-assistant')) {
                // Get raw text if available, otherwise get text content
                text = content.getAttribute('data-raw-text') || content.textContent || '';
                if (text) {
                    messages.push({ role: 'assistant', content: text });
                }
            }
        });

        return messages;
    }

    /**
     * Clear messages and show welcome screen
     */
    clearMessages() {
        const messagesArea = document.getElementById('messages');
        messagesArea.innerHTML = `
            <div class="welcome">
                <h1 class="welcome-title">What can I help with?</h1>
                <p class="welcome-subtitle">Search your documents, browse the web, query databases, and reason through complex problems.</p>
                <div class="capabilities">
                    <div class="capability">
                        <div class="capability-icon">
                            <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                                <circle cx="8" cy="8" r="5.5" stroke="currentColor" stroke-width="1.5"/>
                                <path d="M12 12l5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                            </svg>
                        </div>
                        <span>Document Search</span>
                    </div>
                    <div class="capability">
                        <div class="capability-icon">
                            <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                                <path d="M5 3h10a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2z" stroke="currentColor" stroke-width="1.5"/>
                                <path d="M7 7h6M7 10h6M7 13h4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                            </svg>
                        </div>
                        <span>Document Retrieval</span>
                    </div>
                    <div class="capability">
                        <div class="capability-icon">
                            <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                                <circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="1.5"/>
                                <path d="M6 10h8M10 6v8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                            </svg>
                        </div>
                        <span>Web Search</span>
                    </div>
                    <div class="capability">
                        <div class="capability-icon">
                            <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                                <path d="M3 6h14M3 10h14M3 14h10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                            </svg>
                        </div>
                        <span>URL Fetching</span>
                    </div>
                    <div class="capability">
                        <div class="capability-icon">
                            <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                                <ellipse cx="10" cy="10" rx="8" ry="4" stroke="currentColor" stroke-width="1.5"/>
                                <path d="M2 10v3c0 2.2 3.6 4 8 4s8-1.8 8-4v-3" stroke="currentColor" stroke-width="1.5"/>
                                <path d="M2 7v3" stroke="currentColor" stroke-width="1.5"/>
                                <path d="M18 7v3" stroke="currentColor" stroke-width="1.5"/>
                            </svg>
                        </div>
                        <span>SQL Databases</span>
                    </div>
                    <div class="capability">
                        <div class="capability-icon">
                            <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                                <path d="M10 3l2 5h5l-4 3 1.5 5-4.5-3-4.5 3 1.5-5-4-3h5l2-5z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <span>Advanced Reasoning</span>
                    </div>
                </div>
            </div>
        `;
    }
}

// Global instance
let sidebarManager;

document.addEventListener('DOMContentLoaded', () => {
    sidebarManager = new SidebarManager();
    sidebarManager.initialize();
    window.sidebarManager = sidebarManager;
});

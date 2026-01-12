/**
 * Markdown Renderer for RAG Framework
 * Converts markdown to HTML and renders LaTeX math with KaTeX
 */

class MarkdownRenderer {
    /**
     * Render markdown text to HTML
     */
    static render(text) {
        if (!text) return '';

        // Remove tool call syntax from text
        let html = text;

        // Remove [TOOL_CALLS]tool_name[ARGS]{...} patterns and add paragraph breaks
        html = html.replace(/\[TOOL_CALLS\][^\[]*\[ARGS\][^\n]*/g, '\n\n');

        // Extract and protect LaTeX math BEFORE HTML escaping
        const mathPlaceholders = [];

        // Extract display math \[...\]
        html = html.replace(/\\\[([\s\S]*?)\\\]/g, (match, math) => {
            const id = mathPlaceholders.length;
            mathPlaceholders.push({ type: 'display', math: math.trim() });
            return `__MATH_DISPLAY_${id}__`;
        });

        // Extract display math $$...$$
        html = html.replace(/\$\$([\s\S]*?)\$\$/g, (match, math) => {
            const id = mathPlaceholders.length;
            mathPlaceholders.push({ type: 'display', math: math.trim() });
            return `__MATH_DISPLAY_${id}__`;
        });

        // Extract inline math \(...\)
        html = html.replace(/\\\(([\s\S]*?)\\\)/g, (match, math) => {
            const id = mathPlaceholders.length;
            mathPlaceholders.push({ type: 'inline', math: math.trim() });
            return `__MATH_INLINE_${id}__`;
        });

        // Extract inline math $...$
        html = html.replace(/\$([^$\n]+)\$/g, (match, math) => {
            const id = mathPlaceholders.length;
            mathPlaceholders.push({ type: 'inline', math: math.trim() });
            return `__MATH_INLINE_${id}__`;
        });

        // Now escape HTML to prevent XSS
        html = this.escapeHtml(html);

        // Process code blocks first (to prevent interference with other processing)
        html = this.processCodeBlocks(html);

        // Process inline code
        html = this.processInlineCode(html);

        // Restore LaTeX math with proper spans
        mathPlaceholders.forEach((item, id) => {
            const className = item.type === 'display' ? 'math-display' : 'math-inline';
            const span = `<span class="${className}" data-math="${this.escapeHtml(item.math)}"></span>`;
            html = html.replace(`__MATH_${item.type.toUpperCase()}_${id}__`, span);
        });

        // Process horizontal rules (before other processing)
        html = this.processHorizontalRules(html);

        // Process headings
        html = this.processHeadings(html);

        // Process bold and italic
        html = this.processBoldItalic(html);

        // Process lists
        html = this.processLists(html);

        // Process blockquotes
        html = this.processBlockquotes(html);

        // Process links
        html = this.processLinks(html);

        // Process line breaks and paragraphs
        html = this.processParagraphs(html);

        return html;
    }

    /**
     * Escape HTML characters
     */
    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Process code blocks (```code```)
     */
    static processCodeBlocks(text) {
        return text.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const language = lang || 'text';
            return `<pre><code class="language-${language}">${code.trim()}</code></pre>`;
        });
    }

    /**
     * Process inline code (`code`)
     */
    static processInlineCode(text) {
        return text.replace(/`([^`]+)`/g, '<code>$1</code>');
    }

    /**
     * Process horizontal rules (---, ***, ___)
     */
    static processHorizontalRules(text) {
        // Match --- or *** or ___ on their own line
        return text.replace(/^(?:---|\*\*\*|___)$/gm, '<hr>');
    }


    /**
     * Process headings (# heading)
     */
    static processHeadings(text) {
        // H1-H6
        return text.replace(/^(#{1,6})\s+(.+)$/gm, (match, hashes, content) => {
            const level = hashes.length;
            return `<h${level}>${content}</h${level}>`;
        });
    }

    /**
     * Process bold and italic
     */
    static processBoldItalic(text) {
        // Bold: **text** or __text__
        text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/__(.+?)__/g, '<strong>$1</strong>');

        // Italic: *text* or _text_
        text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
        text = text.replace(/_(.+?)_/g, '<em>$1</em>');

        return text;
    }

    /**
     * Process lists
     */
    static processLists(text) {
        // Unordered lists
        text = text.replace(/^[\*\-]\s+(.+)$/gm, '<li>$1</li>');

        // Ordered lists
        text = text.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

        // Wrap consecutive <li> in <ul> or <ol>
        text = text.replace(/(<li>.*<\/li>\n?)+/g, (match) => {
            return `<ul>${match}</ul>`;
        });

        return text;
    }

    /**
     * Process blockquotes (> quote)
     */
    static processBlockquotes(text) {
        return text.replace(/^&gt;\s+(.+)$/gm, '<blockquote>$1</blockquote>');
    }

    /**
     * Process links [text](url)
     */
    static processLinks(text) {
        return text.replace(/\[([^\]]+)\]\(([^\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    }

    /**
     * Process paragraphs and line breaks
     */
    static processParagraphs(text) {
        // Split by double newlines for paragraphs
        const paragraphs = text.split(/\n\n+/);

        return paragraphs.map(para => {
            // Don't wrap if already has block-level tags
            if (para.match(/^<(h[1-6]|pre|ul|ol|blockquote)/)) {
                return para;
            }
            // Replace single newlines with <br>
            para = para.replace(/\n/g, '<br>');
            return `<p>${para}</p>`;
        }).join('\n');
    }

    /**
     * Render math with KaTeX
     */
    static renderMath(element) {
        if (typeof katex === 'undefined') {
            console.warn('KaTeX not loaded - skipping math rendering');
            return;
        }

        // Render display math
        const displayMath = element.querySelectorAll('.math-display');
        displayMath.forEach((span) => {
            const math = span.getAttribute('data-math');
            try {
                // Clear existing content
                span.innerHTML = '';
                // Render with KaTeX
                katex.render(math, span, {
                    displayMode: true,
                    throwOnError: false,
                    strict: false,
                    trust: false
                });
            } catch (e) {
                console.error('KaTeX display render error:', e);
                span.textContent = `$$${math}$$`;
            }
        });

        // Render inline math
        const inlineMath = element.querySelectorAll('.math-inline');
        inlineMath.forEach((span) => {
            const math = span.getAttribute('data-math');
            try {
                // Clear existing content
                span.innerHTML = '';
                // Render with KaTeX
                katex.render(math, span, {
                    displayMode: false,
                    throwOnError: false,
                    strict: false,
                    trust: false
                });
            } catch (e) {
                console.error('KaTeX inline render error:', e);
                span.textContent = `$${math}$`;
            }
        });
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MarkdownRenderer;
}

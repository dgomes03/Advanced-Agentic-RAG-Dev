/**
 * Markdown Renderer for RAG Framework
 * Uses marked.js for markdown, KaTeX for math
 */

class MarkdownRenderer {
    static render(text) {
        if (!text) return '';

        // 1. Remove tool call syntax
        let html = this.removeToolCallSyntax(text);

        // 2. Protect LaTeX math before marked processes it
        const mathPlaceholders = [];
        html = this.extractMath(html, mathPlaceholders);

        // 3. Use marked.js for markdown -> HTML
        html = marked.parse(html, {
            gfm: true,        // GitHub Flavored Markdown (tables, etc)
            breaks: true,     // Convert \n to <br>
        });

        // 4. Restore math placeholders
        html = this.restoreMath(html, mathPlaceholders);

        return html;
    }

    static removeToolCallSyntax(text) {
        // Remove [TOOL_CALLS]...[ARGS]{...} patterns
        return text.replace(/\[TOOL_CALLS\][^\[]*\[ARGS\]\{[^}]*\}/g, '\n\n')
                   .replace(/\n{3,}/g, '\n\n');
    }

    static extractMath(text, placeholders) {
        // Display math: $$...$$ or \[...\]
        text = text.replace(/\$\$([\s\S]*?)\$\$/g, (m, math) => {
            placeholders.push({ type: 'display', math: math.trim() });
            return `MATHPLACEHOLDER_D_${placeholders.length - 1}_END`;
        });
        text = text.replace(/\\\[([\s\S]*?)\\\]/g, (m, math) => {
            placeholders.push({ type: 'display', math: math.trim() });
            return `MATHPLACEHOLDER_D_${placeholders.length - 1}_END`;
        });

        // Inline math: $...$ or \(...\) (skip currency like $100)
        text = text.replace(/\\\(([\s\S]*?)\\\)/g, (m, math) => {
            placeholders.push({ type: 'inline', math: math.trim() });
            return `MATHPLACEHOLDER_I_${placeholders.length - 1}_END`;
        });
        text = text.replace(/\$([^$\n]+)\$/g, (m, math) => {
            if (/^\d+([.,]\d+)?$/.test(math.trim())) return m; // currency
            placeholders.push({ type: 'inline', math: math.trim() });
            return `MATHPLACEHOLDER_I_${placeholders.length - 1}_END`;
        });

        return text;
    }

    static restoreMath(html, placeholders) {
        placeholders.forEach((item, i) => {
            const cls = item.type === 'display' ? 'math-display' : 'math-inline';
            const placeholder = `MATHPLACEHOLDER_${item.type === 'display' ? 'D' : 'I'}_${i}_END`;
            const escaped = item.math.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
            html = html.replace(placeholder, `<span class="${cls}" data-math="${escaped}"></span>`);
        });
        return html;
    }

    static renderMath(element) {
        if (typeof katex === 'undefined') return;

        element.querySelectorAll('.math-display, .math-inline').forEach(span => {
            const math = span.getAttribute('data-math');
            const isDisplay = span.classList.contains('math-display');
            try {
                katex.render(math, span, {
                    displayMode: isDisplay,
                    throwOnError: false,
                    strict: false,
                });
            } catch (e) {
                span.textContent = isDisplay ? `$$${math}$$` : `$${math}$`;
            }
        });
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = MarkdownRenderer;
}

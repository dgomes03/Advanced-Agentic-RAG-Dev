/**
 * Markdown Renderer for RAG Framework
 * Uses marked.js for markdown, KaTeX for math
 */

class MarkdownRenderer {
    static render(text) {
        if (!text) return '';

        // 1. Remove tool call syntax
        let html = this.removeToolCallSyntax(text);

        // 2. Fix headings missing space after # (e.g. ###Title -> ### Title)
        html = html.replace(/^(#{1,6})([^\s#])/gm, '$1 $2');

        // 3. Protect LaTeX math before marked processes it
        const mathPlaceholders = [];
        html = this.extractMath(html, mathPlaceholders);

        // 4. Use marked.js for markdown -> HTML
        html = marked.parse(html, {
            gfm: true,        // GitHub Flavored Markdown (tables, etc)
            breaks: true,     // Convert \n to <br>
        });

        // 5. Restore math placeholders
        html = this.restoreMath(html, mathPlaceholders);

        return html;
    }

    static removeToolCallSyntax(text) {
        // Remove [THINK]...[/THINK] reasoning markers if they leak through
        let result = text.replace(/\[THINK\][\s\S]*?\[\/THINK\]/g, '');
        // Also strip partial/unclosed [THINK] at end of text
        result = result.replace(/\[THINK\][\s\S]*$/, '');

        while (result.includes('[TOOL_CALLS]')) {
            const start = result.indexOf('[TOOL_CALLS]');
            const afterMarker = result.substring(start + 12);
            const argsIdx = afterMarker.indexOf('[ARGS]');

            if (argsIdx === -1) {
                // No [ARGS] — check for old format [TOOL_CALLS][{...}]
                if (afterMarker.startsWith('[')) {
                    let depth = 0, end = -1;
                    for (let i = 0; i < afterMarker.length; i++) {
                        if (afterMarker[i] === '[') depth++;
                        else if (afterMarker[i] === ']') { depth--; if (depth === 0) { end = i + 1; break; } }
                    }
                    if (end > 0) { result = result.substring(0, start) + result.substring(start + 12 + end); continue; }
                }
                // Partial pattern (streaming) — strip from marker to end
                result = result.substring(0, start);
                continue;
            }

            // Find JSON object after [ARGS] (skip garbage text before '{')
            const afterArgs = afterMarker.substring(argsIdx + 6);
            const braceIdx = afterArgs.indexOf('{');

            if (braceIdx === -1) {
                result = result.substring(0, start);
                continue;
            }

            // Match balanced braces to find end of JSON
            let depth = 0, jsonEnd = -1;
            for (let i = braceIdx; i < afterArgs.length; i++) {
                if (afterArgs[i] === '{') depth++;
                else if (afterArgs[i] === '}') { depth--; if (depth === 0) { jsonEnd = i + 1; break; } }
            }

            if (jsonEnd === -1) {
                // Unbalanced braces (still streaming) — strip to end
                result = result.substring(0, start);
                continue;
            }

            // Remove the complete tool call pattern
            const endInResult = start + 12 + argsIdx + 6 + jsonEnd;
            result = result.substring(0, start) + result.substring(endInResult);
        }

        return result.replace(/\n{3,}/g, '\n\n');
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

        // Inline math: $...$ or \(...\)
        text = text.replace(/\\\(([\s\S]*?)\\\)/g, (m, math) => {
            placeholders.push({ type: 'inline', math: math.trim() });
            return `MATHPLACEHOLDER_I_${placeholders.length - 1}_END`;
        });
        // $...$ with guards: cap length, skip currency ($44, $3.50, $44 billion)
        text = text.replace(/\$([^$\n]{1,100})\$/g, (m, math) => {
            const trimmed = math.trim();
            if (!trimmed) return m;
            if (/^\d/.test(trimmed) && !/[\\^_{}]/.test(trimmed)) return m; // currency unless it has LaTeX chars
            placeholders.push({ type: 'inline', math: trimmed });
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

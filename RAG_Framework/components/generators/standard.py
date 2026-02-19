import json
import random
import string
from mlx_lm import generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from RAG_Framework.core.config import MAX_RESPONSE_TOKENS, ADVANCED_REASONING, GOOGLE_CX, GOOGLE_API_KEY
from RAG_Framework.core.cache_manager import CacheManager
from RAG_Framework.core.conversation_manager import ConversationManager


_SKIP_SPECIAL = {'<s>', '</s>', '<unk>', '<pad>'}


class Generator:
    _byte_decoder = None  # Lazy-initialized inverse of GPT-2 bytes_to_unicode
    _inv_vocab = None     # Lazy-initialized inverse vocabulary {id: string}
    _special_ids = None   # Lazy-initialized set of special token IDs

    @staticmethod
    def _build_byte_decoder():
        """Build the inverse of the GPT-2 byte-level BPE bytes_to_unicode mapping."""
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return {chr(c): b for b, c in zip(bs, cs)}

    @staticmethod
    def _ensure_vocab(tokenizer):
        """Lazily build and cache inverse vocabulary and special token set."""
        if Generator._byte_decoder is None:
            Generator._byte_decoder = Generator._build_byte_decoder()
        if Generator._inv_vocab is None:
            vocab = tokenizer.get_vocab()
            Generator._inv_vocab = {v: k for k, v in vocab.items()}
            special = set(getattr(tokenizer, 'all_special_ids', []))
            if hasattr(tokenizer, 'added_tokens_encoder'):
                special.update(tokenizer.added_tokens_encoder.values())
            Generator._special_ids = special

    @staticmethod
    def _decode_tokens(tokenizer, tokens):
        """Decode token IDs to text, bypassing tokenizer.decode() entirely.

        Handles tokenizers where the BPE vocabulary uses Ġ (U+0120) as the
        space marker but the decoder expects ▁ (U+2581), which causes
        tokenizer.decode() to strip all spaces from output.
        """
        if not tokens:
            return ""
        Generator._ensure_vocab(tokenizer)

        inv_vocab = Generator._inv_vocab
        special_ids = Generator._special_ids
        byte_decoder = Generator._byte_decoder

        parts = []
        byte_buf = bytearray()

        for tok_id in tokens:
            tok_str = inv_vocab.get(tok_id, '')

            if tok_id in special_ids:
                if byte_buf:
                    parts.append(byte_buf.decode('utf-8', errors='ignore'))
                    byte_buf = bytearray()
                if tok_str not in _SKIP_SPECIAL:
                    parts.append(tok_str)
            else:
                for c in tok_str:
                    if c == '\u2581':
                        byte_buf.append(0x20)
                    elif c in byte_decoder:
                        byte_buf.append(byte_decoder[c])
                    else:
                        byte_buf.extend(c.encode('utf-8'))

        if byte_buf:
            parts.append(byte_buf.decode('utf-8', errors='ignore'))

        return ''.join(parts)

    @staticmethod
    def _fix_bpe_artifacts(text):
        """Post-process a string returned by generate() to fix BPE decoding artifacts.

        When tokenizer.decode() has a decoder mismatch (e.g. Ministral), generate()
        returns raw BPE chars like Ġ (space), Ċ (newline), ðŁĺĬ (emoji bytes).
        Applies the GPT-2 byte decoder char-by-char to recover proper UTF-8 text.
        Only activates when Ġ (U+0120) or ▁ (U+2581) artifacts are detected.
        """
        if '\u0120' not in text and '\u2581' not in text:
            return text
        if Generator._byte_decoder is None:
            Generator._byte_decoder = Generator._build_byte_decoder()
        byte_decoder = Generator._byte_decoder
        byte_buf = bytearray()
        parts = []
        for c in text:
            if c == '\u2581':
                byte_buf.append(0x20)
            elif c in byte_decoder:
                byte_buf.append(byte_decoder[c])
            else:
                if byte_buf:
                    parts.append(byte_buf.decode('utf-8', errors='ignore'))
                    byte_buf = bytearray()
                parts.append(c)
        if byte_buf:
            parts.append(byte_buf.decode('utf-8', errors='ignore'))
        return ''.join(parts)

    @staticmethod
    def _safe_json_loads(s):
        """Parse JSON, retrying with sanitized control characters if the first attempt fails.

        LLMs sometimes emit literal control characters (\\n, \\t, etc.) inside
        JSON string values, which json.loads rejects with 'Invalid control character'.
        """
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            import re
            _escape_map = {'\n': '\\n', '\r': '\\r', '\t': '\\t', '\b': '\\b', '\f': '\\f'}
            sanitized = re.sub(r'[\x00-\x1f]', lambda m: _escape_map.get(m.group(), ''), s)
            return json.loads(sanitized)

    @staticmethod
    def _parse_tool_calls(response_text):
        """Parse tool calls from response text. Handles three formats:
        1. Array format: [TOOL_CALLS][{"name": "...", "arguments": {...}}]
        2. Named format: [TOOL_CALLS]tool_name[ARGS]{"key": "value"}
        3. Simple format: [TOOL_CALLS]name\n{json} or name{json}

        For LRM responses, pass only the text after [TOOL_CALLS] marker
        (with any stray [/THINK] already stripped).

        Returns a list of {"name": ..., "arguments": ...} dicts, or empty list on failure.
        """
        import re

        # For standard generator, the full response_text contains [TOOL_CALLS]
        # For LRM, the caller strips the prefix before calling this.
        if "[TOOL_CALLS]" in response_text:
            tc_idx = response_text.find("[TOOL_CALLS]")
            after_tc = response_text[tc_idx + len("[TOOL_CALLS]"):].strip()
        else:
            after_tc = response_text.strip()

        tool_calls = []

        if after_tc.startswith('['):
            # Array format: [{"name": "...", "arguments": {...}}]
            bracket_count = 1
            end_idx = 1
            while end_idx < len(after_tc) and bracket_count > 0:
                if after_tc[end_idx] == '[':
                    bracket_count += 1
                elif after_tc[end_idx] == ']':
                    bracket_count -= 1
                end_idx += 1
            tool_json = after_tc[:end_idx]
            tool_calls = Generator._safe_json_loads(tool_json)

        elif '[ARGS]' in after_tc:
            # Named format: name[ARGS]{json}
            # For standard generator, multiple [TOOL_CALLS]...[ARGS] pairs may exist
            # For LRM, after_tc already has the marker stripped
            tool_pattern = re.compile(r'(?:\[TOOL_CALLS\])?([^\[]+)\[ARGS\]')
            matches = list(tool_pattern.finditer(after_tc))

            for match in matches:
                tool_name = match.group(1).strip()
                args_start = match.end()
                args_part = after_tc[args_start:].strip()

                brace_count = 0
                end_idx = 0
                for i, char in enumerate(args_part):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break

                args_json = args_part[:end_idx] if end_idx > 0 else "{}"
                tool_args = Generator._safe_json_loads(args_json) if args_json else {}
                tool_calls.append({"name": tool_name, "arguments": tool_args})

        elif '{' in after_tc:
            # Simple format: name\n{json} or name{json}
            brace_idx = after_tc.find('{')
            tool_name = after_tc[:brace_idx].strip()
            args_str = after_tc[brace_idx:]

            depth = 0
            end_idx = 0
            for i, c in enumerate(args_str):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

            args_json = args_str[:end_idx] if end_idx > 0 else "{}"
            tool_args = Generator._safe_json_loads(args_json)

            if tool_name:
                tool_calls.append({"name": tool_name, "arguments": tool_args})
            elif isinstance(tool_args, dict) and "name" in tool_args:
                tool_calls.append(tool_args)

        return tool_calls

    @staticmethod
    def _execute_tool(tool_name, tool_args, retriever, llm_model=None, llm_tokenizer=None):
        """Execute a single tool by name and return the result string.

        Returns the tool result (always a string).
        """
        if tool_name == "search_documents":
            tool_result = retriever.search_documents_tool(tool_args.get("query", ""))
        elif tool_name == "retrieve_document_by_name":
            tool_result = retriever.retrieve_document_by_name_tool(tool_args.get("document_name", ""))
        elif tool_name == "list_available_documents":
            tool_result = retriever.list_available_documents_tool(tool_args.get("filter_keyword", ""))
        elif tool_name == "search_wikipedia":
            tool_result = Generator.search_wikipedia(tool_args.get("query", ""))
        elif tool_name == "agentic_generator":
            from RAG_Framework.components.generators import AgenticGenerator
            tool_result = AgenticGenerator.agentic_answer_query(
                tool_args.get("query", ""), llm_model, llm_tokenizer, retriever)
        elif tool_name == "google_custom_search":
            tool_result = Generator.google_custom_search(tool_args.get("query", ""))
        elif tool_name == "duckduckgo_search":
            tool_result = Generator.duckduckgo_search(
                tool_args.get("query", ""), max(7, tool_args.get("max_results", 7)))
        elif tool_name == "fetch_url_content":
            tool_result = Generator.fetch_url_content(
                tool_args.get("url", ""), tool_args.get("max_chars", 8000))
        elif tool_name == "query_database":
            from RAG_Framework.components.database import get_sql_connector
            sql_connector = get_sql_connector()
            if sql_connector is None:
                tool_result = {"success": False, "error": "SQL databases are not configured"}
            else:
                tool_result = sql_connector.execute_query(
                    tool_args.get("db_name", ""), tool_args.get("sql_query", ""))
        elif tool_name == "list_databases":
            from RAG_Framework.components.database import get_sql_connector
            sql_connector = get_sql_connector()
            if sql_connector is None:
                tool_result = {"success": False, "error": "SQL databases are not configured"}
            else:
                tool_result = sql_connector.list_databases()
        elif tool_name == "get_database_schema":
            from RAG_Framework.components.database import get_sql_connector
            sql_connector = get_sql_connector()
            if sql_connector is None:
                tool_result = {"success": False, "error": "SQL databases are not configured"}
            else:
                tool_result = sql_connector.get_schema(tool_args.get("db_name", ""))
        else:
            tool_result = f"Error: Unknown tool: {tool_name}"

        if not isinstance(tool_result, str):
            tool_result = json.dumps(tool_result, ensure_ascii=False)

        return tool_result

    @staticmethod
    def _generate_with_streaming(model, tokenizer, prompt, max_tokens, sampler, logits_processors, prompt_cache, stream_callback, verbose=True):
        """
        Generate text with real-time streaming via callback.
        Captures verbose output from MLX generate() and emits tokens as they're generated.
        """
        if stream_callback is None:
            # TODO: i dont like this too much. another approach needs to be done.
            return Generator._fix_bpe_artifacts(generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=prompt_cache,
                verbose=verbose
            ))

        # Use stream_generate + _decode_tokens to bypass tokenizer.decode() entirely.
        # This avoids BPE artifact issues (e.g. Ministral Ġ/space mismatch) without
        # needing to capture/fix stdout.
        TOOL_MARKER = '[TOOL_CALLS]'
        TOOL_MARKER_LEN = len(TOOL_MARKER)

        all_tokens = []
        emitted = 0       # index into decoded text already sent to callback
        stop_streaming = False

        kwargs = {}
        if prompt_cache is not None:
            kwargs['prompt_cache'] = prompt_cache
        if sampler is not None:
            kwargs['sampler'] = sampler
        if logits_processors is not None:
            kwargs['logits_processors'] = logits_processors

        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            **kwargs
        ):
            all_tokens.append(response.token)
            if stop_streaming:
                continue

            decoded = Generator._decode_tokens(tokenizer, all_tokens)
            remaining = decoded[emitted:]
            mi = remaining.find(TOOL_MARKER)

            if mi >= 0:
                to_emit = remaining[:mi]
                if stream_callback and to_emit.strip():
                    stream_callback('text_chunk', {'text': to_emit})
                stop_streaming = True
                continue

            # Keep a tail buffered to catch markers that span chunk boundaries
            safe_len = len(remaining) - TOOL_MARKER_LEN
            if safe_len > 0:
                if stream_callback:
                    stream_callback('text_chunk', {'text': remaining[:safe_len]})
                emitted += safe_len

        # Flush remaining after generation ends
        full_text = Generator._decode_tokens(tokenizer, all_tokens)
        remaining = full_text[emitted:]
        if not stop_streaming and remaining:
            mi = remaining.find(TOOL_MARKER)
            to_emit = remaining[:mi] if mi >= 0 else remaining
            if stream_callback and to_emit.strip():
                stream_callback('text_chunk', {'text': to_emit})

        return full_text

    @staticmethod
    def summarize_passages(passages, llm_model, llm_tokenizer): # LLM summarizes retrieved info/context.
        context = "\n".join(passages)
        prompt = (
            "Summarize the following retrieved passages.\n"
            f"{context}\n"
            "Summary:"
        )
        summary = generate(llm_model, llm_tokenizer, prompt=prompt, max_tokens=700, verbose=False)
        return summary
    
    @staticmethod
    def search_wikipedia(query: str) -> str:
            """Fetch Wikipedia results with structured data for RAG"""
            try:
                import urllib.parse
                import urllib.request
                import json
                # Search for pages
                encoded_query = urllib.parse.quote(query)
                search_url = (
                    f"https://en.wikipedia.org/w/api.php?action=query&list=search"
                    f"&srsearch={encoded_query}&format=json&srlimit=1&srnamespace=0"
                )
                headers = {'User-Agent': 'YourApp/1.0 (contact@example.com)'}
                search_req = urllib.request.Request(search_url, headers=headers)
                with urllib.request.urlopen(search_req, timeout=10) as response:
                    search_data = json.loads(response.read().decode('utf-8'))

                if not search_data.get('query', {}).get('search'):
                    return {'error': 'No results found'}

                # Get page content
                page_ids = [str(item['pageid']) for item in search_data['query']['search']]
                content_url = (
                    f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts|info"
                    f"&pageids={'|'.join(page_ids)}&format=json&explaintext=true"
                    f"&inprop=url"
                )
                content_req = urllib.request.Request(content_url, headers=headers)
                with urllib.request.urlopen(content_req, timeout=10) as response:
                    content_data = json.loads(response.read().decode('utf-8'))

                # Structure the results for better LLM understanding
                articles = []
                for page_id in page_ids:
                    page_info = content_data['query']['pages'].get(page_id)
                    if page_info and 'extract' in page_info:
                        content = page_info['extract'].strip()
                        words = content.split()
                        if len(words) > 2000:
                            content = ' '.join(words[:2000]) + '... (content truncated to 2000 words)'
                            wordcount = 2000
                        else:
                            wordcount = len(words)
                        articles.append({
                            'title': page_info['title'],
                            'content': content,
                            'pageid': page_info['pageid'],
                            'url': page_info.get('fullurl', f"https://en.wikipedia.org/?curid={page_id}"),
                            'wordcount': wordcount
                        })

                return {
                    'query': query,
                    'articles': articles,
                    'total_results': len(articles)
                }
            except Exception as e:
                return {'error': str(e)}

    @staticmethod
    def google_custom_search(query: str) -> str:
        """Fetch Google Custom Search results"""
        try:
            import urllib.parse
            import urllib.request
            import json

            if GOOGLE_CX == "YOUR_GOOGLE_CX_HERE":
                return "Error: Google Custom Search Engine ID (CX) is not configured. Please set GOOGLE_CX in the code."

            encoded_query = urllib.parse.quote(query)
            search_url = (
                f"https://www.googleapis.com/customsearch/v1?"
                f"key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={encoded_query}"
            )
            
            with urllib.request.urlopen(search_url, timeout=10) as response:
                search_data = json.loads(response.read().decode('utf-8'))

            if 'items' not in search_data:
                return "No results found."

            results = []
            for item in search_data['items'][:8]: # Limit to top 8 results
                title = item.get('title', 'No title')
                snippet = item.get('snippet', 'No snippet')
                link = item.get('link', 'No link')
                results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")

            return "\n---\n".join(results)
        except Exception as e:
            return f"Error performing Google Search: {str(e)}"

    @staticmethod
    def duckduckgo_search(query: str, max_results: int = 7) -> dict:
        """Search DuckDuckGo (free, no API key required)."""
        try:
            # Try the newer ddgs package first, fallback to duckduckgo_search
            DDGS = None
            try:
                from ddgs import DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS
                except ImportError:
                    return {'error': 'DuckDuckGo search requires the ddgs package. Install with: pip install ddgs', 'query': query}

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'snippet': r.get('body', '')
                    })

            return {'query': query, 'results': results, 'total_results': len(results)}
        except Exception as e:
            return {'error': str(e), 'query': query}

    @staticmethod
    def fetch_url_content(url: str, max_chars: int = 8000) -> dict:
        """Fetch and extract main text content from a URL."""
        try:
            import urllib.request
            import urllib.parse
            import re

            # Validate URL
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ['http', 'https']:
                return {'error': 'Invalid URL scheme', 'url': url}

            headers = {'User-Agent': 'Mozilla/5.0 (compatible; RAGBot/1.0)'}
            request = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(request, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')

            # Strip scripts, styles, nav, header, footer
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL|re.I)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL|re.I)
            html = re.sub(r'<(nav|header|footer)[^>]*>.*?</\1>', '', html, flags=re.DOTALL|re.I)

            # Extract text
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) > max_chars:
                text = text[:max_chars] + '...'

            return {'url': url, 'content': text, 'char_count': len(text), 'success': True}
        except Exception as e:
            return {'error': str(e), 'url': url}

    @staticmethod
    def _get_advanced_reasoning_tool():
        """Returns the activate_advanced_reasoning tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "activate_advanced_reasoning",
                "description": (
                    "Activate advanced multi-step reasoning for complex questions that require "
                    "deep analysis, multi-source research, planning, and synthesis. Use this for "
                    "questions that are difficult, multi-faceted, require cross-referencing multiple "
                    "sources, involve step-by-step reasoning, or need comprehensive analysis. "
                    "Do NOT use for simple factual questions that can be answered directly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The complex question or task to reason about."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    @staticmethod
    def answer_query_with_llm(query, llm_model, llm_tokenizer, retriever, prompt_cache=None, stream_callback=None, verbose=True, conversation_manager=None):

        from RAG_Framework.agents.tools import get_tools_for_standard_generator

        tools = get_tools_for_standard_generator()
        if ADVANCED_REASONING:
            tools = tools + [Generator._get_advanced_reasoning_tool()]

        # Temperature and top sampling
        sampler = make_sampler(temp=0.3, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.0, repetition_context_size=128)
        current_response = None

        if conversation_manager is None:
            conversation_manager = ConversationManager()

        # Add current query to persistent conversation
        conversation_manager.add_user_message(query)

        # Get the conversation for use in the loop
        conversation = conversation_manager.get_conversation()

        # Save cache checkpoint before this query so we can restore it
        # between tool call iterations (each iteration's generated tokens
        # would corrupt the cache prefix for the next iteration)
        pre_query_checkpoint = CacheManager.get_checkpoint(prompt_cache)
        is_tool_call_continuation = False

        while True:
            # After a tool call iteration, restore cache to pre-query state.
            # The generated response tokens don't match what the chat template
            # produces for the tool call/result entries, so the cache prefix
            # would be invalid for the next iteration.
            if is_tool_call_continuation:
                CacheManager.restore_checkpoint(prompt_cache, pre_query_checkpoint)

            prompt = llm_tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tools=tools,
                tokenize=False
            )

            print("\nFORMATTED PROMPT:")
            print(prompt)

            # Tokenize full prompt
            full_tokens = llm_tokenizer.encode(prompt, add_special_tokens=False)

            # Get current cache size
            cache_offset = prompt_cache[0].offset if prompt_cache and len(prompt_cache) > 0 else 0

            # Only pass NEW tokens if cache has content
            if cache_offset > 0 and cache_offset < len(full_tokens):
                # Pass only the suffix tokens that aren't cached
                prompt_tokens = full_tokens[cache_offset:]
            else:
                prompt_tokens = full_tokens
                if cache_offset > 0:
                    # Cache doesn't match prompt — reset it before generating
                    print(f"\n[KV-Cache] Cache invalidated (offset {cache_offset} >= prompt {len(full_tokens)}), resetting cache")
                    for layer in prompt_cache:
                        if layer.offset > 0:
                            layer.trim(layer.offset)

            response = Generator._generate_with_streaming(
                model=llm_model,
                tokenizer=llm_tokenizer,
                prompt=prompt_tokens,
                max_tokens=MAX_RESPONSE_TOKENS,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=prompt_cache,
                stream_callback=stream_callback,
                verbose=verbose
            )

            response_text = response.strip()

            # tool call present?
            if "[TOOL_CALLS]" in response_text:
                print(f"\nModel requested to use tools. Processing tool calls...")
                try:
                    tool_calls = Generator._parse_tool_calls(response_text)

                    if not tool_calls:
                        print("No valid tool calls found")
                        break

                    print(f"Found {len(tool_calls)} tool call(s): {[call.get('name', 'unknown') for call in tool_calls]}")

                    # Special case: activate_advanced_reasoning delegates to AgenticGenerator
                    if ADVANCED_REASONING:
                        adv_call = next(
                            (call for call in tool_calls if call.get("name") == "activate_advanced_reasoning"),
                            None
                        )
                        if adv_call is not None:
                            reasoning_query = adv_call.get("arguments", {}).get("query", query)
                            call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
                            print(f"\n[ADVANCED REASONING] Delegating to AgenticGenerator for: {reasoning_query}")

                            if stream_callback:
                                stream_callback('tool_call_start', {
                                    'tool_id': call_id,
                                    'tool_name': 'activate_advanced_reasoning',
                                    'arguments': {'query': reasoning_query}
                                })

                            from RAG_Framework.components.generators.reasoning import AgenticGenerator
                            # Pass conversation_manager=None so AgenticGenerator manages its own
                            # synthesis conversation without re-adding the user message.
                            response_text = AgenticGenerator.agentic_answer_query(
                                query=reasoning_query,
                                llm_model=llm_model,
                                llm_tokenizer=llm_tokenizer,
                                retriever=retriever,
                                prompt_cache=prompt_cache,
                                conversation_manager=None,
                                stream_callback=stream_callback
                            )

                            # Keep standard conversation_manager in sync
                            conversation_manager.add_assistant_message(response_text)

                            return response_text, prompt

                    # Format tool calls for conversation
                    formatted_tool_calls = []
                    for call in tool_calls:
                        call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
                        if isinstance(call.get("arguments"), dict):
                            arguments_str = json.dumps(call["arguments"])
                        elif isinstance(call.get("arguments"), str):
                            try:
                                json.loads(call["arguments"])
                                arguments_str = call["arguments"]
                            except json.JSONDecodeError:
                                tool_name = call.get("name", "")
                                if tool_name in ["search_documents", "search_wikipedia"]:
                                    arguments_str = json.dumps({"query": call["arguments"]})
                                elif tool_name == "retrieve_document_by_name":
                                    arguments_str = json.dumps({"document_name": call["arguments"]})
                                else:
                                    arguments_str = json.dumps({"query": str(call.get("arguments", ""))})
                        else:
                            arguments_str = "{}"

                        formatted_tool_calls.append({
                            "id": call_id,
                            "function": {
                                "name": call["name"],
                                "arguments": arguments_str
                            },
                            "type": "function"
                        })

                    conversation.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": formatted_tool_calls
                    })

                    # Execute tools sequentially
                    for i, tool_call in enumerate(formatted_tool_calls):
                        tool_name = tool_call["function"]["name"]
                        tool_id = tool_call["id"]
                        args_str = tool_call["function"]["arguments"]
                        try:
                            tool_args = json.loads(args_str)
                        except json.JSONDecodeError as e:
                            tool_result = f"Tool call error: Invalid arguments format - {str(e)}"
                            tool_args = {}
                        else:
                            print(f"Executing tool {i+1}/{len(formatted_tool_calls)}: {tool_name} with args: {tool_args}")

                            if stream_callback:
                                stream_callback('tool_call_start', {
                                    'tool_id': tool_id,
                                    'tool_name': tool_name,
                                    'arguments': tool_args
                                })

                            tool_result = Generator._execute_tool(
                                tool_name, tool_args, retriever, llm_model, llm_tokenizer)

                        if stream_callback:
                            stream_callback('tool_call_result', {
                                'tool_id': tool_id,
                                'tool_name': tool_name,
                                'result': tool_result,
                                'status': 'success' if not tool_result.startswith('Error') else 'error'
                            })

                        conversation.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_result,
                            "tool_call_id": tool_call["id"]
                        })

                    print("All tool executions completed. Preparing final response...")
                    is_tool_call_continuation = True
                    continue
                except Exception as e:
                    print(f"Error processing tool calls: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break
            else:
                # No tool calls needed, return the response
                print("\nNo tool calls detected, returning response")

                # Add assistant response to conversation history for multi-turn caching
                conversation_manager.add_assistant_message(response_text)

                # Log cache stats for verification
                if prompt_cache is not None:
                    CacheManager.log_cache_stats(prompt_cache, f"After query (turn {conversation_manager.get_turn_count()})")

                return response_text, prompt

        # If we reach maximum iterations, return the current response
        print(f"Fatal Error")
        return current_response, prompt

    @staticmethod
    def eval(query, llm_model, llm_tokenizer, rag_response):
        system_prompt = f"""
            You are a RAG evaluator judge. Please evaluate the following response based on these criteria with a 0-10 scale:
            1. Context Relevance: How well retrieved documents align with the user's query and are able to address it.
                Example: Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The biggest earthquake in Lisbon was in 1755, which killed more than 30000 people. Score: 10; Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The white house is where the president lives. Score: 0; Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The earthquake. Lisbon. Earthquake in Lisbon. The earthquake in Lisbon. The earthquake in Lisbon. The biggest earthquake in Lisbon Score: 3;
            2. Groundedness: How accurately the response is based on the retrieved context.
            3. Answer Relevance: How well the response addresses the original query.
            4. Faithfulness: Is the output contradicting the retrieved facts?
            5. Contextual Recall: Did we retrieve ALL the info needed?
            6. Contextual Relevancy: What % of retrieved chunks actually matter?
            Be precise and objective on your scores! Be very critic.
        """
        user_prompt = f"""
            User Query: {query}
            Response and context: {rag_response}
        """
        eval_conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]
        sampler = make_sampler(temp=0.2, top_k=30, top_p=0.6)
        prompt = llm_tokenizer.apply_chat_template(
            eval_conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        eval_response = generate(
            model=llm_model,
            tokenizer=llm_tokenizer,
            prompt=prompt,
            sampler=sampler,
            verbose=False
        )
        print("Evaluation Results:\n", eval_response)

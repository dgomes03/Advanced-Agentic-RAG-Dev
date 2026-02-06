import json
import random
import string
import sys
import io
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from RAG_Framework.core.config import MAX_RESPONSE_TOKENS, ADVANCED_REASONING, GOOGLE_CX, GOOGLE_API_KEY
from RAG_Framework.core.cache_manager import CacheManager
from RAG_Framework.core.conversation_manager import ConversationManager


class Generator:
    @staticmethod
    def _generate_with_streaming(model, tokenizer, prompt, max_tokens, sampler, logits_processors, prompt_cache, stream_callback, verbose=True):
        """
        Generate text with real-time streaming via callback.
        Captures verbose output from MLX generate() and emits tokens as they're generated.
        """
        if stream_callback is None:
            # TODO: i dont like this too much. another approach needs to be done.
            return generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=prompt_cache,
                verbose=verbose
            )

        # Redirect stdout to capture verbose output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            # Generate with verbose=True (prints tokens to stdout)
            # We'll capture these and emit them via callback
            import threading
            import time

            # Track what we've already emitted
            last_pos = 0
            generation_complete = False

            def monitor_output():
                nonlocal last_pos, generation_complete
                while not generation_complete:
                    # Get current output
                    current_output = captured_output.getvalue()

                    # Check if there's new content
                    if len(current_output) > last_pos:
                        new_content = current_output[last_pos:]
                        last_pos = len(current_output)

                        # Filter out verbose stats lines (Prompt:, Generation:, Peak memory:, =====)
                        if not verbose and stream_callback:
                            # Skip stats lines
                            lines_to_emit = []
                            for line in new_content.split('\n'):
                                if not any(marker in line for marker in ['Prompt:', 'Generation:', 'Peak memory:', '====', 'tokens-per-sec']):
                                    lines_to_emit.append(line)
                            new_content = '\n'.join(lines_to_emit)

                        # Emit the new token(s)
                        if stream_callback and new_content.strip():
                            stream_callback('text_chunk', {'text': new_content})

                    # Small delay to avoid busy waiting
                    time.sleep(0.01)

            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_output, daemon=True)
            monitor_thread.start()

            # Generate (this will print to captured_output)
            # Always use verbose=True for streaming, but filter output if verbose=False
            result = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=prompt_cache,
                verbose=True
            )

            # Mark generation as complete
            generation_complete = True
            monitor_thread.join(timeout=0.1)

            # Emit any remaining content
            final_output = captured_output.getvalue()
            if len(final_output) > last_pos:
                remaining = final_output[last_pos:]

                # Filter out verbose stats lines if needed
                if not verbose and stream_callback:
                    lines_to_emit = []
                    for line in remaining.split('\n'):
                        if not any(marker in line for marker in ['Prompt:', 'Generation:', 'Peak memory:', '====', 'tokens-per-sec']):
                            lines_to_emit.append(line)
                    remaining = '\n'.join(lines_to_emit)

                if remaining.strip():
                    stream_callback('text_chunk', {'text': remaining})

            return result

        finally:
            # Restore stdout
            sys.stdout = old_stdout

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
                        if len(words) > 1000:
                            content = ' '.join(words[:1000]) + '... (content truncated to 1000 words)'
                            wordcount = 1000
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
            for item in search_data['items'][:5]: # Limit to top 5 results
                title = item.get('title', 'No title')
                snippet = item.get('snippet', 'No snippet')
                link = item.get('link', 'No link')
                results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")

            return "\n---\n".join(results)
        except Exception as e:
            return f"Error performing Google Search: {str(e)}"

    @staticmethod
    def duckduckgo_search(query: str, max_results: int = 5) -> dict:
        """Search DuckDuckGo (free, no API key required)."""
        try:
            # Try the newer ddgs package first, fallback to duckduckgo_search
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

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
    def fetch_url_content(url: str, max_chars: int = 5000) -> dict:
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
    def answer_query_with_llm(query, llm_model, llm_tokenizer, retriever, prompt_cache=None, stream_callback=None, verbose=True, conversation_manager=None):

        if ADVANCED_REASONING:
            from RAG_Framework.components.generators import AgenticGenerator
        from RAG_Framework.agents.tools import get_tools_for_standard_generator

        tools = get_tools_for_standard_generator()

        # Temperature and top sampling
        sampler = make_sampler(temp=0.7, top_k=50, top_p=0.9) # TODO: should i just leave the default values here? 
        logits_processors = make_logits_processors(repetition_penalty=1.1, repetition_context_size=128)
        current_response = None

        if conversation_manager is None:
            conversation_manager = ConversationManager()

        # Add current query to persistent conversation
        conversation_manager.add_user_message(query)

        # Get the conversation for use in the loop
        conversation = conversation_manager.get_conversation()

        while True:
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
                #print(f"[KV-Cache] Reusing {cache_offset} cached tokens, processing {len(prompt_tokens)} new tokens")
            else:
                prompt_tokens = full_tokens
                if cache_offset > 0:
                    print(f"\n[KV-Cache] Cache invalidated (offset {cache_offset} >= prompt {len(full_tokens)}), processing full prompt")

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
                    tool_calls = []
                    
                    # Check which format we're dealing with
                    if "[TOOL_CALLS][" in response_text:
                        # OLD FORMAT: [TOOL_CALLS][{"name": "...", "arguments": {...}}]
                        start_marker = "[TOOL_CALLS]["
                        start_idx = response_text.find(start_marker) + len(start_marker) - 1
                        bracket_count = 1
                        end_idx = start_idx + 1
                        while end_idx < len(response_text) and bracket_count > 0:
                            if response_text[end_idx] == '[':
                                bracket_count += 1
                            elif response_text[end_idx] == ']':
                                bracket_count -= 1
                            end_idx += 1
                        tool_json = response_text[start_idx:end_idx]
                        tool_calls = json.loads(tool_json)
                        
                    elif "[ARGS]" in response_text:
                        # NEW FORMAT: [TOOL_CALLS]tool_name[ARGS]{"key": "value"}
                        # Handle multiple tool calls by finding all [TOOL_CALLS]...[ARGS]... patterns
                        import re
                        tool_pattern = re.compile(r'\[TOOL_CALLS\]([^\[]+)\[ARGS\]')
                        matches = list(tool_pattern.finditer(response_text))

                        for match in matches:
                            tool_name = match.group(1).strip()
                            args_start = match.end()
                            args_part = response_text[args_start:].strip()

                            # Extract JSON object
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
                            tool_args = json.loads(args_json) if args_json else {}
                            tool_calls.append({"name": tool_name, "arguments": tool_args})
                    
                    else:
                        print("Unknown tool call format")
                        break

                    if not tool_calls:
                        print("No valid tool calls found")
                        break

                    print(f"Found {len(tool_calls)} tool call(s): {[call.get('name', 'unknown') for call in tool_calls]}")

                    # Add tool call to conversation
                    formatted_tool_calls = []
                    for call in tool_calls:
                        call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
                        # Handle different argument structures for different tools
                        if isinstance(call.get("arguments"), dict):
                            arguments_str = json.dumps(call["arguments"])
                        elif isinstance(call.get("arguments"), str):
                            try:
                                # Try to parse as JSON first
                                parsed_args = json.loads(call["arguments"])
                                arguments_str = call["arguments"]
                            except json.JSONDecodeError:
                                # If it's a string, create appropriate JSON based on tool
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

                    # Add to conversation
                    conversation.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": formatted_tool_calls
                    })

                    # Execute tools sequentially
                    tool_results = []
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

                            # Emit tool call start event
                            if stream_callback:
                                stream_callback('tool_call_start', {
                                    'tool_id': tool_id,
                                    'tool_name': tool_name,
                                    'arguments': tool_args
                                })

                            # Execute the appropriate tool with proper argument extraction
                            if tool_name == "search_documents":
                                query_str = tool_args.get("query", "")
                                tool_result = retriever.search_documents_tool(query_str)
                            elif tool_name == "retrieve_document_by_name":
                                doc_name = tool_args.get("document_name", "")
                                tool_result = retriever.retrieve_document_by_name_tool(doc_name)
                            elif tool_name == "list_available_documents":
                                filter_keyword = tool_args.get("filter_keyword", "")
                                tool_result = retriever.list_available_documents_tool(filter_keyword)
                            elif tool_name == "search_wikipedia":
                                query_str = tool_args.get("query", "")
                                tool_result = Generator.search_wikipedia(query_str)
                            elif tool_name == "agentic_generator":
                                query_str = tool_args.get("query", "")
                                tool_result = AgenticGenerator.agentic_answer_query(query_str, llm_model, llm_tokenizer, retriever)
                            elif tool_name == "google_custom_search":
                                query_str = tool_args.get("query", "")
                                tool_result = Generator.google_custom_search(query_str)
                            elif tool_name == "duckduckgo_search":
                                query_str = tool_args.get("query", "")
                                max_results = tool_args.get("max_results", 5)
                                tool_result = Generator.duckduckgo_search(query_str, max_results)
                            elif tool_name == "fetch_url_content":
                                url = tool_args.get("url", "")
                                max_chars = tool_args.get("max_chars", 5000)
                                tool_result = Generator.fetch_url_content(url, max_chars)
                            elif tool_name == "query_database":
                                from RAG_Framework.components.database import get_sql_connector
                                sql_connector = get_sql_connector()
                                if sql_connector is None:
                                    tool_result = {"success": False, "error": "SQL databases are not configured"}
                                else:
                                    db_name = tool_args.get("db_name", "")
                                    sql_query = tool_args.get("sql_query", "")
                                    tool_result = sql_connector.execute_query(db_name, sql_query)
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
                                    db_name = tool_args.get("db_name", "")
                                    tool_result = sql_connector.get_schema(db_name)
                            else:
                                tool_result = f"Error: Unknown tool: {tool_name}"

                        # Convert result to string if it's not already
                        if not isinstance(tool_result, str):
                            tool_result = json.dumps(tool_result, ensure_ascii=False)

                        # Emit tool call result event
                        if stream_callback:
                            stream_callback('tool_call_result', {
                                'tool_id': tool_id,
                                'tool_name': tool_name,
                                'result': tool_result,
                                'status': 'success' if not tool_result.startswith('Error') else 'error'
                            })

                        #print(tool_result) # mostrar resultados da tool use antes de resposta

                        # Add tool result to conversation
                        conversation.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_result,
                            "tool_call_id": tool_call["id"]
                        })
                        tool_results.append(tool_result)

                    print("All tool executions completed. Preparing final response...")
                    # Continue to next iteration to generate final response
                    continue
                except Exception as e:
                    print(f"Error processing tool calls: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break
            else:
                # No tool calls needed, return the response
                print("No tool calls detected, returning response")

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

import json
import random
import string
from mlx_lm import generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from RAG_Framework.core.config import MAX_RESPONSE_TOKENS, ADVANCED_REASONING
from RAG_Framework.core.cache_manager import CacheManager
from RAG_Framework.core.conversation_manager import ConversationManager
from RAG_Framework.core.parser import parse_tool_calls
from RAG_Framework.core.BPE_decode import BPEDecoder
from RAG_Framework.tools.wikipedia import search_wikipedia
from RAG_Framework.tools.web_search import duckduckgo_search, google_custom_search
from RAG_Framework.tools.fetch_url import fetch_url_content


class Generator:

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
            tool_result = search_wikipedia(tool_args.get("query", ""))
        elif tool_name == "agentic_generator":
            from RAG_Framework.components.generators import AgenticGenerator
            tool_result = AgenticGenerator.agentic_answer_query(
                tool_args.get("query", ""), llm_model, llm_tokenizer, retriever)
        elif tool_name == "google_custom_search":
            tool_result = google_custom_search(tool_args.get("query", ""))
        elif tool_name == "duckduckgo_search":
            tool_result = duckduckgo_search(tool_args.get("query", ""), tool_args.get("max_results"))
        elif tool_name == "fetch_url_content":
            tool_result = fetch_url_content(tool_args.get("url", ""), tool_args.get("max_chars"))
        elif tool_name == "query_database":
            from RAG_Framework.tools.SQL_database import get_sql_connector
            sql_connector = get_sql_connector()
            if sql_connector is None:
                tool_result = {"success": False, "error": "SQL databases are not configured"}
            else:
                tool_result = sql_connector.execute_query(
                    tool_args.get("db_name", ""), tool_args.get("sql_query", ""))
        elif tool_name == "list_databases":
            from RAG_Framework.tools.SQL_database import get_sql_connector
            sql_connector = get_sql_connector()
            if sql_connector is None:
                tool_result = {"success": False, "error": "SQL databases are not configured"}
            else:
                tool_result = sql_connector.list_databases()
        elif tool_name == "get_database_schema":
            from RAG_Framework.tools.SQL_database import get_sql_connector
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
            return BPEDecoder.fix_bpe_artifacts(generate(
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

            decoded = BPEDecoder.decode_tokens(tokenizer, all_tokens)
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
        full_text = BPEDecoder.decode_tokens(tokenizer, all_tokens)
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
                    tool_calls = parse_tool_calls(response_text)

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

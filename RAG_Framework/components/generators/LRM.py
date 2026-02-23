"""
LRM Generator: Language Reasoning Model generator with [THINK] marker support.

Uses stream_generate() from mlx_lm for generation. Decodes tokens by converting
IDs to raw BPE strings (convert_ids_to_tokens) and applying the GPT-2
bytes_to_unicode inverse mapping, bypassing the broken tokenizer.decode().
Handles [THINK]...[/THINK] reasoning blocks during streaming.
"""

import json
import random
import re
import string

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from RAG_Framework.core.config import MAX_RESPONSE_TOKENS
from RAG_Framework.core.cache_manager import CacheManager
from RAG_Framework.core.conversation_manager import ConversationManager
from RAG_Framework.core.parser import parse_tool_calls
from RAG_Framework.core.BPE_decode import BPEDecoder


# State machine states for streaming
_STATE_NORMAL = "NORMAL"
_STATE_THINKING = "THINKING"

# Markers
_THINK_START = "[THINK]"
_THINK_END = "[/THINK]"
_TOOL_MARKER = "[TOOL_CALLS]"
# Buffer size must cover the longest marker
_BUFFER_TAIL = max(len(_THINK_START), len(_THINK_END), len(_TOOL_MARKER))


class LRMGenerator:
    """Generator for reasoning models with [THINK] block support."""

    @staticmethod
    def _extract_thinking(text):
        """Strip [THINK]...[/THINK] from raw response.

        If the model placed everything inside [THINK] (nothing outside),
        try to split reasoning from the actual response by looking for
        common boundary markers like "Response:", "Answer:", etc.

        Returns:
            (thinking_content, clean_text) tuple.
        """
        pattern = re.compile(r'\[THINK\](.*?)\[/THINK\]', re.DOTALL)
        thinking_parts = pattern.findall(text)
        thinking = "\n".join(thinking_parts).strip() if thinking_parts else ""
        clean = pattern.sub("", text).strip()

        # Handle unclosed [THINK] block (model didn't emit [/THINK])
        if not thinking and '[THINK]' in clean:
            idx = clean.find('[THINK]')
            rest = clean[idx + len('[THINK]'):]
            # Don't swallow [TOOL_CALLS] content into thinking
            tc_idx = rest.find('[TOOL_CALLS]')
            if tc_idx >= 0:
                thinking = rest[:tc_idx].strip()
                clean = clean[:idx].strip() + rest[tc_idx:]
            else:
                thinking = rest.strip()
                clean = clean[:idx].strip()

        return thinking, clean

    @staticmethod
    def _stream_with_thinking(model, tokenizer, prompt_tokens, max_tokens,
                              stream_callback, prompt_cache=None,
                              sampler=None, logits_processors=None):
        """
        Stream generate with [THINK] marker detection.

        Uses a simple index-based approach: accumulate the full decoded text
        and track how far we've emitted.  A ``while`` loop inside each token
        step handles multiple state transitions (e.g. [THINK]...[/THINK] in
        one chunk) without waiting for the next token.

        Emits via *stream_callback*:
          thinking_start → thinking_chunk* → thinking_complete
          text_chunk (regular response text)
        """
        state = _STATE_NORMAL
        all_tokens = []
        emitted = 0           # index into decoded text already sent to UI
        text_emitted = False
        stop_streaming = False

        kwargs = {}
        if prompt_cache is not None:
            kwargs["prompt_cache"] = prompt_cache
        if sampler is not None:
            kwargs["sampler"] = sampler
        if logits_processors is not None:
            kwargs["logits_processors"] = logits_processors

        def _emit(event, data=None):
            if stream_callback:
                stream_callback(event, data or {})

        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_tokens,
            max_tokens=max_tokens,
            **kwargs
        ):
            all_tokens.append(response.token)
            if stop_streaming:
                continue

            decoded = BPEDecoder.decode_tokens(tokenizer, all_tokens)

            # Process all transitions possible with current decoded text
            while emitted < len(decoded):
                remaining = decoded[emitted:]

                if state == _STATE_NORMAL:
                    ti = remaining.find(_THINK_START)
                    mi = remaining.find(_TOOL_MARKER)

                    # Pick whichever marker comes first
                    if ti >= 0 and (mi < 0 or ti <= mi):
                        before = remaining[:ti]
                        if before.strip():
                            _emit('text_chunk', {'text': before})
                            text_emitted = True
                        emitted += ti + len(_THINK_START)
                        state = _STATE_THINKING
                        _emit('thinking_start')
                        continue  # re-check for [/THINK] immediately

                    if mi >= 0 and (ti < 0 or mi < ti):
                        before = remaining[:mi]
                        if before.strip():
                            _emit('text_chunk', {'text': before})
                            text_emitted = True
                        stop_streaming = True
                        break

                    # No marker yet — emit text safely outside the marker zone
                    safe_len = len(remaining) - _BUFFER_TAIL
                    if safe_len > 0:
                        _emit('text_chunk', {'text': remaining[:safe_len]})
                        text_emitted = True
                        emitted += safe_len
                    break  # wait for more tokens

                elif state == _STATE_THINKING:
                    ei = remaining.find(_THINK_END)
                    if ei >= 0:
                        if remaining[:ei]:
                            _emit('thinking_chunk', {'text': remaining[:ei]})
                        _emit('thinking_complete')
                        emitted += ei + len(_THINK_END)
                        state = _STATE_NORMAL
                        continue  # re-check for text / new [THINK]

                    # No end marker yet — stream safe portion
                    safe_len = len(remaining) - _BUFFER_TAIL
                    if safe_len > 0:
                        _emit('thinking_chunk', {'text': remaining[:safe_len]})
                        emitted += safe_len
                    break  # wait for more tokens

        # --- Flush after generation ends ---
        full_text = BPEDecoder.decode_tokens(tokenizer, all_tokens)
        remaining = full_text[emitted:]

        if remaining and state == _STATE_THINKING:
            ei = remaining.find(_THINK_END)
            if ei >= 0:
                if remaining[:ei]:
                    _emit('thinking_chunk', {'text': remaining[:ei]})
                _emit('thinking_complete')
                rest = remaining[ei + len(_THINK_END):]
                if rest.strip():
                    _emit('text_chunk', {'text': rest})
                    text_emitted = True
            else:
                _emit('thinking_chunk', {'text': remaining})
                _emit('thinking_complete')
        elif remaining:
            mi = remaining.find(_TOOL_MARKER)
            to_emit = remaining[:mi] if mi >= 0 else remaining
            if to_emit.strip():
                _emit('text_chunk', {'text': to_emit})
                text_emitted = True

        return full_text, text_emitted

    @staticmethod
    def answer_query_with_llm(query, llm_model, llm_tokenizer, retriever,
                              prompt_cache=None, stream_callback=None,
                              verbose=True, conversation_manager=None):
        from RAG_Framework.agents.tools import get_tools_for_standard_generator
        from RAG_Framework.components.generators.standard import Generator  # for _execute_tool

        tools = get_tools_for_standard_generator()

        sampler = make_sampler(temp=0.6, top_k=40, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.1, repetition_context_size=128)
        current_response = None

        if conversation_manager is None:
            conversation_manager = ConversationManager(reasoning_model=True)

        conversation_manager.add_user_message(query)
        conversation = conversation_manager.get_conversation()

        pre_query_checkpoint = CacheManager.get_checkpoint(prompt_cache)
        is_tool_call_continuation = False

        while True:
            if is_tool_call_continuation:
                CacheManager.restore_checkpoint(prompt_cache, pre_query_checkpoint)

            prompt = llm_tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tools=tools,
                tokenize=False
            )

            #print("\nFORMATTED PROMPT:")
            #print(prompt)

            full_tokens = llm_tokenizer.encode(prompt, add_special_tokens=False)

            cache_offset = prompt_cache[0].offset if prompt_cache and len(prompt_cache) > 0 else 0

            if cache_offset > 0 and cache_offset < len(full_tokens):
                prompt_tokens = full_tokens[cache_offset:]
            else:
                prompt_tokens = full_tokens
                if cache_offset > 0:
                    print(f"\n[KV-Cache] Cache invalidated (offset {cache_offset} >= prompt {len(full_tokens)}), resetting cache")
                    for layer in prompt_cache:
                        if layer.offset > 0:
                            layer.trim(layer.offset)

            # Use stream_generate with manual token decoding
            text_emitted = False
            if stream_callback is not None:
                response_text, text_emitted = LRMGenerator._stream_with_thinking(
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    prompt_tokens=prompt_tokens,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    stream_callback=stream_callback,
                    prompt_cache=prompt_cache,
                    sampler=sampler,
                    logits_processors=logits_processors
                )
            else:
                # Offline mode: use a console callback for real-time verbose output
                def _console_callback(event, data):
                    if event == 'thinking_start':
                        print("\n--- Thinking ---", flush=True)
                    elif event == 'thinking_chunk':
                        print(data.get('text', ''), end='', flush=True)
                    elif event == 'thinking_complete':
                        print("\n--- End Thinking ---", flush=True)
                    elif event == 'text_chunk':
                        print(data.get('text', ''), end='', flush=True)

                response_text, text_emitted = LRMGenerator._stream_with_thinking(
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    prompt_tokens=prompt_tokens,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    stream_callback=_console_callback if verbose else None,
                    prompt_cache=prompt_cache,
                    sampler=sampler,
                    logits_processors=logits_processors
                )
                if not verbose:
                    # If not verbose, _stream_with_thinking ran without callbacks
                    # (text_emitted will be False, which is fine for offline)
                    pass
                else:
                    print(flush=True)  # Final newline after streaming

            response_text = response_text.replace('</s>', '').strip()

            # Tool call handling
            if "[TOOL_CALLS]" in response_text:
                print(f"\nModel requested to use tools. Processing tool calls...")
                try:
                    # Extract text after [TOOL_CALLS], strip stray [/THINK]
                    tc_idx = response_text.find('[TOOL_CALLS]')
                    after_tc = response_text[tc_idx + len('[TOOL_CALLS]'):].strip()
                    after_tc = after_tc.replace('[/THINK]', '').strip()
                    print(f"[LRM Debug] Tool call text after marker: {after_tc[:300]!r}")

                    tool_calls = parse_tool_calls(after_tc)

                    if not tool_calls:
                        print("No valid tool calls found")
                        break

                    print(f"Found {len(tool_calls)} tool call(s): {[call.get('name', 'unknown') for call in tool_calls]}")

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
                # No tool calls — extract thinking and store in conversation
                print("\nNo tool calls detected, returning response")

                thinking, clean_text = LRMGenerator._extract_thinking(response_text)

                # If streaming didn't emit any text_chunk to the message area,
                # emit the extracted response now (not the full thinking content).
                if not text_emitted and stream_callback and clean_text:
                    print(f"[LRM] Response not streamed, emitting clean_text ({len(clean_text)} chars)")
                    stream_callback('text_chunk', {'text': clean_text})

                response_for_history = clean_text if clean_text else thinking
                if thinking:
                    conversation_manager.add_assistant_message_with_thinking(thinking, response_for_history)
                else:
                    conversation_manager.add_assistant_message(response_for_history)

                if prompt_cache is not None:
                    CacheManager.log_cache_stats(prompt_cache, f"After query (turn {conversation_manager.get_turn_count()})")

                return response_for_history, prompt

        print("Fatal Error")
        return current_response, prompt

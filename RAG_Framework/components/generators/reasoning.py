import json
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from RAG_Framework.core.config import MAX_RESPONSE_TOKENS, MIN_CONFIDENCE_THRESHOLD, MAX_REASONING_STEPS
from RAG_Framework.core.conversation_manager import ConversationManager


_SKIP_SPECIAL = {'<s>', '</s>', '<unk>', '<pad>'}


class AgenticGenerator:
    """Main agentic reasoning coordinator.

    Orchestrates multi-step reasoning with planning, LLM-driven tool selection,
    evaluation with information-gain tracking, dynamic replanning, and
    streaming synthesis.
    """

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
        if AgenticGenerator._byte_decoder is None:
            AgenticGenerator._byte_decoder = AgenticGenerator._build_byte_decoder()
        if AgenticGenerator._inv_vocab is None:
            vocab = tokenizer.get_vocab()
            AgenticGenerator._inv_vocab = {v: k for k, v in vocab.items()}
            special = set(getattr(tokenizer, 'all_special_ids', []))
            if hasattr(tokenizer, 'added_tokens_encoder'):
                special.update(tokenizer.added_tokens_encoder.values())
            AgenticGenerator._special_ids = special

    @staticmethod
    def _decode_tokens(tokenizer, tokens):
        """Decode token IDs to text, bypassing tokenizer.decode() entirely.

        Handles tokenizers where the BPE vocabulary uses Ġ (U+0120) as the
        space marker but the decoder expects ▁ (U+2581), which causes
        tokenizer.decode() to strip all spaces from output.
        """
        if not tokens:
            return ""
        AgenticGenerator._ensure_vocab(tokenizer)

        inv_vocab = AgenticGenerator._inv_vocab
        special_ids = AgenticGenerator._special_ids
        byte_decoder = AgenticGenerator._byte_decoder

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
        if AgenticGenerator._byte_decoder is None:
            AgenticGenerator._byte_decoder = AgenticGenerator._build_byte_decoder()
        byte_decoder = AgenticGenerator._byte_decoder
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
    def agentic_answer_query(
        query: str,
        llm_model,
        llm_tokenizer,
        retriever,
        prompt_cache=None,
        conversation_manager=None,
        stream_callback=None
    ) -> str:
        """
        Main agentic loop with multi-step reasoning, evaluation, and replanning.

        Uses Generator._execute_tool() for all tool execution (no duplication),
        streams events to the frontend, and synthesises a final answer using
        the shared conversation manager and KV-cache.
        """
        from RAG_Framework.agents import AgenticPlanner, AgenticEvaluator
        from RAG_Framework.components.generators import Generator
        from RAG_Framework.agents.tools import get_tools_for_standard_generator

        print(f"\n{'='*60}")
        print(f"AGENTIC REASONING: {query}")
        print(f"{'='*60}\n")

        # ── Step 1: Create initial plan ──────────────────────────────
        print("Creating reasoning plan...")
        plan = AgenticPlanner.create_initial_plan(query, llm_model, llm_tokenizer)

        if stream_callback:
            stream_callback('reasoning_step', {
                'step': 0,
                'total_steps': MAX_REASONING_STEPS,
                'goal': 'Planning complete',
                'strategy': 'planning',
                'goals': [{'description': g.description, 'priority': g.priority, 'strategy': g.strategy}
                          for g in plan.goals]
            })

        # Track all retrieved contexts for information-gain comparison
        all_previous_contexts = []
        reasoning_step = 0

        # ── Step 2: Iterative reasoning loop ─────────────────────────
        while reasoning_step < MAX_REASONING_STEPS:
            reasoning_step += 1

            # Get next pending goal
            current_goal = plan.get_next_goal()
            if current_goal is None:
                print("All goals completed!")
                break

            print(f"\n{'─'*60}")
            print(f"REASONING STEP {reasoning_step}/{MAX_REASONING_STEPS}")
            print(f"Goal: {current_goal.description}")
            print(f"Priority: {current_goal.priority} | Strategy: {current_goal.strategy}")
            print(f"{'─'*60}")

            current_goal.status = "in_progress"

            if stream_callback:
                stream_callback('reasoning_step', {
                    'step': reasoning_step,
                    'total_steps': MAX_REASONING_STEPS,
                    'goal': current_goal.description,
                    'strategy': current_goal.strategy
                })

            # ── Step 2b: Retrieve via LLM tool-calling ───────────────
            try:
                retrieved_context = AgenticGenerator._retrieve_for_goal(
                    goal=current_goal,
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                    retriever=retriever,
                    stream_callback=stream_callback
                )

                if retrieved_context:
                    current_goal.retrieved_info.append(retrieved_context)
                    print(f"\n[RETRIEVE] Retrieved {len(retrieved_context)} characters of context")
                    print(f"[RETRIEVE] Preview: {retrieved_context[:300]}...")
                else:
                    print("[RETRIEVE] No context retrieved (empty response)")
                    retrieved_context = ""
            except Exception as e:
                print(f"[RETRIEVE] Search failed: {e}")
                import traceback
                traceback.print_exc()
                current_goal.status = "failed"
                continue

            if stream_callback:
                stream_callback('reasoning_retrieval', {
                    'goal': current_goal.description,
                    'chars_retrieved': len(retrieved_context),
                    'preview': retrieved_context[:200] if retrieved_context else ""
                })

            # ── Step 2c: Evaluate with information gain ──────────────
            print(f"\n[EVAL] Evaluating goal completion...")
            evaluation = AgenticEvaluator.evaluate_goal_completion_with_gain(
                current_goal,
                retrieved_context,
                all_previous_contexts,
                llm_model,
                llm_tokenizer
            )

            current_goal.confidence = evaluation.get("confidence", 0.5)
            info_gain = evaluation.get("information_gain", 0.5)
            sparse = evaluation.get("sparse_results", False)
            contradictory = evaluation.get("contradictory_info", False)

            print(f"[EVAL] Complete: {evaluation.get('is_complete', False)}")
            print(f"[EVAL] Confidence: {current_goal.confidence:.2f}")
            print(f"[EVAL] Information gain: {info_gain:.2f}")
            print(f"[EVAL] Sparse: {sparse} | Contradictory: {contradictory}")
            print(f"[EVAL] Reasoning: {evaluation.get('reasoning', 'N/A')}")

            if stream_callback:
                stream_callback('reasoning_evaluation', {
                    'confidence': current_goal.confidence,
                    'information_gain': info_gain,
                    'is_complete': evaluation.get('is_complete', False),
                    'sparse_results': sparse,
                    'contradictory_info': contradictory,
                    'reasoning': evaluation.get('reasoning', '')
                })

            # Add to previous contexts for future info-gain comparison
            if retrieved_context:
                all_previous_contexts.append(retrieved_context)

            # Update goal status
            if evaluation.get("is_complete", False) and current_goal.confidence >= MIN_CONFIDENCE_THRESHOLD:
                current_goal.status = "completed"
                print(f"[EVAL] Goal completed successfully!")
            else:
                # Mark as completed but with low confidence — move on
                current_goal.status = "completed"
                print(f"[EVAL] Goal marked completed (confidence: {current_goal.confidence:.2f})")

            # ── Step 2d: Autonomous stop checks ──────────────────────
            # Check 1: All goals have high confidence
            all_confident = all(
                g.confidence >= MIN_CONFIDENCE_THRESHOLD
                for g in plan.goals if g.status == "completed"
            ) and plan.is_complete()

            if all_confident and plan.is_complete():
                print(f"\nAUTONOMOUS STOP: All goals completed with sufficient confidence")
                break

            # Check 2: Information gain too low (no new info)
            if info_gain < 0.1 and reasoning_step > 1:
                print(f"\nAUTONOMOUS STOP: Information gain too low ({info_gain:.2f})")
                break

            # ── Step 2e: Replan if needed ────────────────────────────
            completion_rate = plan.get_completion_rate()
            if completion_rate >= 0.5 and not plan.is_complete():
                # Check if confidence is below threshold — trigger replan
                avg_confidence = sum(
                    g.confidence for g in plan.goals if g.status == "completed"
                ) / max(1, sum(1 for g in plan.goals if g.status == "completed"))

                if avg_confidence < MIN_CONFIDENCE_THRESHOLD or sparse or contradictory:
                    print(f"\n[REPLAN] Replanning (avg confidence: {avg_confidence:.2f}, sparse: {sparse}, contradictory: {contradictory})...")
                    plan = AgenticPlanner.replan(plan, evaluation, llm_model, llm_tokenizer)

                    if stream_callback:
                        stream_callback('reasoning_replan', {
                            'action': 'replan',
                            'new_goals': [g.description for g in plan.goals if g.status == "pending"],
                            'reasoning': evaluation.get('reasoning', '')
                        })

        # ── Step 3: Synthesize final answer ──────────────────────────
        print(f"\n{'='*60}")
        print(f"SYNTHESIZING FINAL ANSWER")
        print(f"{'='*60}")

        response = AgenticGenerator._synthesize_answer(
            query=query,
            plan=plan,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            conversation_manager=conversation_manager,
            prompt_cache=prompt_cache,
            stream_callback=stream_callback
        )

        return response

    @staticmethod
    def _retrieve_for_goal(goal, llm_model, llm_tokenizer, retriever, stream_callback=None):
        """Use LLM tool-calling to retrieve information for a single goal.

        Builds a mini conversation, lets the LLM pick the right tool(s),
        executes via Generator._execute_tool(), and returns the combined
        retrieval results as a string.
        """
        from RAG_Framework.components.generators import Generator
        from RAG_Framework.agents.tools import get_tools_for_standard_generator

        tools = get_tools_for_standard_generator()

        system_prompt = (
            "You are a research assistant. For the given research goal, call the most appropriate tool "
            "to find the information needed. Pick ONE tool and call it with appropriate arguments.\n\n"
            "TOOL PRIORITY: search_documents > query_database > duckduckgo_search > search_wikipedia > google_custom_search\n\n"
            "Call the tool now — do not write explanatory text."
        )

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research goal: {goal.description}"}
        ]

        prompt = llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tools=tools,
            tokenize=False
        )

        print(f"\n[TOOL_SELECT] Formatted prompt:\n{prompt}")

        # Short generation — we only need a tool call (~100 tokens)
        response = generate(
            model=llm_model,
            tokenizer=llm_tokenizer,
            prompt=prompt,
            max_tokens=200,
            verbose=True
        )

        response_text = AgenticGenerator._fix_bpe_artifacts(response.strip())
        print(f"\n[TOOL_SELECT] Raw LLM response:\n{repr(response_text)}")

        # If no tool call, fall back to duckduckgo_search (not search_documents,
        # which would trigger embedding model loading)
        if "[TOOL_CALLS]" not in response_text:
            print(f"[TOOL_SELECT] No tool call from LLM, falling back to duckduckgo_search")
            result = Generator._execute_tool(
                "duckduckgo_search",
                {"query": goal.description},
                retriever, llm_model, llm_tokenizer
            )
            return result

        # Parse and execute tool calls
        try:
            tool_calls = Generator._parse_tool_calls(response_text)
            print(f"[TOOL_SELECT] Parsed tool calls: {tool_calls}")

            if not tool_calls:
                print(f"[TOOL_SELECT] No valid tool calls parsed, falling back to duckduckgo_search")
                result = Generator._execute_tool(
                    "duckduckgo_search",
                    {"query": goal.description},
                    retriever, llm_model, llm_tokenizer
                )
                return result

            # Execute each tool call and collect results
            all_results = []
            for call in tool_calls:
                tool_name = call.get("name", "")
                tool_args = call.get("arguments", {})

                # Ensure arguments is a dict
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {"query": tool_args}

                # Strategy override: if goal.strategy == "bm25_only" and tool is search_documents,
                # override to use keyword-only search
                if tool_name == "search_documents" and goal.strategy == "bm25_only":
                    print(f"[TOOL_EXEC] Executing: {tool_name} (BM25-only override) with args: {tool_args}")
                    query_str = tool_args.get("query", goal.description)
                    result = retriever.search_documents_tool(query_str, weight_dense=0.0, weight_sparse=1.0)
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)
                elif tool_name == "search_documents" and goal.strategy == "dense_only":
                    print(f"[TOOL_EXEC] Executing: {tool_name} (dense-only override) with args: {tool_args}")
                    query_str = tool_args.get("query", goal.description)
                    result = retriever.search_documents_tool(query_str, weight_dense=1.0, weight_sparse=0.0)
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)
                else:
                    print(f"[TOOL_EXEC] Executing: {tool_name} with args: {tool_args}")
                    result = Generator._execute_tool(
                        tool_name, tool_args, retriever, llm_model, llm_tokenizer
                    )

                print(f"[TOOL_EXEC] Result preview ({len(result)} chars): {result[:300]}...")
                all_results.append(result)

            return "\n---\n".join(all_results)

        except Exception as e:
            print(f"[TOOL_SELECT] Error parsing/executing tool calls: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to duckduckgo (avoids loading embedding models)
            print(f"[TOOL_SELECT] Falling back to duckduckgo_search")
            result = Generator._execute_tool(
                "duckduckgo_search",
                {"query": goal.description},
                retriever, llm_model, llm_tokenizer
            )
            return result

    @staticmethod
    def _synthesize_answer(query, plan, llm_model, llm_tokenizer,
                           conversation_manager, prompt_cache, stream_callback):
        """Build a synthesis prompt from all accumulated context and generate
        the final answer with streaming.

        Does NOT call answer_query_with_llm() (which would retrieve again).
        Instead, injects all gathered context directly and generates.
        """
        from RAG_Framework.components.generators import Generator
        from RAG_Framework.core.cache_manager import CacheManager

        # Build context from all goals' retrieved_info
        synthesis_context = ""
        for goal in plan.goals:
            for info in goal.retrieved_info:
                if isinstance(info, str):
                    synthesis_context += info + "\n---\n"
                elif isinstance(info, tuple):
                    synthesis_context += str(info[0]) + "\n---\n"
                else:
                    synthesis_context += str(info) + "\n---\n"

        # Truncate context to avoid exceeding model limits
        if len(synthesis_context) > 8000:
            synthesis_context = synthesis_context[:8000] + "\n...(truncated)"

        print(f"\n[SYNTHESIS] Total context: {len(synthesis_context)} chars from {sum(len(g.retrieved_info) for g in plan.goals)} retrievals")
        print(f"[SYNTHESIS] Context preview:\n{synthesis_context[:500]}...")

        # Build user message with context + query
        user_msg = (
            f"Based on the following retrieved information, answer the query.\n\n"
            f"RETRIEVED INFORMATION:\n{synthesis_context}\n\n"
            f"QUERY: {query}\n\n"
            f"Answer using ONLY the retrieved information above. Cite sources."
        )

        # Use existing conversation manager or create a new one
        if conversation_manager is None:
            conversation_manager = ConversationManager()

        conversation_manager.add_user_message(user_msg)

        # Generate with streaming using the shared infrastructure
        prompt = llm_tokenizer.apply_chat_template(
            conversation_manager.get_conversation(),
            add_generation_prompt=True,
            tokenize=False
        )

        print(f"\n[SYNTHESIS] Formatted prompt:\n{prompt[:500]}...(truncated)")

        sampler = make_sampler(temp=0.3, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.0)

        # Handle KV-cache: tokenize and pass only new tokens if cache is warm
        full_tokens = llm_tokenizer.encode(prompt, add_special_tokens=False)
        cache_offset = prompt_cache[0].offset if prompt_cache and len(prompt_cache) > 0 else 0

        print(f"[SYNTHESIS] Full prompt tokens: {len(full_tokens)}, Cache offset: {cache_offset}")

        if cache_offset > 0 and cache_offset < len(full_tokens):
            prompt_tokens = full_tokens[cache_offset:]
            print(f"[SYNTHESIS] Passing {len(prompt_tokens)} new tokens (cache hit)")
        else:
            prompt_tokens = full_tokens
            if cache_offset > 0:
                print(f"[SYNTHESIS] Cache invalidated (offset {cache_offset} >= prompt {len(full_tokens)}), resetting")
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
            stream_callback=stream_callback
        )

        response_text = response.strip()
        print(f"\n[SYNTHESIS] Response length: {len(response_text)} chars")
        conversation_manager.add_assistant_message(response_text)

        # Log cache stats
        if prompt_cache is not None:
            CacheManager.log_cache_stats(prompt_cache, f"After agentic synthesis (turn {conversation_manager.get_turn_count()})")

        return response_text

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from mlx_lm import generate
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from RAG_Framework.core.config import MAX_RESPONSE_TOKENS, MIN_CONFIDENCE_THRESHOLD, MAX_REASONING_STEPS
from RAG_Framework.core.conversation_manager import ConversationManager


_SKIP_SPECIAL = {'<s>', '</s>', '<unk>', '<pad>'}


class AgenticGenerator:
    """Main agentic reasoning coordinator.

    Orchestrates multi-step reasoning with planning, LLM-driven tool selection,
    evaluation with information-gain tracking, dynamic replanning, and
    streaming synthesis.

    The retrieval phase uses BatchGenerator so that all pending goals are
    processed simultaneously by the LLM (tool selection + evaluation), while
    actual tool execution (web search, document search, etc.) runs concurrently
    via ThreadPoolExecutor.
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

    # ─────────────────────────────────────────────────────────────
    # Batch generation core
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _run_batch_generate(llm_model, llm_tokenizer, prompts, max_tokens, label="BATCH", on_complete=None):
        """Run BatchGenerator for a list of prompts simultaneously.

        Returns a list of decoded strings in the same order as *prompts*.
        Falls back to sequential generate_fixed() for a single prompt.

        on_complete(idx, result): optional callback fired immediately when each
        individual sequence finishes — before the full batch is done.  Use this
        to dispatch downstream work (e.g. tool execution) without waiting for
        the slowest sequence to complete.
        """
        from RAG_Framework.agents.planner import generate_fixed

        if not prompts:
            return []

        if len(prompts) == 1:
            result = generate_fixed(
                llm_model, llm_tokenizer,
                prompt=prompts[0], max_tokens=max_tokens, verbose=True
            )
            if on_complete:
                on_complete(0, result)
            return [result]

        print(f"\n[{label}] BatchGenerator: {len(prompts)} prompts, max_tokens={max_tokens}")

        # BatchGenerator.insert() requires token ID lists, not strings.
        token_prompts = []
        for p in prompts:
            if isinstance(p, str):
                token_prompts.append(llm_tokenizer.encode(p, add_special_tokens=False))
            else:
                token_prompts.append(p)

        batch_gen = BatchGenerator(llm_model, max_tokens=max_tokens)
        uids = batch_gen.insert(token_prompts)
        uid_to_idx = {uid: i for i, uid in enumerate(uids)}   # O(1) uid→index
        uid_to_token_ids = {uid: [] for uid in uids}
        active = set(uids)
        results = [None] * len(prompts)

        while active:
            for resp in batch_gen.next():
                if resp.uid in active:
                    uid_to_token_ids[resp.uid].append(resp.token)
                    if resp.finish_reason:
                        active.remove(resp.uid)
                        idx = uid_to_idx[resp.uid]
                        # Use _decode_tokens to avoid BPE artifact issues
                        # that llm_tokenizer.decode() produces for some models
                        text = AgenticGenerator._decode_tokens(llm_tokenizer, uid_to_token_ids[resp.uid])
                        results[idx] = text
                        print(f"[{label}] Sequence {idx} done ({len(text)} chars)")
                        if on_complete:
                            on_complete(idx, text)

        batch_gen.close()
        print(f"[{label}] BatchGenerator complete")
        return results

    # ─────────────────────────────────────────────────────────────
    # Iterative batch retrieval (multi-round tool use per goal)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _batch_retrieve_for_goals(goals, llm_model, llm_tokenizer, retriever,
                                   stream_callback=None, goal_indices=None):
        """Retrieve information for every goal using multi-round iterative tool use.

        Each round consists of two phases:
          Phase A — tool selection (BatchGenerator): all active goals' prompts run
              simultaneously on the GPU.  As each selection finishes its tool
              execution is dispatched immediately to the thread pool.
          Phase B — tool execution (ThreadPoolExecutor): I/O-bound tools run
              concurrently, overlapping with Phase A.

        After each round, results are injected back into each goal's conversation.
        Goals that produce no further tool calls (or hit MAX_TOOL_ITERS) are
        finalised.  This allows each goal to drill down: e.g. search → see URLs
        in results → fetch specific URLs in the next round.

        reasoning_retrieval events are emitted once per goal after all its rounds
        complete, with the full accumulated source list.

        Returns a list of (retrieved_context: str, tool_name: str) in the same
        order as *goals*.
        """
        from RAG_Framework.components.generators import Generator
        from RAG_Framework.agents.tools import get_tools_for_standard_generator
        from datetime import datetime

        if goal_indices is None:
            goal_indices = list(range(len(goals)))

        MAX_TOOLS_PER_GOAL = 6   # Hard cap on total tool calls per goal
        MAX_TOOL_ITERS     = MAX_TOOLS_PER_GOAL  # Upper bound on rounds (≤ cap)

        tools = get_tools_for_standard_generator()
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        system_prompt = (
            f"Current date and time: {current_datetime}\n\n"
            "You are a research assistant. Use tools to gather information for the research goal.\n"
            "Call ONE tool per turn. After seeing results you may call another tool to refine or\n"
            "deepen the research — for example: search first, then fetch specific URLs from the results.\n"
            f"IMPORTANT: You have a budget of {MAX_TOOLS_PER_GOAL} tool calls total for this goal. "
            "Use them wisely: prioritise the most relevant sources and stop as soon as you have "
            "sufficient information — do not waste calls on redundant or low-value lookups.\n\n"
            "CHALLENGE FALSE PREMISES:\n"
            "If the user's question contains incorrect facts, state the correction FIRST:\n"
            "'[X] is not accurate. According to [source], [correct fact]...'\n\n"
            "TOOL SELECTION RULES:\n"
            "- search_documents: ONLY for questions about the user's own uploaded files/documents\n"
            "- query_database / list_databases / get_database_schema: ONLY for the user's databases\n"
            "- duckduckgo_search: for current events, recent news, product releases, general knowledge\n"
            "- search_wikipedia: for encyclopedic background on established topics\n"
            "- google_custom_search: alternative web search if duckduckgo gives poor results\n"
            "- fetch_url_content: ONLY fetch URLs that literally appeared in a duckduckgo_search or search_wikipedia result. NEVER invent, guess, or construct URLs. Do not fetch PDF links.\n\n"
            "If the goal involves recent events or anything unlikely to be in the user's files, use web search.\n\n"
            "Call the tool now — do not write explanatory text."
        )

        # ── Per-goal mutable state ────────────────────────────────────────
        # Each goal gets its own conversation that accumulates tool calls + results
        goal_convs = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Research goal: {goal.description}"}
            ]
            for goal in goals
        ]
        goal_all_results  = [[] for _ in goals]  # accumulated result strings
        goal_all_sources  = [[] for _ in goals]  # accumulated source dicts
        goal_primary_tool = [None] * len(goals)
        goal_tool_count   = [0] * len(goals)      # total tool calls used so far
        goal_done         = [False] * len(goals)

        # ── Iterative tool-use rounds ─────────────────────────────────────
        for tool_iter in range(MAX_TOOL_ITERS):
            active_indices = [i for i in range(len(goals)) if not goal_done[i]]
            if not active_indices:
                break

            print(f"\n[ITER_TOOLS round={tool_iter}] {len(active_indices)} active goal(s): {active_indices}")

            # Build prompts for active goals only
            prompts = []
            for i in active_indices:
                p = llm_tokenizer.apply_chat_template(
                    goal_convs[i],
                    add_generation_prompt=True,
                    tools=tools,
                    tokenize=False
                )
                prompts.append(p)
                print(f"[ITER_TOOLS] Prompt built for goal {i}: {goals[i].description}")

            # Capture loop variables by value for nested closures
            batch_to_goal = {b: i for b, i in enumerate(active_indices)}
            _cur_iter = tool_iter

            # ── Phase A+B: dispatch immediately as each selection finishes ─
            max_workers = min(len(active_indices), 4)
            executor_r = ThreadPoolExecutor(max_workers=max_workers)
            pending_futures: list = []  # list of Futures; result = (goal_idx, iter_results, wants_more)

            def _do_execute(batch_idx, response_text,
                            _iter=_cur_iter, _b2g=batch_to_goal,
                            _counts=goal_tool_count):
                """Execute tool calls for one goal in one round (runs in thread)."""
                goal_idx = _b2g[batch_idx]
                goal = goals[goal_idx]
                response_text = response_text.strip()
                print(f"[ITER_TOOLS r{_iter}] Goal {goal_idx} response: {repr(response_text[:200])}")

                # ── No tool call: fallback on first round, stop on later rounds ─
                if "[TOOL_CALLS]" not in response_text:
                    if _iter == 0:
                        print(f"[ITER_TOOLS r{_iter}] Goal {goal_idx}: no tool — fallback duckduckgo")
                        try:
                            result = Generator._execute_tool(
                                "duckduckgo_search", {"query": goal.description},
                                retriever, llm_model, llm_tokenizer
                            )
                            if not isinstance(result, str):
                                result = json.dumps(result, ensure_ascii=False)
                            return (goal_idx,
                                    [("duckduckgo_search", {"query": goal.description}, result)],
                                    False)
                        except Exception as e:
                            print(f"[ITER_TOOLS r{_iter}] Fallback failed: {e}")
                            return goal_idx, [], False
                    print(f"[ITER_TOOLS r{_iter}] Goal {goal_idx}: no tool call — done")
                    return goal_idx, [], False

                tool_calls = Generator._parse_tool_calls(response_text)
                if not tool_calls:
                    if _iter == 0:
                        try:
                            result = Generator._execute_tool(
                                "duckduckgo_search", {"query": goal.description},
                                retriever, llm_model, llm_tokenizer
                            )
                            if not isinstance(result, str):
                                result = json.dumps(result, ensure_ascii=False)
                            return (goal_idx,
                                    [("duckduckgo_search", {"query": goal.description}, result)],
                                    False)
                        except Exception:
                            return goal_idx, [], False
                    return goal_idx, [], False

                # Trim to the remaining budget for this goal
                remaining = MAX_TOOLS_PER_GOAL - _counts[goal_idx]
                tool_calls = tool_calls[:remaining]
                if not tool_calls:
                    print(f"[ITER_TOOLS r{_iter}] Goal {goal_idx}: tool cap reached — done")
                    return goal_idx, [], False

                # Execute each tool call
                iter_results = []
                for call in tool_calls:
                    tool_name = call.get("name", "")
                    tool_args = call.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {"query": tool_args}
                    try:
                        if tool_name == "search_documents" and goal.strategy == "bm25_only":
                            q = tool_args.get("query", goal.description)
                            result = retriever.search_documents_tool(q, weight_dense=0.0, weight_sparse=1.0)
                            if not isinstance(result, str):
                                result = json.dumps(result, ensure_ascii=False)
                        elif tool_name == "search_documents" and goal.strategy == "dense_only":
                            q = tool_args.get("query", goal.description)
                            result = retriever.search_documents_tool(q, weight_dense=1.0, weight_sparse=0.0)
                            if not isinstance(result, str):
                                result = json.dumps(result, ensure_ascii=False)
                        else:
                            print(f"[ITER_TOOLS r{_iter}] Goal {goal_idx}: {tool_name} {tool_args}")
                            result = Generator._execute_tool(
                                tool_name, tool_args, retriever, llm_model, llm_tokenizer
                            )
                        if not isinstance(result, str):
                            result = json.dumps(result, ensure_ascii=False)
                    except Exception as e:
                        print(f"[ITER_TOOLS r{_iter}] Goal {goal_idx} tool error: {e}")
                        import traceback; traceback.print_exc()
                        result = ""

                    print(f"[ITER_TOOLS r{_iter}] Goal {goal_idx}/{tool_name}: {len(result)} chars")
                    iter_results.append((tool_name, tool_args, result))

                return goal_idx, iter_results, True  # wants_more=True → may call tools again

            def _on_selection(batch_idx, response_text):
                f = executor_r.submit(_do_execute, batch_idx, response_text)
                pending_futures.append(f)

            AgenticGenerator._run_batch_generate(
                llm_model, llm_tokenizer, prompts,
                max_tokens=200,
                label=f"TOOL_SEL_r{tool_iter}",
                on_complete=_on_selection
            )

            # ── Collect results, update per-goal conversations ─────────────
            for future in as_completed(pending_futures):
                goal_idx, iter_results, wants_more = future.result()

                if not iter_results:
                    goal_done[goal_idx] = True
                    continue

                # Build assistant tool-calls entry for conversation
                formatted_calls = []
                for k, (tool_name, tool_args, _r) in enumerate(iter_results):
                    formatted_calls.append({
                        "id": f"r{tool_iter}_c{k}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args)
                        }
                    })
                goal_convs[goal_idx].append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": formatted_calls
                })

                # Inject tool result messages (truncated to keep context manageable)
                for k, (tool_name, tool_args, result) in enumerate(iter_results):
                    truncated = result[:2000] if len(result) > 2000 else result
                    goal_convs[goal_idx].append({
                        "role": "tool",
                        "name": tool_name,
                        "content": truncated,
                        "tool_call_id": f"r{tool_iter}_c{k}"
                    })
                    # Accumulate full result for synthesis context
                    goal_all_results[goal_idx].append(result)
                    if goal_primary_tool[goal_idx] is None:
                        goal_primary_tool[goal_idx] = tool_name
                    label_val = (tool_args.get("url") or tool_args.get("query")
                                 or tool_args.get("document_name") or "")
                    goal_all_sources[goal_idx].append({
                        "tool_name": tool_name,
                        "label": label_val,
                        "chars": len(result),
                        "result": result
                    })

                # Update total tool count for this goal
                goal_tool_count[goal_idx] += len(iter_results)

                # Goal done if LLM stopped, rounds exhausted, or tool cap reached
                if (not wants_more
                        or tool_iter + 1 >= MAX_TOOL_ITERS
                        or goal_tool_count[goal_idx] >= MAX_TOOLS_PER_GOAL):
                    print(f"[ITER_TOOLS] Goal {goal_idx} finalised "
                          f"({goal_tool_count[goal_idx]}/{MAX_TOOLS_PER_GOAL} tools used)")
                    goal_done[goal_idx] = True

            executor_r.shutdown(wait=False)

        # ── Emit reasoning_retrieval events (once per goal, all sources) ──
        if stream_callback:
            for goal_idx, goal in enumerate(goals):
                sources = goal_all_sources[goal_idx]
                combined = "\n---\n".join(goal_all_results[goal_idx])
                primary = goal_primary_tool[goal_idx] or "failed"
                stream_callback('reasoning_retrieval', {
                    'goal': goal.description,
                    'goal_index': goal_indices[goal_idx],
                    'tool_name': primary,
                    'chars_retrieved': len(combined),
                    'preview': combined if combined else "",
                    'sources': sources
                })

        # ── Compile and return final results ──────────────────────────────
        return [
            (
                "\n---\n".join(goal_all_results[i]),
                goal_primary_tool[i] or "failed"
            )
            for i in range(len(goals))
        ]

    # ─────────────────────────────────────────────────────────────
    # Batch evaluation
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _batch_evaluate_goals(goals, retrieved_contexts, all_previous_contexts, # Isto não é oq ta no evaluator.py??
                               llm_model, llm_tokenizer,
                               stream_callback=None, goal_indices=None):
        """Evaluate all goals in parallel using BatchGenerator.

        Evaluation events (reasoning_evaluation) are emitted immediately as
        each goal's evaluation finishes — goals don't wait for each other.

        Returns a list of evaluation dicts in the same order as *goals*.
        """
        from RAG_Framework.agents.planner import parse_json_safely

        if goal_indices is None:
            goal_indices = list(range(len(goals)))

        system_prompt = (
            "Evaluate retrieved information quality and novelty.\n\n"
            "SCORING (0.0-1.0):\n"
            "- Direct answer: +0.4, Complete coverage: +0.3, Specific facts: +0.2, Source quality: +0.1\n\n"
            "NOVELTY (information_gain 0.0-1.0):\n"
            "- 1.0 = entirely new facts not in previous findings\n"
            "- 0.0 = complete duplicate of what we already have\n\n"
            "FLAGS:\n"
            "- sparse_results: true if retrieval returned very little useful content\n"
            "- contradictory_info: true if new findings contradict previous ones\n\n"
            "OUTPUT (JSON only):\n"
            "{\n"
            '    "is_complete": true,\n'
            '    "confidence": 0.8,\n'
            '    "information_gain": 0.7,\n'
            '    "sparse_results": false,\n'
            '    "contradictory_info": false,\n'
            '    "missing_aspects": [],\n'
            '    "reasoning": "brief"\n'
            "}"
        )

        prompts = []
        for goal, retrieved_context in zip(goals, retrieved_contexts):
            prev_summary = ""
            if all_previous_contexts:
                for i, ctx in enumerate(all_previous_contexts[-3:]):
                    prev_summary += f"\n--- Previous finding {i+1} ---\n{ctx[:500]}\n"

            user_prompt = (
                f"Goal: {goal.description}\n\n"
                f"Retrieved context:\n{retrieved_context[:1000]}\n\n"
                f"Previous findings:{prev_summary if prev_summary else ' (none)'}\n\n"
                "Evaluate quality and novelty. Output JSON only."
            )

            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt = llm_tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt)

        evaluations = [None] * len(goals)

        def on_eval_complete(idx, response):
            goal = goals[idx]
            retrieved_context = retrieved_contexts[idx]
            print(f"\n[BATCH_EVAL] Goal {idx} done ({len(response)} chars): {repr(response)}")

            eval_result = parse_json_safely(response)
            if eval_result and "is_complete" in eval_result:
                eval_result.setdefault("information_gain", 0.5)
                eval_result.setdefault("sparse_results", False)
                eval_result.setdefault("contradictory_info", False)
                eval_result.setdefault("missing_aspects", [])
                eval_result.setdefault("reasoning", "")
            else:
                print(f"[BATCH_EVAL] Goal {idx}: parse failed, using fallback")
                eval_result = {
                    "is_complete": len(retrieved_context) > 100 if retrieved_context else False,
                    "confidence": 0.5,
                    "information_gain": 0.5,
                    "sparse_results": len(retrieved_context) < 50 if retrieved_context else True,
                    "contradictory_info": False,
                    "missing_aspects": [],
                    "reasoning": "Fallback evaluation"
                }

            evaluations[idx] = eval_result

            if stream_callback:
                stream_callback('reasoning_evaluation', {
                    'confidence': eval_result.get('confidence', 0.5),
                    'information_gain': eval_result.get('information_gain', 0.5),
                    'is_complete': eval_result.get('is_complete', False),
                    'sparse_results': eval_result.get('sparse_results', False),
                    'contradictory_info': eval_result.get('contradictory_info', False),
                    'reasoning': eval_result.get('reasoning', ''),
                    'goal_index': goal_indices[idx]
                })

        AgenticGenerator._run_batch_generate(
            llm_model, llm_tokenizer, prompts,
            max_tokens=1200, label="BATCH_EVAL",
            on_complete=on_eval_complete
        )

        return evaluations


    # ─────────────────────────────────────────────────────────────
    # Main agentic loop
    # ─────────────────────────────────────────────────────────────

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
        Main agentic loop with batch multi-step reasoning, evaluation, and replanning.

        All pending goals in each round are processed simultaneously:
          - Tool selection via BatchGenerator (parallel LLM inference)
          - Tool execution via ThreadPoolExecutor (parallel I/O)
          - Evaluation via BatchGenerator (parallel LLM inference)

        After each batch round, replanning may add new goals which are then
        processed in the next round.
        """
        from RAG_Framework.agents import AgenticPlanner, AgenticEvaluator
        from RAG_Framework.components.generators import Generator
        from RAG_Framework.agents.tools import get_tools_for_standard_generator

        print(f"\n{'='*60}")
        print(f"AGENTIC REASONING (BATCH): {query}")
        print(f"{'='*60}\n")


        # ── Step 1: Create initial plan ───────────────────────────────
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

        all_previous_contexts = []
        batch_round = 0

        # ── Step 2: Priority-grouped reasoning loop ────────────────────────
        # Goals sharing the same priority are processed as a batch simultaneously.
        # After each priority group is evaluated, the loop checks whether the
        # accumulated results are already sufficient to answer the query.  Only
        # if they are not does it proceed to the next (lower) priority group.
        while batch_round < MAX_REASONING_STEPS:
            pending_goals = [g for g in plan.goals if g.status == "pending"]
            if not pending_goals:
                print("All goals completed!")
                break

            # Group pending goals by priority and sort (lower number = higher priority)
            priority_groups: dict[int, list] = {}
            for g in pending_goals:
                priority_groups.setdefault(g.priority, []).append(g)
            sorted_priorities = sorted(priority_groups.keys())

            sufficient_to_answer = False
            last_evaluation: dict = {}
            any_sparse = False
            any_contradictory = False
            _sparse_replanned = False

            for current_priority in sorted_priorities:
                if batch_round >= MAX_REASONING_STEPS:
                    break

                current_goals = priority_groups[current_priority]
                batch_round += 1

                print(f"\n{'─'*60}")
                print(f"BATCH ROUND {batch_round} — Priority {current_priority}: "
                      f"{len(current_goals)} goal(s)")
                for i, g in enumerate(current_goals):
                    print(f"  {i+1}. [P{g.priority}|{g.strategy}] {g.description}")
                print(f"{'─'*60}")

                for goal in current_goals:
                    goal.status = "in_progress"

                if stream_callback:
                    stream_callback('reasoning_step', {
                        'step': batch_round,
                        'total_steps': MAX_REASONING_STEPS,
                        'goal': f'Priority {current_priority}: {len(current_goals)} goal(s)',
                        'strategy': 'batch' if len(current_goals) > 1 else 'single',
                        'goals': [{'description': g.description, 'priority': g.priority,
                                   'strategy': g.strategy}
                                  for g in current_goals]
                    })

                goal_indices = [plan.goals.index(g) for g in current_goals]

                # ── Step 2a: Tool selection (batch) + execution ────────────
                print(f"\n[P{current_priority}-BATCH] Tool selection + execution "
                      f"for {len(current_goals)} goal(s)...")
                try:
                    batch_results = AgenticGenerator._batch_retrieve_for_goals(
                        goals=current_goals,
                        llm_model=llm_model,
                        llm_tokenizer=llm_tokenizer,
                        retriever=retriever,
                        stream_callback=stream_callback,
                        goal_indices=goal_indices
                    )
                except Exception as e:
                    print(f"[P{current_priority}-BATCH] Retrieval failed: {e}")
                    import traceback
                    traceback.print_exc()
                    for goal in current_goals:
                        goal.status = "failed"
                    continue

                retrieved_contexts = [ctx for ctx, _ in batch_results]

                for goal, retrieved_context in zip(current_goals, retrieved_contexts):
                    if retrieved_context:
                        goal.retrieved_info.append(retrieved_context)

                # ── Step 2b: Evaluation ────────────────────────────────────
                print(f"\n[P{current_priority}-BATCH] Evaluating {len(current_goals)} goal(s)...")
                evaluations = AgenticGenerator._batch_evaluate_goals(
                    goals=current_goals,
                    retrieved_contexts=retrieved_contexts,
                    all_previous_contexts=all_previous_contexts,
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                    stream_callback=stream_callback,
                    goal_indices=goal_indices
                )

                # ── Step 2c: Update goal statuses ─────────────────────────
                info_gain_values = []
                all_current_complete = True
                all_current_confident = True

                for goal, retrieved_context, evaluation in zip(
                        current_goals, retrieved_contexts, evaluations):
                    goal.confidence = evaluation.get("confidence", 0.5)
                    info_gain = evaluation.get("information_gain", 0.5)
                    sparse = evaluation.get("sparse_results", False)
                    contradictory = evaluation.get("contradictory_info", False)
                    is_complete = evaluation.get("is_complete", False)

                    info_gain_values.append(info_gain)
                    any_sparse = any_sparse or sparse
                    any_contradictory = any_contradictory or contradictory
                    last_evaluation = evaluation

                    if not is_complete:
                        all_current_complete = False
                    if goal.confidence < MIN_CONFIDENCE_THRESHOLD:
                        all_current_confident = False

                    print(f"[EVAL] Goal '{goal.description[:60]}':")
                    print(f"  complete={is_complete}, confidence={goal.confidence:.2f}, "
                          f"gain={info_gain:.2f}, sparse={sparse}, "
                          f"contradictory={contradictory}")
                    print(f"  reasoning: {evaluation.get('reasoning', 'N/A')}")

                    goal.status = "completed"
                    if retrieved_context:
                        all_previous_contexts.append(retrieved_context)

                avg_info_gain = (
                    sum(info_gain_values) / len(info_gain_values) if info_gain_values else 0
                )

                # ── Step 2d: Sparse detection — replan immediately ─────────
                # When sparse results are detected, cancel all remaining pre-planned
                # goals and trigger the replanner to produce a reformulated query
                # instead of blindly proceeding to P2, P3, etc.
                if any_sparse:
                    print(f"\n[SPARSE-REPLAN] P{current_priority} returned sparse results "
                          f"— cancelling pre-planned goals and reformulating")
                    cancelled = [g.description for g in plan.goals if g.status == "pending"]
                    for g in plan.goals:
                        if g.status == "pending":
                            g.status = "failed"
                    if cancelled:
                        print(f"[SPARSE-REPLAN] Cancelled pre-planned goals: {cancelled}")
                    sparse_eval = dict(last_evaluation)
                    sparse_eval["failed_query"] = "; ".join(g.description for g in current_goals)
                    pending_before_replan = {g.description for g in plan.goals if g.status == "pending"}
                    plan = AgenticPlanner.replan(plan, sparse_eval, llm_model, llm_tokenizer)
                    if stream_callback:
                        new_pending = [
                            g.description for g in plan.goals
                            if g.status == "pending"
                            and g.description not in pending_before_replan
                        ]
                        if new_pending:
                            stream_callback('reasoning_replan', {
                                'action': 'reformulate',
                                'new_goals': new_pending,
                                'reasoning': last_evaluation.get(
                                    'reasoning', 'Sparse results — reformulating query')
                            })
                    _sparse_replanned = True
                    break  # break inner priority loop; outer while picks up new pending goals

                # ── Step 2e: Decide whether to advance to next priority ────
                if all_current_complete and all_current_confident:
                    print(f"\nPRIORITY STOP: P{current_priority} goals complete and "
                          f"confident — skipping lower-priority goals")
                    sufficient_to_answer = True
                    break

                if avg_info_gain < 0.1 and batch_round > 1:
                    print(f"\nAUTONOMOUS STOP: Information gain too low "
                          f"({avg_info_gain:.2f}) after P{current_priority}")
                    sufficient_to_answer = True
                    break

                print(f"\n[P{current_priority}-BATCH] Results not sufficient "
                      f"(complete={all_current_complete}, "
                      f"confident={all_current_confident}) "
                      f"— advancing to next priority group")

            # ── After priority groups: exit or replan ──────────────────────
            if sufficient_to_answer or plan.is_complete():
                print("Sufficient information gathered — proceeding to synthesis.")
                break

            # ── Step 2f: Replan if needed (non-sparse fallback) ───────────
            completion_rate = plan.get_completion_rate()
            if not _sparse_replanned and completion_rate >= 0.5 and not plan.is_complete():
                completed = [g for g in plan.goals if g.status == "completed"]
                avg_confidence = (
                    sum(g.confidence for g in completed) / max(1, len(completed))
                )

                if avg_confidence < MIN_CONFIDENCE_THRESHOLD or any_sparse or any_contradictory:
                    print(f"\n[REPLAN] Replanning "
                          f"(avg_conf={avg_confidence:.2f}, sparse={any_sparse}, "
                          f"contradictory={any_contradictory})...")
                    pending_before = {g.description for g in plan.goals if g.status == "pending"}
                    plan = AgenticPlanner.replan(plan, last_evaluation, llm_model, llm_tokenizer)

                    if stream_callback:
                        new_goals = [g.description for g in plan.goals
                                     if g.status == "pending"
                                     and g.description not in pending_before]
                        if new_goals:
                            stream_callback('reasoning_replan', {
                                'action': 'replan',
                                'new_goals': new_goals,
                                'reasoning': last_evaluation.get('reasoning', '')
                            })

            if plan.is_complete():
                print("All goals completed after priority cycle.")
                break

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

    # ─────────────────────────────────────────────────────────────
    # Single-goal fallback (kept for compatibility / direct use)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _retrieve_for_goal(goal, llm_model, llm_tokenizer, retriever, stream_callback=None):
        """Use LLM tool-calling to retrieve information for a single goal.

        Kept for external callers / ad-hoc use.  The main agentic loop now
        uses _batch_retrieve_for_goals() for parallel processing.
        """
        results = AgenticGenerator._batch_retrieve_for_goals(
            goals=[goal],
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            retriever=retriever,
            stream_callback=stream_callback
        )
        return results[0]  # (retrieved_context, tool_name)

    # ─────────────────────────────────────────────────────────────
    # Answer synthesis
    # ─────────────────────────────────────────────────────────────

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

        print(f"\n[SYNTHESIS] Total context: {len(synthesis_context)} chars "
              f"from {sum(len(g.retrieved_info) for g in plan.goals)} retrievals")
        print(f"[SYNTHESIS] Context preview:\n{synthesis_context[:500]}...")

        # Build user message with context + query
        user_msg = (
            f"Based on the following retrieved information, answer the query.\n\n"
            f"RETRIEVED INFORMATION:\n{synthesis_context}\n\n"
            f"QUERY: {query}\n\n"
            f"Answer in detail using the retrieved information above. Cite sources where possible."
        )

        if conversation_manager is None:
            conversation_manager = ConversationManager()

        # Build a FRESH synthesis conversation using only the system prompt +
        # the synthesis request.  We intentionally exclude prior conversation
        # history so the model is not influenced by any previous failed/
        # irrelevant assistant turns (e.g. "no info found" from an earlier
        # non-agentic response on the same question).
        system_msgs = [m for m in conversation_manager.get_conversation()
                       if m["role"] == "system"]
        synthesis_conv = system_msgs + [{"role": "user", "content": user_msg}]

        prompt = llm_tokenizer.apply_chat_template(
            synthesis_conv,
            add_generation_prompt=True,
            tokenize=False
        )

        print(f"\n[SYNTHESIS] Formatted prompt:\n{prompt[:500]}...(truncated)")

        sampler = make_sampler(temp=0.3, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.0)

        # Synthesis uses a fresh prompt layout so the KV-cache offset from
        # prior turns will not align.  Pass the full token sequence and reset
        # the cache so it can be rebuilt from this synthesis turn forward.
        full_tokens = llm_tokenizer.encode(prompt, add_special_tokens=False)
        if prompt_cache and len(prompt_cache) > 0 and prompt_cache[0].offset > 0:
            print(f"[SYNTHESIS] Resetting KV-cache for fresh synthesis conversation")
            for layer in prompt_cache:
                if layer.offset > 0:
                    layer.trim(layer.offset)

        print(f"[SYNTHESIS] Prompt tokens: {len(full_tokens)}")

        response = Generator._generate_with_streaming(
            model=llm_model,
            tokenizer=llm_tokenizer,
            prompt=full_tokens,
            max_tokens=MAX_RESPONSE_TOKENS,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=prompt_cache,
            stream_callback=stream_callback
        )

        response_text = response.strip()
        print(f"\n[SYNTHESIS] Response length: {len(response_text)} chars")

        # Record the ORIGINAL query (not the bulky synthesis prompt) in the
        # conversation manager so future turns have clean, readable history.
        conversation_manager.add_user_message(query)
        conversation_manager.add_assistant_message(response_text)

        if prompt_cache is not None:
            CacheManager.log_cache_stats(
                prompt_cache, f"After agentic synthesis (turn {conversation_manager.get_turn_count()})")

        return response_text

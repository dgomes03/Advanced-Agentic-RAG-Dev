from mlx_lm.sample_utils import make_sampler, make_logits_processors

from RAG_Framework.core.config import MAX_RESPONSE_TOKENS, MAX_REASONING_STEPS, MAX_REPLANS_PER_GOAL, MAX_TOTAL_REPLANS
from RAG_Framework.core.conversation_manager import ConversationManager


class AgenticGenerator:
    """Agentic reasoning coordinator.

    Orchestrates multi-step reasoning: planning → batch retrieval →
    batch evaluation → dynamic replanning → streaming synthesis.

    All heavy logic lives in the agents/ package:
      - agents.planner       — plan creation and goal replanning
      - agents.retriever     — DDG→fetch retrieval loops
      - agents.evaluator     — batch goal evaluation
      - agents.batch_generator — BatchGenerator wrapper
      - agents.decoder       — BPE decoding utilities
    """

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
        """Main agentic loop: plan → retrieve → evaluate → replan → synthesize."""
        from RAG_Framework.agents import AgenticPlanner
        from RAG_Framework.agents.evaluator import AgenticEvaluator
        from RAG_Framework.agents.retriever import AgenticRetriever

        print(f"\n{'='*60}")
        print(f"[AGENTIC] REASONING: {query}")
        print(f"{'='*60}\n")

        # ── Step 1: Create initial plan ───────────────────────────────
        plan = AgenticPlanner.create_initial_plan(query, llm_model, llm_tokenizer)

        if stream_callback:
            stream_callback('reasoning_step', {
                'step': 0,
                'total_steps': MAX_REASONING_STEPS,
                'goal': 'Planning complete',
                'strategy': 'planning',
                'goals': [{'description': g.description} for g in plan.goals]
            })

        all_previous_contexts = []
        total_replans = 0
        main_loop_iter = 0

        # ── Step 2: Flat batch execution loop ─────────────────────────
        while not plan.is_complete() and main_loop_iter < MAX_REASONING_STEPS:
            pending_goals = [g for g in plan.goals if g.status == "pending"]
            if not pending_goals:
                print("[AGENTIC] No pending goals — done.")
                break

            main_loop_iter += 1
            print(f"\n{'─'*60}")
            print(f"[AGENTIC] Loop iter {main_loop_iter} — {len(pending_goals)} pending goal(s)")
            for i, g in enumerate(pending_goals, 1):
                print(f"  {i}. {g.description}")
            print(f"{'─'*60}")

            for goal in pending_goals:
                goal.status = "in_progress"

            if stream_callback:
                stream_callback('reasoning_step', {
                    'step': main_loop_iter,
                    'total_steps': MAX_REASONING_STEPS,
                    'goal': f'{len(pending_goals)} goal(s)',
                    'goals': [{'description': g.description} for g in pending_goals]
                })

            goal_indices = [plan.goals.index(g) for g in pending_goals]

            # ── RETRIEVE ──────────────────────────────────────────────
            try:
                batch_results = AgenticRetriever.batch_retrieve_for_goals(
                    goals=pending_goals,
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                    retriever=retriever,
                    stream_callback=stream_callback,
                    goal_indices=goal_indices
                )
            except Exception as e:
                print(f"[AGENTIC] Retrieval failed: {e}")
                import traceback; traceback.print_exc()
                for goal in pending_goals:
                    goal.status = "failed"
                continue

            retrieved_contexts = [ctx for ctx, _ in batch_results]

            for goal, retrieved_context in zip(pending_goals, retrieved_contexts):
                if retrieved_context:
                    goal.retrieved_info.append(retrieved_context)

            # ── EVALUATE ──────────────────────────────────────────────
            evaluations = AgenticEvaluator.batch_evaluate_goals(
                goals=pending_goals,
                retrieved_contexts=retrieved_contexts,
                all_previous_contexts=all_previous_contexts,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                stream_callback=stream_callback,
                goal_indices=goal_indices
            )

            # ── PROCESS each evaluation ────────────────────────────────
            for goal, retrieved_context, evaluation in zip(
                    pending_goals, retrieved_contexts, evaluations):
                gidx = plan.goals.index(goal)
                goal.confidence = evaluation.get("confidence", 0.5)
                sparse = evaluation.get("sparse_results", False)

                print(f"[EVAL] Goal {gidx} '{goal.description[:60]}':")
                print(f"  sparse={sparse}, confidence={goal.confidence:.2f}, "
                      f"gain={evaluation.get('information_gain', 0.5):.2f}")
                print(f"  reasoning: {evaluation.get('reasoning', 'N/A')}")

                if not sparse:
                    goal.status = "completed"
                    if retrieved_context:
                        all_previous_contexts.append(retrieved_context)
                    print(f"[EVAL] Goal {gidx} → completed")
                else:
                    if (goal.replan_count >= MAX_REPLANS_PER_GOAL
                            or total_replans >= MAX_TOTAL_REPLANS):
                        goal.status = "cancelled"
                        print(f"[EVAL] Goal {gidx} → cancelled "
                              f"(replan_count={goal.replan_count}, total_replans={total_replans})")
                        if stream_callback:
                            stream_callback('reasoning_evaluation', {
                                'confidence': goal.confidence,
                                'information_gain': evaluation.get('information_gain', 0.5),
                                'is_complete': False,
                                'sparse_results': True,
                                'contradictory_info': evaluation.get('contradictory_info', False),
                                'reasoning': evaluation.get('reasoning', ''),
                                'goal_index': gidx,
                                'cancelled': True
                            })
                    else:
                        goal.status = "failed"
                        print(f"[REPLAN] Reformulating goal {gidx}: {goal.description}")
                        new_goal = AgenticPlanner.replan_goal(
                            goal, evaluation, llm_model, llm_tokenizer
                        )
                        plan.add_goal(new_goal)
                        total_replans += 1
                        print(f"[REPLAN] New goal {len(plan.goals)-1}: {new_goal.description} "
                              f"(replan_count={new_goal.replan_count}, total_replans={total_replans})")
                        if stream_callback:
                            stream_callback('reasoning_replan', {
                                'action': 'reformulate',
                                'new_goals': [new_goal.description],
                                'replaced_goal_index': gidx,
                                'reasoning': evaluation.get(
                                    'reasoning', 'Sparse results — reformulating')
                            })

        # ── Step 3: Synthesize final answer ──────────────────────────
        print(f"\n{'='*60}")
        print(f"[SYNTHESIS] Synthesizing final answer")
        print(f"{'='*60}")

        return AgenticGenerator._synthesize_answer(
            query=query,
            plan=plan,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            conversation_manager=conversation_manager,
            prompt_cache=prompt_cache,
            stream_callback=stream_callback
        )

    @staticmethod
    def _synthesize_answer(query, plan, llm_model, llm_tokenizer,
                            conversation_manager, prompt_cache, stream_callback):
        """Build a synthesis prompt from all accumulated context and generate
        the final answer with streaming."""
        from RAG_Framework.components.generators import Generator
        from RAG_Framework.core.cache_manager import CacheManager

        synthesis_context = ""
        for goal in plan.goals:
            for info in goal.retrieved_info:
                if isinstance(info, str):
                    synthesis_context += info + "\n---\n"
                elif isinstance(info, tuple):
                    synthesis_context += str(info[0]) + "\n---\n"
                else:
                    synthesis_context += str(info) + "\n---\n"

        if len(synthesis_context) > 8000:
            synthesis_context = synthesis_context[:8000] + "\n...(truncated)"

        print(f"\n[SYNTHESIS] Total context: {len(synthesis_context)} chars "
              f"from {sum(len(g.retrieved_info) for g in plan.goals)} retrievals")

        user_msg = (
            f"Based on the following retrieved information, answer the query.\n\n"
            f"RETRIEVED INFORMATION:\n{synthesis_context}\n\n"
            f"QUERY: {query}\n\n"
            f"Answer in detail using the retrieved information above. Cite sources where possible."
        )

        if conversation_manager is None:
            conversation_manager = ConversationManager()

        system_msgs = [m for m in conversation_manager.get_conversation()
                       if m["role"] == "system"]
        synthesis_conv = system_msgs + [{"role": "user", "content": user_msg}]

        prompt = llm_tokenizer.apply_chat_template(
            synthesis_conv, add_generation_prompt=True, tokenize=False
        )

        sampler = make_sampler(temp=0.3, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.0)

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
        conversation_manager.add_user_message(query)
        conversation_manager.add_assistant_message(response_text)

        if prompt_cache is not None:
            CacheManager.log_cache_stats(
                prompt_cache,
                f"After agentic synthesis (turn {conversation_manager.get_turn_count()})")

        return response_text

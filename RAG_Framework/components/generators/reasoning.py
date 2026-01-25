import json
import random
import string
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from RAG_Framework.core.config import MAX_RESPONSE_TOKENS, MIN_CONFIDENCE_THRESHOLD, MAX_REASONING_STEPS, ENABLE_SELECTIVE_CACHING
from RAG_Framework.core.cache_manager import CacheManager


class AgenticGenerator:

    """Main agentic reasoning coordinator"""
    
    @staticmethod
    def agentic_answer_query(
        query: str,
        llm_model,
        llm_tokenizer,
        retriever,
        prompt_cache=None
    ) -> str:
        """
        Main agentic loop with multi-step reasoning, evaluation, and replanning
        """
        # Import here to avoid circular dependency
        from RAG_Framework.agents import AgenticPlanner, AgenticEvaluator
        from RAG_Framework.components.generators import Generator

        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}\n")

        # Step 1: Create initial plan
        print("Creating reasoning plan...")
        plan = AgenticPlanner.create_initial_plan(query, llm_model, llm_tokenizer)
        
        reasoning_step = 0
        max_steps = MAX_REASONING_STEPS
        
        # Step 2: Iterative reasoning loop
        while reasoning_step < max_steps:
            reasoning_step += 1
            print(f"\n{'─'*60}")
            print(f"REASONING STEP {reasoning_step}/{max_steps}")
            print(f"{'─'*60}")
            
            # Get next goal to work on
            current_goal = plan.get_next_goal()
            
            if current_goal is None:
                print("All goals completed!")
                break
            
            print(f"Current Goal: {current_goal.description}")
            print(f"Priority: {current_goal.priority} | Status: {current_goal.status}")
            
            # Mark goal as in progress
            current_goal.status = "in_progress"
            
            # Step 3: Search for information
            print(f"\nSearching for relevant information...")
            try:
                retrieved_context, _ = AgenticGenerator.available_tools(current_goal.description, llm_model, llm_tokenizer, retriever)
                if retrieved_context:
                    current_goal.retrieved_info.append(retrieved_context)
                    print(f"Retrieved {len(retrieved_context)} characters of context")
                else:
                    print("No context retrieved (empty response)")
                    retrieved_context = ""  # Ensure it's a string for downstream use
            except Exception as e:
                print(f"Search failed: {e}")
                current_goal.status = "failed"
                continue
            
            # Step 4: Evaluate goal completion
            print(f"\nEvaluating goal completion...")
            evaluation = AgenticEvaluator.evaluate_goal_completion(
                current_goal,
                retrieved_context,
                llm_model,
                llm_tokenizer
            )
            
            current_goal.confidence = evaluation.get("confidence", 0.5)
            
            print(f"Complete: {evaluation.get('is_complete', False)}")
            print(f"Confidence: {current_goal.confidence:.2f}")
            print(f"Reasoning: {evaluation.get('reasoning', 'N/A')}")
            
            if evaluation.get("missing_aspects"):
                print(f"Missing: {', '.join(evaluation.get('missing_aspects', []))}")
            
            # Update goal status based on evaluation
            if evaluation.get("is_complete", False) and current_goal.confidence >= MIN_CONFIDENCE_THRESHOLD:
                current_goal.status = "completed"
                print(f"Goal completed successfully!")
            elif current_goal.confidence < MIN_CONFIDENCE_THRESHOLD:
                print(f"Low confidence - may need more information")
                current_goal.status = "completed"  # Move on but flag low confidence
            else:
                current_goal.status = "completed"  # Mark as done even if not perfect
            
            # Step 5: Check overall progress and decide if replanning needed
            print(f"\nOverall Progress: {plan.get_completion_rate()*100:.0f}% complete")
            
            # Evaluate overall completeness
            if plan.get_completion_rate() >= 0.5:  # Check after 50% completion
                print(f"\nEvaluating overall completeness...")
                overall_eval = AgenticEvaluator.evaluate_overall_completeness(
                    plan,
                    llm_model,
                    llm_tokenizer
                )
                
                print(f"Can answer: {overall_eval.get('can_answer', False)}")
                print(f"Overall confidence: {overall_eval.get('overall_confidence', 0):.2f}")
                print(f"Assessment: {overall_eval.get('coverage_assessment', 'N/A')}")
                
                # Step 6: Autonomous stopping decision
                if overall_eval.get("can_answer", False) and overall_eval.get("overall_confidence", 0) >= MIN_CONFIDENCE_THRESHOLD:
                    print(f"\nAUTONOMOUS STOP: Sufficient information gathered")
                    print(f"Confidence threshold met: {overall_eval.get('overall_confidence', 0):.2f} >= {MIN_CONFIDENCE_THRESHOLD}")
                    break
                
                # Step 7: Dynamic replanning if needed
                if overall_eval.get("needs_more_search", False) and reasoning_step < max_steps - 1:
                    print(f"\nReplanning needed...")
                    plan = AgenticPlanner.replan(plan, overall_eval, llm_model, llm_tokenizer)
        
        # Step 8: Generate final answer
        print(f"\n{'='*60}")
        print(f"GENERATING FINAL ANSWER")
        print(f"{'='*60}")
        
        current_response = Generator.answer_query_with_llm(query, llm_model, llm_tokenizer, retriever, prompt_cache=prompt_cache)
        

        return current_response
    
    @staticmethod
    def available_tools(query, llm_model, llm_tokenizer, retriever, prompt_cache=None):
        # Import here to avoid circular dependency
        from RAG_Framework.components.generators import Generator
        from RAG_Framework.agents.tools import get_tools_for_agentic_generator

        tools = get_tools_for_agentic_generator()

        # Temperature and top sampling
        sampler = make_sampler(temp=0.7, top_k=50, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.1, repetition_context_size=128)
        current_response = None

        # Selective caching: track checkpoint before tool results are added
        pre_tool_checkpoint = None

        # Enhanced system prompt to guide tool selection
        conversation = [
            {"role": "system", "content": "You are a helpful assistant with document search and Wikipedia search capabilities. "
            "Decide whether you need tools. If you use tools, answer based *only* on the provided results. "
            "If you're not sure if the user wants you to access tools, ask the user. "
            "After receiving tool results, provide a final answer. "
            "If not enough information is found after tool calling, alert the user that there's no available information to answer the user. "
            "At the end of an informative response, ask if the user needs more information or wants to explore more a certain fact. "
            "**Do not make sequential tool calls**!"}
        ]

        # Add current query
        conversation.append({"role": "user", "content": query})
        
        while True: ############### este loop é mesmo necessário? ##################
            prompt = llm_tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tools=tools,
                tokenize=False
            )

            #print("\nFORMATTED PROMPT:") 
            #print(prompt)

            response = generate(
                model=llm_model,
                tokenizer=llm_tokenizer,
                prompt=prompt,
                max_tokens=MAX_RESPONSE_TOKENS,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=prompt_cache,
                verbose=True
            )

            response_text = response.strip()

            # tool call present?
            if "[TOOL_CALLS]" in response_text:
                print(f"\nModel requested to use tools. Processing tool calls...")

                # Save cache checkpoint BEFORE processing tool results (selective caching)
                if ENABLE_SELECTIVE_CACHING and prompt_cache is not None:
                    pre_tool_checkpoint = CacheManager.get_checkpoint(prompt_cache)
                    CacheManager.log_cache_stats(prompt_cache, "Pre-tool checkpoint saved")

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
                        tool_section = response_text.split("[TOOL_CALLS]")[1]
                        tool_name, args_part = tool_section.split("[ARGS]", 1)
                        tool_name = tool_name.strip()
                        
                        # Extract JSON object
                        args_part = args_part.strip()
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
                        tool_calls = [{"name": tool_name, "arguments": tool_args}]
                    
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
                        args_str = tool_call["function"]["arguments"]
                        try:
                            tool_args = json.loads(args_str)
                        except json.JSONDecodeError as e:
                            tool_result = f"Tool call error: Invalid arguments format - {str(e)}"
                        else:
                            print(f"Executing tool {i+1}/{len(formatted_tool_calls)}: {tool_name} with args: {tool_args}")
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

                # Restore cache checkpoint to exclude tool results (selective caching)
                if ENABLE_SELECTIVE_CACHING and pre_tool_checkpoint is not None:
                    CacheManager.log_cache_stats(prompt_cache, "Before cache restore")
                    CacheManager.restore_checkpoint(prompt_cache, pre_tool_checkpoint)
                    CacheManager.log_cache_stats(prompt_cache, "After cache restore (tool results excluded)")
                    pre_tool_checkpoint = None  # Reset for next query

                # FIXED: Return the actual response text and prompt
                return response_text, prompt
        # If we reach maximum iterations or break due to error, return empty response
        print(f"Fatal Error: Tool processing loop ended unexpectedly")
        return "", prompt

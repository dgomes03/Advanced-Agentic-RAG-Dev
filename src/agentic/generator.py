"""Main agentic reasoning coordinator"""

import json
import random
import string
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from ..config import config
from .planner import AgenticPlanner
from .evaluator import AgenticEvaluator


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
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}\n")

        # Step 1: Create initial plan
        print("Creating reasoning plan...")
        plan = AgenticPlanner.create_initial_plan(query, llm_model, llm_tokenizer)

        reasoning_step = 0
        max_steps = config.MAX_REASONING_STEPS

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
                retrieved_context = AgenticGenerator.available_tools(
                    current_goal.description, llm_model, llm_tokenizer, retriever
                )
                current_goal.retrieved_info.append(retrieved_context)
                print(f"Retrieved {len(retrieved_context)} characters of context")
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
            if evaluation.get("is_complete", False) and current_goal.confidence >= config.MIN_CONFIDENCE_THRESHOLD:
                current_goal.status = "completed"
                print(f"Goal completed successfully!")
            elif current_goal.confidence < config.MIN_CONFIDENCE_THRESHOLD:
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
                if overall_eval.get("can_answer", False) and overall_eval.get("overall_confidence", 0) >= config.MIN_CONFIDENCE_THRESHOLD:
                    print(f"\nAUTONOMOUS STOP: Sufficient information gathered")
                    print(f"Confidence threshold met: {overall_eval.get('overall_confidence', 0):.2f} >= {config.MIN_CONFIDENCE_THRESHOLD}")
                    break

                # Step 7: Dynamic replanning if needed
                if overall_eval.get("needs_more_search", False) and reasoning_step < max_steps - 1:
                    print(f"\nReplanning needed...")
                    plan = AgenticPlanner.replan(plan, overall_eval, llm_model, llm_tokenizer)

        # Step 8: Generate final answer
        print(f"\n{'='*60}")
        print(f"GENERATING FINAL ANSWER")
        print(f"{'='*60}")

        # Import Generator here to avoid circular import
        from ..generation import Generator
        current_response = Generator.answer_query_with_llm(
            query, llm_model, llm_tokenizer, retriever, prompt_cache=prompt_cache
        )

        return current_response

    @staticmethod
    def available_tools(query, llm_model, llm_tokenizer, retriever, prompt_cache=None):
        """Execute tool calls for retrieving information"""
        # Import Generator here to avoid circular import
        from ..generation import Generator

        # Define available tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search for documents relevant to the user's query. Use for general questions or when you need to find information across all documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_document_by_name",
                    "description": "Retrieve an entire document by its filename. Use when user specifically asks for a particular book, report, or document by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_name": {
                                "type": "string",
                                "description": "The name of the document to retrieve (e.g., 'annual_report.pdf')"
                            }
                        },
                        "required": ["document_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_available_documents",
                    "description": "List all available documents in the system. Use when user asks what documents are available or wants to browse the document collection.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_wikipedia",
                    "description": "Search Wikipedia for factual information and general knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string for Wikipedia."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "google_custom_search",
                    "description": "Search the internet using Google Custom Search JSON API. Use for current events or information not found in documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string."}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # Temperature and top sampling
        sampler = make_sampler(temp=0.7, top_k=50, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.1, repetition_context_size=128)
        current_response = None

        # System prompt
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

        MAX_TOOL_ITERATIONS = 3
        for iteration in range(MAX_TOOL_ITERATIONS):
            prompt = llm_tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tools=tools,
                tokenize=False
            )

            response = generate(
                model=llm_model,
                tokenizer=llm_tokenizer,
                prompt=prompt,
                max_tokens=config.MAX_RESPONSE_TOKENS,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=prompt_cache,
                verbose=True
            )

            response_text = response.strip()

            # Check for tool calls
            if "[TOOL_CALLS][" not in response_text:
                return response_text, prompt

            # Process tool calls
            print(f"\nModel requested to use tools. Processing tool calls...")
            try:
                # Extract tool calls JSON
                start_marker = "[TOOL_CALLS]["
                start_idx = response_text.find(start_marker)
                if start_idx == -1:
                    print("Tool call pattern not found correctly")
                    break

                start_idx += len(start_marker) - 1
                bracket_count = 1
                end_idx = start_idx + 1

                while end_idx < len(response_text) and bracket_count > 0:
                    if response_text[end_idx] == '[':
                        bracket_count += 1
                    elif response_text[end_idx] == ']':
                        bracket_count -= 1
                    end_idx += 1

                if bracket_count != 0:
                    print("Unbalanced brackets in tool calls")
                    break

                tool_json = response_text[start_idx:end_idx]
                print(f"Extracted tool JSON: {tool_json}")

                try:
                    tool_calls = json.loads(tool_json)
                    print(f"Successfully parsed {len(tool_calls)} tool calls")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse tool calls JSON: {e}")
                    break

                if not tool_calls:
                    print("No valid tool calls found")
                    break

                print(f"Found {len(tool_calls)} tool call(s): {[call.get('name', 'unknown') for call in tool_calls]}")

                # Format tool calls
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
                            if tool_name in ["search_documents", "search_wikipedia", "google_custom_search"]:
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
                    "content": None,
                    "tool_calls": formatted_tool_calls
                })

                # Execute tools
                for i, tool_call in enumerate(formatted_tool_calls):
                    tool_name = tool_call["function"]["name"]
                    args_str = tool_call["function"]["arguments"]

                    try:
                        tool_args = json.loads(args_str)
                    except json.JSONDecodeError as e:
                        tool_result = f"Tool call error: Invalid arguments format - {str(e)}"
                    else:
                        print(f"Executing tool {i+1}/{len(formatted_tool_calls)}: {tool_name} with args: {tool_args}")

                        # Execute the appropriate tool
                        if tool_name == "search_documents":
                            query_str = tool_args.get("query", "")
                            tool_result = retriever.search_documents_tool(query_str)
                        elif tool_name == "retrieve_document_by_name":
                            doc_name = tool_args.get("document_name", "")
                            tool_result = retriever.retrieve_document_by_name_tool(doc_name)
                        elif tool_name == "list_available_documents":
                            tool_result = retriever.list_available_documents_tool()
                        elif tool_name == "search_wikipedia":
                            query_str = tool_args.get("query", "")
                            tool_result = Generator.search_wikipedia(query_str)
                        elif tool_name == "google_custom_search":
                            query_str = tool_args.get("query", "")
                            tool_result = Generator.google_custom_search(query_str)
                        else:
                            tool_result = f"Error: Unknown tool: {tool_name}"

                    if not isinstance(tool_result, str):
                        tool_result = json.dumps(tool_result, ensure_ascii=False)

                    conversation.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": tool_result,
                        "tool_call_id": tool_call["id"]
                    })

                print("All tool executions completed. Preparing final response...")
                continue

            except Exception as e:
                print(f"Error processing tool calls: {str(e)}")
                import traceback
                traceback.print_exc()
                break

        # If we reach maximum iterations, return the current response
        print(f"Warning: Reached maximum tool iterations ({MAX_TOOL_ITERATIONS})")
        return current_response or "Unable to complete request", prompt

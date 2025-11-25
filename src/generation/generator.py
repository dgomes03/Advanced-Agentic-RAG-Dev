"""LLM-based response generation and external tools"""

import json
import random
import string
import urllib.parse
import urllib.request
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from ..config import config


class Generator:
    """Handles LLM-based response generation and external tool integration"""

    @staticmethod
    def summarize_passages(passages, llm_model, llm_tokenizer):
        """Summarize retrieved passages using the LLM"""
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

            # Structure the results
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
            if not config.GOOGLE_CX or config.GOOGLE_CX == "YOUR_GOOGLE_CX_HERE":
                return "Error: Google Custom Search Engine ID (CX) is not configured. Please set GOOGLE_CX in the code."

            encoded_query = urllib.parse.quote(query)
            search_url = (
                f"https://www.googleapis.com/customsearch/v1?"
                f"key={config.GOOGLE_API_KEY}&cx={config.GOOGLE_CX}&q={encoded_query}"
            )

            with urllib.request.urlopen(search_url, timeout=10) as response:
                search_data = json.loads(response.read().decode('utf-8'))

            if 'items' not in search_data:
                return "No results found."

            results = []
            for item in search_data['items'][:5]:  # Limit to top 5
                title = item.get('title', 'No title')
                snippet = item.get('snippet', 'No snippet')
                link = item.get('link', 'No link')
                results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")

            return "\n---\n".join(results)
        except Exception as e:
            return f"Error performing Google Search: {str(e)}"

    @staticmethod
    def answer_query_with_llm(query, llm_model, llm_tokenizer, retriever, prompt_cache=None):
        """Generate answer using LLM with tool calling capabilities"""

        # Define tools based on advanced reasoning mode
        if config.ADVANCED_REASONING:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "agentic_generator",
                        "description": "Call advanced task agent if user is asking for a difficult task.",
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
        else:
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

        # Enhanced system prompt
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

        while True:
            prompt = llm_tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tools=tools,
                tokenize=False
            )

            print("\nFORMATTED PROMPT:")
            print(prompt)

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
            if "[TOOL_CALLS][" in response_text:
                print(f"\nModel requested to use tools. Processing tool calls...")
                try:
                    start_marker = "[TOOL_CALLS]["
                    end_marker = "]"
                    start_idx = response_text.find(start_marker)
                    if start_idx == -1:
                        print("Tool call pattern not found correctly")
                        break
                    start_idx += len(start_marker) - 1

                    # Find matching closing bracket
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

                    # Extract JSON array
                    tool_json = response_text[start_idx:end_idx]
                    print(f"Extracted tool JSON: {tool_json}")

                    # Parse JSON
                    try:
                        tool_calls = json.loads(tool_json)
                        print(f"Successfully parsed {len(tool_calls)} tool calls")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse tool calls JSON: {e}")
                        # Manual extraction fallback
                        tool_content = response_text[start_idx+1:end_idx-1]
                        print(f"Trying manual extraction with: {tool_content}")
                        tool_calls = []
                        if "}, {" in tool_content:
                            parts = tool_content.split("}, {")
                            for i, part in enumerate(parts):
                                if i == 0:
                                    part = part + "}"
                                elif i == len(parts) - 1:
                                    part = "{" + part
                                else:
                                    part = "{" + part + "}"
                                try:
                                    tool_call = json.loads(part)
                                    tool_calls.append(tool_call)
                                except:
                                    print(f"Failed to parse part: {part}")
                        else:
                            try:
                                tool_call = json.loads("{" + tool_content + "}")
                                tool_calls = [tool_call]
                            except:
                                print("Failed to parse single tool call")
                                break

                    if not tool_calls:
                        print("No valid tool calls found")
                        break

                    print(f"Found {len(tool_calls)} tool call(s): {[call.get('name', 'unknown') for call in tool_calls]}")

                    # Format tool calls
                    formatted_tool_calls = []
                    for call in tool_calls:
                        call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))

                        # Handle different argument structures
                        if isinstance(call.get("arguments"), dict):
                            arguments_str = json.dumps(call["arguments"])
                        elif isinstance(call.get("arguments"), str):
                            try:
                                parsed_args = json.loads(call["arguments"])
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
                            elif tool_name == "agentic_generator":
                                query_str = tool_args.get("query", "")
                                # Import here to avoid circular import
                                from ..agentic import AgenticGenerator
                                tool_result = AgenticGenerator.agentic_answer_query(
                                    query_str, llm_model, llm_tokenizer, retriever
                                )
                            elif tool_name == "google_custom_search":
                                query_str = tool_args.get("query", "")
                                tool_result = Generator.google_custom_search(query_str)
                            else:
                                tool_result = f"Error: Unknown tool: {tool_name}"

                        # Convert result to string if needed
                        if not isinstance(tool_result, str):
                            tool_result = json.dumps(tool_result, ensure_ascii=False)

                        # Add tool result to conversation
                        conversation.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_result,
                            "tool_call_id": tool_call["id"]
                        })
                        tool_results.append(tool_result)

                    print("All tool executions completed. Preparing final response...")
                    continue

                except Exception as e:
                    print(f"Error processing tool calls: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break
            else:
                # No tool calls needed, return the response
                print("No tool calls detected, returning response")
                return response_text, prompt

        # If we reach here, return current response
        print(f"Fatal Error")
        return current_response, prompt

    @staticmethod
    def eval(query, llm_model, llm_tokenizer, rag_response):
        """Evaluate RAG response quality"""
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

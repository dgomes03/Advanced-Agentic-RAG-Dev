import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from RAG_Framework.core.config import (
    DUCKDUCKGO_RESULTS_COUNT, MAX_URLS_PER_SEARCH_LOOP, MAX_SEARCH_LOOPS, MAX_FETCH_URL_CHARS
)


class AgenticRetriever:
    """Handles deterministic web retrieval for reasoning goals.

    Uses a DDG→fetch pattern: DuckDuckGo search followed by URL fetching.
    """

    @staticmethod
    def batch_retrieve_for_goals(goals, llm_model, llm_tokenizer, retriever,
                                  stream_callback=None, goal_indices=None):
        """Retrieve information for every goal using DDG→fetch loops.

        Returns a list of (retrieved_context: str, tool_name: str) tuples
        in the same order as *goals*.
        """
        from RAG_Framework.components.generators import Generator

        if goal_indices is None:
            goal_indices = list(range(len(goals)))

        goal_all_results = [[] for _ in goals]

        for search_loop in range(MAX_SEARCH_LOOPS):
            print(f"\n[SEARCH_LOOP {search_loop+1}/{MAX_SEARCH_LOOPS}] {len(goals)} goal(s)")

            # ── Step A: DDG search for all goals in parallel ───────────────
            def _ddg_search(i):
                goal = goals[i]
                print(f"[DDG] Goal {i}: {goal.description}")
                try:
                    result = Generator._execute_tool(
                        "duckduckgo_search",
                        {"query": goal.description, "max_results": DUCKDUCKGO_RESULTS_COUNT},
                        retriever, llm_model, llm_tokenizer
                    )
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)
                    print(f"[DDG] Goal {i}: {len(result)} chars")
                    return i, result
                except Exception as e:
                    print(f"[DDG] Goal {i} error: {e}")
                    return i, ""

            ddg_results = {}
            with ThreadPoolExecutor(max_workers=min(len(goals), 4)) as ex:
                futures = {ex.submit(_ddg_search, i): i for i in range(len(goals))}
                for f in as_completed(futures):
                    i, result = f.result()
                    ddg_results[i] = result
                    if result:
                        goal_all_results[i].append(result)
                        if stream_callback:
                            stream_callback('reasoning_retrieval', {
                                'goal': goals[i].description,
                                'goal_index': goal_indices[i],
                                'tool_name': 'duckduckgo_search',
                                'chars_retrieved': len(result),
                                'preview': result[:500],
                                'sources': [{'tool_name': 'duckduckgo_search',
                                             'label': goals[i].description,
                                             'chars': len(result), 'result': result}]
                            })

            # ── Step B: Extract URLs and fetch in parallel ─────────────────
            def _extract_urls(ddg_result):
                urls = []
                try:
                    data = json.loads(ddg_result)
                    for item in data.get("results", []):
                        url = item.get("url", "")
                        if url and not url.lower().endswith(".pdf") and "youtube.com" not in url.lower() and "youtu.be" not in url.lower():
                            urls.append(url)
                            if len(urls) >= MAX_URLS_PER_SEARCH_LOOP:
                                break
                except (json.JSONDecodeError, AttributeError):
                    pass
                return urls

            fetch_tasks = []
            for i in range(len(goals)):
                for url in _extract_urls(ddg_results.get(i, ""))[:MAX_URLS_PER_SEARCH_LOOP]:
                    fetch_tasks.append((i, url))

            def _fetch_url(task):
                goal_idx, url = task
                print(f"[FETCH] Goal {goal_idx}: {url}")
                try:
                    result = Generator._execute_tool(
                        "fetch_url_content", {"url": url},
                        retriever, llm_model, llm_tokenizer
                    )
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)
                    result = result[:MAX_FETCH_URL_CHARS]
                    print(f"[FETCH] Goal {goal_idx} {url[:60]}: {len(result)} chars")
                    return goal_idx, url, result
                except Exception as e:
                    print(f"[FETCH] Goal {goal_idx} {url[:60]} error: {e}")
                    return goal_idx, url, ""

            if fetch_tasks:
                with ThreadPoolExecutor(max_workers=min(len(fetch_tasks), 6)) as ex:
                    futures = {ex.submit(_fetch_url, task): task for task in fetch_tasks}
                    for f in as_completed(futures):
                        goal_idx, url, result = f.result()
                        if result:
                            goal_all_results[goal_idx].append(result)
                            if stream_callback:
                                stream_callback('reasoning_retrieval', {
                                    'goal': goals[goal_idx].description,
                                    'goal_index': goal_indices[goal_idx],
                                    'tool_name': 'fetch_url_content',
                                    'chars_retrieved': len(result),
                                    'preview': result[:500],
                                    'sources': [{'tool_name': 'fetch_url_content',
                                                 'label': url, 'chars': len(result),
                                                 'result': result}]
                                })

        return [
            ("\n---\n".join(goal_all_results[i]), "duckduckgo_search")
            for i in range(len(goals))
        ]

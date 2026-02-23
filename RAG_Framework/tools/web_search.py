import json
import urllib.parse
import urllib.request

from RAG_Framework.core.config import GOOGLE_API_KEY, GOOGLE_CX, DUCKDUCKGO_RESULTS_COUNT


def duckduckgo_search(query: str, max_results: int = None) -> dict:
    """Search DuckDuckGo (free, no API key required)."""
    if max_results is None:
        max_results = DUCKDUCKGO_RESULTS_COUNT
    else:
        # Never return fewer results than the configured minimum
        max_results = max(DUCKDUCKGO_RESULTS_COUNT, max_results)
    try:
        DDGS = None
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return {
                    'error': 'DuckDuckGo search requires the ddgs package. Install with: pip install ddgs',
                    'query': query,
                }

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', ''),
                })

        return {'query': query, 'results': results, 'total_results': len(results)}
    except Exception as e:
        return {'error': str(e), 'query': query}


def google_custom_search(query: str) -> str:
    """Fetch Google Custom Search results."""
    try:
        if GOOGLE_CX == "YOUR_GOOGLE_CX_HERE":
            return "Error: Google Custom Search Engine ID (CX) is not configured."

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
        for item in search_data['items'][:8]:
            title = item.get('title', 'No title')
            snippet = item.get('snippet', 'No snippet')
            link = item.get('link', 'No link')
            results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")

        return "\n---\n".join(results)
    except Exception as e:
        return f"Error performing Google Search: {str(e)}"

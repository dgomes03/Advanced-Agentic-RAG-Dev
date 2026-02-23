import json
import urllib.parse
import urllib.request


def search_wikipedia(query: str) -> dict:
    """Fetch Wikipedia results with structured data for RAG."""
    try:
        encoded_query = urllib.parse.quote(query)
        search_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={encoded_query}&format=json&srlimit=1&srnamespace=0"
        )
        headers = {'User-Agent': 'RAGFramework/1.0 (contact@example.com)'}
        search_req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(search_req, timeout=10) as response:
            search_data = json.loads(response.read().decode('utf-8'))

        if not search_data.get('query', {}).get('search'):
            return {'error': 'No results found'}

        page_ids = [str(item['pageid']) for item in search_data['query']['search']]
        content_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts|info"
            f"&pageids={'|'.join(page_ids)}&format=json&explaintext=true"
            f"&inprop=url"
        )
        content_req = urllib.request.Request(content_url, headers=headers)
        with urllib.request.urlopen(content_req, timeout=10) as response:
            content_data = json.loads(response.read().decode('utf-8'))

        articles = []
        for page_id in page_ids:
            page_info = content_data['query']['pages'].get(page_id)
            if page_info and 'extract' in page_info:
                content = page_info['extract'].strip()
                words = content.split()
                if len(words) > 2000:
                    content = ' '.join(words[:2000]) + '... (content truncated to 2000 words)'
                    wordcount = 2000
                else:
                    wordcount = len(words)
                articles.append({
                    'title': page_info['title'],
                    'content': content,
                    'pageid': page_info['pageid'],
                    'url': page_info.get('fullurl', f"https://en.wikipedia.org/?curid={page_id}"),
                    'wordcount': wordcount
                })

        return {'query': query, 'articles': articles, 'total_results': len(articles)}
    except Exception as e:
        return {'error': str(e)}

import re
import urllib.parse
import urllib.request

from RAG_Framework.core.config import MAX_FETCH_URL_CHARS


def fetch_url_content(url: str, max_chars: int = None) -> dict:
    """Fetch and extract main text content from a URL."""
    if max_chars is None:
        max_chars = MAX_FETCH_URL_CHARS
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ['http', 'https']:
            return {'error': 'Invalid URL scheme', 'url': url}

        headers = {'User-Agent': 'Mozilla/5.0 (compatible; RAGBot/1.0)'}
        request = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(request, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')

        # Strip scripts, styles, nav, header, footer
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.I)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.I)
        html = re.sub(r'<(nav|header|footer)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.I)

        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > max_chars:
            text = text[:max_chars] + '...'

        return {'url': url, 'content': text, 'char_count': len(text), 'success': True}
    except Exception as e:
        return {'error': str(e), 'url': url}

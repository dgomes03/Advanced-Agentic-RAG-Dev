"""
Text processing utilities for indexing and retrieval.

This module provides functions for:
- BM25 tokenization with stemming and stop word removal
- Text cleaning and normalization
- E5 embedding prefix preparation
"""

import re
import unicodedata
from functools import lru_cache

# Lazy imports for NLTK to avoid loading unless needed
_stemmer_cache = {}
_stop_words_cache = None


def _get_stop_words(languages):
    """Lazy load stop words from NLTK."""
    global _stop_words_cache
    if _stop_words_cache is None:
        try:
            from nltk.corpus import stopwords
            import nltk
            # Ensure stopwords are downloaded
            try:
                stopwords.words('english')
            except LookupError:
                nltk.download('stopwords', quiet=True)

            _stop_words_cache = set()
            for lang in languages:
                try:
                    _stop_words_cache.update(stopwords.words(lang))
                except OSError:
                    # Language not available, skip
                    pass
        except ImportError:
            # NLTK not installed, use minimal default stop words
            _stop_words_cache = {
                'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'must',
                'this', 'that', 'these', 'those', 'it', 'its'
            }
    return _stop_words_cache


def _get_stemmer(language='english'):
    """Lazy load stemmer from NLTK."""
    if language not in _stemmer_cache:
        try:
            from nltk.stem import SnowballStemmer
            _stemmer_cache[language] = SnowballStemmer(language)
        except ImportError:
            # Return None if NLTK not available
            _stemmer_cache[language] = None
        except ValueError:
            # Language not supported by SnowballStemmer
            _stemmer_cache[language] = None
    return _stemmer_cache[language]


def tokenize_for_bm25(text, enable_stemming=True, enable_stopwords=True,
                      languages=None):
    """
    Tokenize text for BM25 indexing/retrieval with optional stemming and stop word removal.

    Args:
        text: Input text to tokenize
        enable_stemming: Whether to apply stemming (default True)
        enable_stopwords: Whether to remove stop words (default True)
        languages: List of languages for stop words (default ['english', 'portuguese'])

    Returns:
        List of processed tokens
    """
    if languages is None:
        languages = ['english', 'portuguese']

    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())

    # Split into tokens
    tokens = text.split()

    # Remove stop words if enabled
    if enable_stopwords:
        stop_words = _get_stop_words(languages)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # Apply stemming if enabled
    if enable_stemming:
        stemmer = _get_stemmer('english')
        if stemmer:
            tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def clean_text(text):
    """
    Clean and normalize text for better embedding quality.

    Handles:
    - Unicode normalization
    - Whitespace cleanup
    - PDF hyphenation fixes
    - Header/footer removal patterns

    Args:
        text: Raw text from PDF extraction

    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""

    # Unicode normalization (NFKC for compatibility)
    text = unicodedata.normalize('NFKC', text)

    # Fix PDF hyphenation (word-\n continuation)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Fix line breaks within sentences
    text = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', text)

    # Remove page numbers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone numbers on lines
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Page numbers

    # Remove common header/footer patterns
    text = re.sub(r'^\s*Page\s+\d+\s*(of\s+\d+)?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double

    return text.strip()


def prepare_for_embedding(text, is_query=False, use_prefix=True):
    """
    Prepare text for E5 embedding model with instruction prefixes.

    E5 models are trained with specific prefixes:
    - "query: " for search queries
    - "passage: " for documents/passages being indexed

    Args:
        text: Text to prepare
        is_query: True if this is a search query, False for passages
        use_prefix: Whether to add prefix (can be disabled via config)

    Returns:
        Text with appropriate prefix (or unchanged if use_prefix=False)
    """
    if not use_prefix:
        return text

    prefix = "query: " if is_query else "passage: "
    return prefix + text


@lru_cache(maxsize=10000)
def cached_tokenize(text, enable_stemming=True, enable_stopwords=True):
    """
    Cached version of tokenize_for_bm25 for frequently queried terms.

    Note: This uses a tuple of tokens as return for hashability.
    Convert to list if needed: list(cached_tokenize(...))
    """
    return tuple(tokenize_for_bm25(text, enable_stemming, enable_stopwords))

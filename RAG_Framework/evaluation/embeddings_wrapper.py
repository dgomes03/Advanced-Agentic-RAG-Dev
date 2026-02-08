"""Custom RAGAS embeddings wrapper using the existing SentenceTransformer model."""

import asyncio
import typing as t

from ragas.embeddings import BaseRagasEmbedding

from RAG_Framework.core.text_processing import prepare_for_embedding


class E5RagasEmbeddings(BaseRagasEmbedding):
    """Wraps an already-loaded SentenceTransformer (E5) model for RAGAS 0.4+ evaluation."""

    def __init__(self, embedding_model):
        self._model = embedding_model

    def embed_text(self, text: str, **kwargs) -> t.List[float]:
        prefixed = prepare_for_embedding(text, is_query=True, use_prefix=True)
        return self._model.encode(prefixed).tolist()

    async def aembed_text(self, text: str, **kwargs) -> t.List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)

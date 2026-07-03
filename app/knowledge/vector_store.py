from .base import Document
from .embedder import OllamaEmbedder


class VectorStore:
    def __init__(self):
        self._documents: list[Document] = []
        self._embedder = OllamaEmbedder()

    def add(self, doc: Document):
        self._documents.append(doc)

    def add_many(self, docs: list[Document]):
        self._documents.extend(docs)

    def remove(self, doc_id: str):
        self._documents = [d for d in self._documents if d.id != doc_id]

    def clear(self):
        self._documents = []

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        if not self._documents:
            return []
        query_vec = self._embedder.embed(query)
        doc_vecs = self._embedder.embed_batch([d.text for d in self._documents])
        scored = []
        for i, doc in enumerate(self._documents):
            if i < len(doc_vecs):
                score = self._embedder.cosine_similarity(query_vec, doc_vecs[i])
                doc.score = score
                scored.append(doc)
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]

    @property
    def count(self) -> int:
        return len(self._documents)

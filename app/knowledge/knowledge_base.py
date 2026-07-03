import os
from pathlib import Path
from .base import Document
from .vector_store import VectorStore
from .retriever import Retriever

DOCS_DIR = Path(__file__).parent / "docs"


def _load_docs() -> list[Document]:
    """Load and chunk all markdown files from the docs/ directory."""
    docs = []
    if not DOCS_DIR.exists():
        return docs

    for md_file in sorted(DOCS_DIR.glob("*.md")):
        source = md_file.stem  # e.g. "about_zyncjobs"
        text = md_file.read_text(encoding="utf-8").strip()

        # Split into chunks by H2 headings (## ) so each chunk is focused
        chunks = []
        current_heading = ""
        current_lines = []

        for line in text.splitlines():
            if line.startswith("## "):
                if current_lines:
                    chunks.append((current_heading, "\n".join(current_lines).strip()))
                current_heading = line[3:].strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            chunks.append((current_heading, "\n".join(current_lines).strip()))

        # If no H2 headings found, treat whole file as one chunk
        if not chunks:
            chunks = [("", text)]

        for i, (heading, chunk_text) in enumerate(chunks):
            if not chunk_text.strip():
                continue
            doc_id = f"{source}_{i}" if heading else source
            docs.append(Document(
                id=doc_id,
                text=chunk_text,
                source=source,
                metadata={"file": md_file.name, "heading": heading},
            ))

    return docs


class KnowledgeBase:
    def __init__(self):
        self.store = VectorStore()
        self.retriever = Retriever(self.store)
        self._load()

    def _load(self):
        docs = _load_docs()
        if docs:
            self.store.add_many(docs)

    def add_document(self, doc: Document):
        self.store.add(doc)

    def add_documents(self, docs: list[Document]):
        self.store.add_many(docs)

    def remove_document(self, doc_id: str):
        self.store.remove(doc_id)

    def query(self, query: str, top_k: int = 3) -> list[Document]:
        return self.retriever.retrieve(query, top_k=top_k, min_score=0.05)

    def build_context(self, query: str, max_chars: int = 3000) -> str:
        docs = self.query(query, top_k=4)
        if not docs:
            return ""
        parts = []
        used = 0
        for doc in docs:
            chunk = f"[{doc.metadata.get('heading') or doc.source}]\n{doc.text}"
            if used + len(chunk) > max_chars:
                remaining = max_chars - used
                if remaining > 100:
                    parts.append(chunk[:remaining])
                break
            parts.append(chunk)
            used += len(chunk)
        return "\n\n---\n\n".join(parts)

    @property
    def document_count(self) -> int:
        return self.store.count


knowledge_base = KnowledgeBase()

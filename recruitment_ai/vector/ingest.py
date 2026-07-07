import uuid
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from recruitment_ai.vector.embedder import OllamaEmbedder
from recruitment_ai.vector.base import Document
import logging
logger = logging.getLogger(__name__)


VECTOR_DIR = Path(__file__).resolve().parent / "vector_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class ChromaIngester:
    def __init__(self, collection_name: str = "zyncjobs"):
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = OllamaEmbedder()

    def chunk_text(self, text: str, source: str = "") -> list[tuple[str, dict]]:
        chunks: list[tuple[str, dict]] = []
        lines = text.split("\n")
        current = []
        current_len = 0
        section_heading = ""
        for line in lines:
            if line.startswith("# "):
                section_heading = line[2:].strip()
            if line.startswith("## "):
                if current:
                    chunk_text = "\n".join(current).strip()
                    if chunk_text:
                        chunks.append((chunk_text, {"source": source, "section": section_heading}))
                    current = [line]
                    current_len = len(line)
                else:
                    current = [line]
                    current_len = len(line)
                continue
            current.append(line)
            current_len += len(line) + 1
            if current_len >= CHUNK_SIZE:
                chunk_text = "\n".join(current).strip()
                if chunk_text:
                    chunks.append((chunk_text, {"source": source, "section": section_heading}))
                overlap_lines = []
                overlap_len = 0
                for l in reversed(current):
                    overlap_lines.insert(0, l)
                    overlap_len += len(l) + 1
                    if overlap_len >= CHUNK_OVERLAP:
                        break
                current = overlap_lines
                current_len = overlap_len
        if current:
            chunk_text = "\n".join(current).strip()
            if chunk_text:
                chunks.append((chunk_text, {"source": source, "section": section_heading}))
        return chunks

    def ingest_page(self, text: str, url: str, title: str, category: str) -> int:
        chunks = self.chunk_text(text, source=url)
        if not chunks:
            logger.warn(f"Ingest | No chunks for {url}")
            return 0

        ids: list[str] = []
        metadatas: list[dict] = []
        documents: list[str] = []

        for chunk_text, extra in chunks:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            metadatas.append({
                "url": url,
                "title": title,
                "category": category,
                "section": extra.get("section", ""),
            })
            documents.append(chunk_text)

        embeddings = self.embedder.embed_batch(documents)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        logger.info(f"Ingest | {len(chunks)} chunks from {url} ({title})")
        return len(chunks)

    def ingest_pages(self, pages: list) -> int:
        total = 0
        for page in pages:
            total += self.ingest_page(
                text=page.markdown,
                url=page.relative_path,
                title=page.title,
                category=page.category,
            )
        logger.info(f"Ingest | Total {total} chunks ingested")
        return total

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        query_embedding = self.embedder.embed(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        docs: list[Document] = []
        if not results["ids"] or not results["ids"][0]:
            return docs
        for i in range(len(results["ids"][0])):
            doc = Document(
                id=results["ids"][0][i],
                text=results["documents"][0][i] if results["documents"] else "",
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                score=results["distances"][0][i] if results.get("distances") else 0.0,
            )
            docs.append(doc)
        return docs

    @property
    def count(self) -> int:
        return self.collection.count()


ingester = ChromaIngester()

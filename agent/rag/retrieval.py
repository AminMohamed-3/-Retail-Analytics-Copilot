"""RAG retrieval system with TF-IDF search and chunking."""
import re
from typing import List
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Chunk:
    """Document chunk with metadata."""
    def __init__(self, chunk_id: str, content: str, source: str, score: float = 0.0):
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.score = score

    def __repr__(self):
        return f"Chunk({self.chunk_id}, score={self.score:.3f})"


class DocumentRetriever:
    """TF-IDF based retriever for document chunks."""
    
    def __init__(self, docs_dir: str = "docs", top_k: int = 5):
        self.docs_dir = Path(docs_dir)
        self.top_k = top_k
        self.chunks: List[Chunk] = []
        self.vectorizer = None
        self.chunk_vectors = None
        
    def _chunk_document(self, content: str, filename: str) -> List[Chunk]:
        """Split document into paragraph-level chunks."""
        chunks = []
        paragraphs = re.split(r'\n\n+|\n##+', content)
        
        chunk_idx = 0
        for para in paragraphs:
            para = para.strip()
            if len(para) < 10:
                continue
            para = re.sub(r'^#+\s*', '', para).strip()
            if para:
                chunk_id = f"{Path(filename).stem}::chunk{chunk_idx}"
                chunks.append(Chunk(chunk_id, para, filename))
                chunk_idx += 1
        return chunks
    
    def load_documents(self):
        """Load all markdown files from docs directory and chunk them."""
        self.chunks = []
        
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")
        
        for md_file in self.docs_dir.glob("*.md"):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                file_chunks = self._chunk_document(content, md_file.name)
                self.chunks.extend(file_chunks)
        
        if not self.chunks:
            raise ValueError(f"No chunks found in {self.docs_dir}")
        
        texts = [chunk.content for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.chunk_vectors = self.vectorizer.fit_transform(texts)
    
    def retrieve(self, query: str, top_k: int = None) -> List[Chunk]:
        """Retrieve top-k chunks for a query."""
        if top_k is None:
            top_k = self.top_k
        
        if not self.chunks or self.vectorizer is None:
            raise ValueError("Documents not loaded. Call load_documents() first.")
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk.score = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def get_all_chunks(self) -> List[Chunk]:
        """Get all loaded chunks."""
        return self.chunks


if __name__ == "__main__":
    retriever = DocumentRetriever()
    retriever.load_documents()
    
    print(f"Loaded {len(retriever.chunks)} chunks")
    print("\nTesting retrieval:")
    results = retriever.retrieve("Summer Beverages 1997 dates", top_k=3)
    for chunk in results:
        print(f"\n{chunk.chunk_id} (score: {chunk.score:.3f})")
        print(f"Content: {chunk.content[:100]}...")

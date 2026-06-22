import os
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.backend = "faiss"
        self._faiss = self._load_faiss()
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def _load_faiss(self):
        try:
            import faiss
            return faiss
        except Exception as e:
            self.backend = "numpy"
            print(f"[WARN] Faiss is unavailable, using NumPy vector store instead: {e}")
            return None

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page"),
                "sheet": chunk.metadata.get("sheet"),
            }
            for chunk in chunks
        ]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        embeddings = np.asarray(embeddings).astype("float32")
        dim = embeddings.shape[1]
        if self.backend == "faiss":
            if self.index is None:
                self.index = self._faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
        else:
            if self.index is None:
                self.index = embeddings
            else:
                self.index = np.vstack([self.index, embeddings])

        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to {self.backend} index.")

    def save(self):
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        backend_path = os.path.join(self.persist_dir, "backend.txt")

        if self.backend == "faiss":
            index_path = os.path.join(self.persist_dir, "faiss.index")
            self._faiss.write_index(self.index, index_path)
        else:
            index_path = os.path.join(self.persist_dir, "numpy_index.npy")
            np.save(index_path, self.index)

        with open(backend_path, "w", encoding="utf-8") as f:
            f.write(self.backend)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved {self.backend} index and metadata to {self.persist_dir}")

    def load(self):
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        backend_path = os.path.join(self.persist_dir, "backend.txt")

        if os.path.exists(backend_path):
            with open(backend_path, "r", encoding="utf-8") as f:
                self.backend = f.read().strip() or self.backend

        if self.backend == "faiss":
            if self._faiss is None:
                raise RuntimeError("This vector store was saved with Faiss, but Faiss is unavailable.")
            index_path = os.path.join(self.persist_dir, "faiss.index")
            self.index = self._faiss.read_index(index_path)
        else:
            index_path = os.path.join(self.persist_dir, "numpy_index.npy")
            self.index = np.load(index_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded {self.backend} index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = np.asarray(query_embedding).astype("float32")

        if self.backend == "faiss":
            D, I = self.index.search(query_embedding, top_k)
        else:
            distances = np.sum((self.index[None, :, :] - query_embedding[:, None, :]) ** 2, axis=2)
            I = np.argsort(distances, axis=1)[:, :top_k]
            D = np.take_along_axis(distances, I, axis=1)

        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# # Example usage
# if __name__ == "__main__":
#     from data_loader import load_all_documents
#     docs = load_all_documents("data")
#     store = FaissVectorStore("faiss_store")
#     store.build_from_documents(docs)
#     store.load()
#     print(store.query("What is attention mechanism?", top_k=3))

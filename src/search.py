import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemini-2.5-flash"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        backend_path = os.path.join(persist_dir, "backend.txt")
        faiss_path = os.path.join(persist_dir, "faiss.index")
        numpy_path = os.path.join(persist_dir, "numpy_index.npy")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(meta_path) and (os.path.exists(faiss_path) or os.path.exists(numpy_path) or os.path.exists(backend_path))):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(api_key=google_api_key, model=llm_model)
        print(f"[INFO] Google LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Answer the query using only the provided context.

Query:
{query}

Context:
{context}

Return the answer in this exact format:
Answer:
Write 2-4 clear sentences.

Key points:
- Bullet 1
- Bullet 2
- Bullet 3

Limitations:
Mention if the retrieved context seems incomplete or only covers part of the topic.
"""
        response = self.llm.invoke([prompt])
        return self.format_response(query, response.content, results)

    def format_response(self, query: str, answer: str, results: list[dict]) -> str:
        source_lines = []
        evidence_lines = []
        seen_sources = set()

        for rank, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            source = metadata.get("source", "unknown")
            page = metadata.get("page")
            sheet = metadata.get("sheet")
            source_label = os.path.basename(source) if source != "unknown" else source

            details = []
            if page is not None:
                details.append(f"page {page + 1}")
            if sheet:
                details.append(f"sheet {sheet}")

            display_source = source_label
            if details:
                display_source = f"{source_label} ({', '.join(details)})"

            if display_source not in seen_sources:
                source_lines.append(f"{len(source_lines) + 1}. {display_source}")
                seen_sources.add(display_source)

            text = " ".join((metadata.get("text") or "").split())
            if len(text) > 240:
                text = text[:237] + "..."
            evidence_lines.append(f"{rank}. distance={result['distance']:.4f} | {text}")

        return f"""
================ RAG Answer ================

Question:
{query}

{answer.strip()}

Sources:
{chr(10).join(source_lines) if source_lines else "No sources available."}

Retrieved evidence:
{chr(10).join(evidence_lines) if evidence_lines else "No evidence available."}

============================================
""".strip()

# # Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "What is attention mechanism?"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)

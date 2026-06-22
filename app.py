from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

## Example usage

if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What are the main benchmarks used in vision-language navigation?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print(summary)

    # Initialize vector store and search (commented out for now)
    # vector_store = FaissVectorStore()
    # vector_store.add_documents(docs)
    # search = RAGSearch(vector_store)

    # Example query (commented out for now)
    # query = "What is the capital of France?"
    # results = search.search(query)
    # print(f"Search results: {results}")


import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from googlesearch import search  # Requires: pip install googlesearch-python
import ollama  # Requires: pip install ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory RAG storage
rag_store = {}

def index_text(query, context):
    """Store the context in memory for RAG retrieval"""
    if context:
        embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
        rag_store[query] = (embedding, context)

def search_and_index_google(query, num_results=5):
    """Perform a Google search and store the context in RAG"""
    search_results = search(query, num_results=num_results)
    context = "\n".join(search_results) if search_results else None
    index_text(query, context)

def retrieve_answer(query):
    """Retrieve the most relevant context from the in-memory RAG and use LLM to generate an answer"""
    if not rag_store:
        return "No relevant data available."
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
    best_match, best_score = None, float("inf")
    for stored_query, (embedding, context) in rag_store.items():
        score = np.linalg.norm(query_embedding - embedding)
        if score < best_score:
            best_score, best_match = score, context
    if best_match:
        response = ollama.generate(model="gemma3:1b", prompt=f"Based on the following context, answer the question: {query}\n\nContext:\n{best_match}")
        return response.get("response")
    return "No relevant answer found."

def main():
    search_query = input("Enter your search query: ")
    search_and_index_google(search_query)
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = retrieve_answer(query)
        if answer:
            print(answer)

if __name__ == "__main__":
    main()

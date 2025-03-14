import requests
from bs4 import BeautifulSoup
import urllib.parse
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import subprocess
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index setup
embedding_dim = 384  # Dimension of MiniLM embeddings
index = faiss.IndexFlatL2(embedding_dim)
text_data = []


def is_valid_fqdn(url):
    parsed_url = urllib.parse.urlparse(url)
    return bool(parsed_url.netloc) and bool(parsed_url.scheme)


def scrape_text_from_url(url):
    """Scrape and clean text from a given URL"""
    try:
        logging.info(f"Scraping: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        return soup.get_text(separator='\n', strip=True)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping {url}: {e}")
        return None


def index_text(text):
    """Convert text to vector embeddings and store in FAISS"""
    global index, text_data
    if not text:
        return
    
    text_chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    
    index.add(embeddings)
    text_data.extend(text_chunks)
    logging.info(f"Indexed {len(text_chunks)} chunks")


def retrieve_relevant_text(query, top_k=3):
    """Retrieve most relevant text snippets based on the query"""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [text_data[i] for i in indices[0] if i < len(text_data)]
    return results


def answer_question(query):
    """Use Ollama to generate answers based on retrieved text"""
    relevant_text = retrieve_relevant_text(query)
    context = '\n'.join(relevant_text)
    if not context:
        return "No relevant information found."
    
    prompt = f"Answer based on the documentation:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = ollama.chat(model="gemma3:1b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']


def execute_code(code):
    """Execute Python code securely"""
    try:
        logging.info("Executing user-requested code.")
        result = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=5)
        return result.stdout or result.stderr
    except Exception as e:
        return str(e)


def main():
    base_url = input("Enter the documentation URL to index: ")
    if not is_valid_fqdn(base_url):
        logging.error("Invalid URL provided.")
        return
    
    text = scrape_text_from_url(base_url)
    if text:
        index_text(text)
    else:
        logging.error("Failed to scrape content.")
        return
    
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        if query.startswith("run: "):
            code = query[5:]
            print(execute_code(code))
        else:
            print(answer_question(query))


if __name__ == "__main__":
    main()

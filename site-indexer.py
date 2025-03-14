import requests
from bs4 import BeautifulSoup
import urllib.parse
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from googlesearch import search  # Requires: pip install googlesearch-python

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index setup
embedding_dim = 384  # Dimension of MiniLM embeddings
index = faiss.IndexFlatL2(embedding_dim)
text_data = []

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

def search_and_index_google(query, num_results=5):
    """Perform a Google search and index the text from top results"""
    logging.info(f"Searching Google for: {query}")
    
    search_results = search(query, num=num_results, stop=num_results)
    
    for url in search_results:
        logging.info(f"Processing: {url}")
        text = scrape_text_from_url(url)
        if text:
            index_text(text)

def main():
    search_query = input("Enter your search query: ")
    search_and_index_google(search_query)

    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        else:
            print("Retrieval-based answering not implemented yet.")  # Extend with a retrieval function

if __name__ == "__main__":
    main()

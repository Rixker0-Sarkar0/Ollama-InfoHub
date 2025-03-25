import os
import logging
import faiss
import numpy as np
import fitz  # PyMuPDF for PDF processing
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index setup
embedding_dim = 384  # MiniLM embedding dimension
index = faiss.IndexFlatL2(embedding_dim)
text_data = []

def get_pdf_files(pdf_dir):
    """Retrieve all PDFs in the specified directory."""
    if not os.path.exists(pdf_dir):
        logging.error(f"Directory not found: {pdf_dir}")
        return []
    return [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF with a progress bar per page."""
    doc = fitz.open(pdf_path)
    text = []

    with tqdm(total=len(doc), desc=f"Extracting {os.path.basename(pdf_path)}", unit="page") as pbar:
        for page in doc:
            text.append(page.get_text("text"))
            pbar.update(1)

    return "\n".join(text)

def index_pdfs(pdf_dir):
    """Extract and index text from all PDFs in the directory."""
    global index, text_data

    pdf_files = get_pdf_files(pdf_dir)
    if not pdf_files:
        logging.warning("No PDFs found to index.")
        return
    
    logging.info(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_path in pdf_files:
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text.strip():
            logging.warning(f"No text found in {os.path.basename(pdf_path)}")
            continue

        text_chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
        embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)

        index.add(embeddings)
        text_data.extend(text_chunks)
        logging.info(f"Indexed {len(text_chunks)} chunks from {os.path.basename(pdf_path)}")

def retrieve_relevant_text(query, top_k=3):
    """Retrieve the most relevant text snippets based on the query."""
    if not text_data:
        return []
    
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [text_data[i] for i in indices[0] if i < len(text_data)]
    return results

def answer_question(query):
    """Determine query relevance and respond accordingly."""
    relevant_text = retrieve_relevant_text(query)

    if relevant_text:
        context = '\n'.join(relevant_text)
        prompt = f"Answer based on the provided documentation:\n{context}\n\nQuestion: {query}\nAnswer:"
    else:
        logging.info("No relevant content found. Switching to general chat mode.")
        prompt = f"{query}"

    response = ollama.chat(model="gemma3:1b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

def main():
    pdf_dir = os.getcwd()  # Use current working directory for PDFs
    index_pdfs(pdf_dir)
    
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        print(answer_question(query))

if __name__ == "__main__":
    main()
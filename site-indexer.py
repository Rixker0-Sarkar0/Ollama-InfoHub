import requests
 from bs4 import BeautifulSoup
 import subprocess
 import os
 import urllib.parse
 import logging
 from collections import deque
 from tqdm import tqdm  # Progress bar for tasks
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
 
 def direct_fqdn_fetch(url):
     if not is_valid_fqdn(url):
         logging.error("Invalid FQDN URL provided.")
         return None
     return [url]
 
 def scrape_text_from_url(url):
     """Scrape and clean text from a given URL"""
     try:
         logging.info(f"Scraping content from: {url}")
         logging.info(f"Scraping: {url}")
         response = requests.get(url, timeout=10)
         response.raise_for_status()
         soup = BeautifulSoup(response.content, 'html.parser')
         
         for script_or_style in soup(["script", "style", "nav", "footer", "header", "aside"]):
             script_or_style.decompose()
         
         text = soup.get_text(separator='\n', strip=True)
         return text
 
         for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
             tag.decompose()
 
         return soup.get_text(separator='\n', strip=True)
     except requests.exceptions.RequestException as e:
         logging.error(f"Error scraping URL {url}: {e}")
         logging.error(f"Error scraping {url}: {e}")
         return None
 
 def brute_force_crawl(domain, start_url, max_pages=20):
     visited = set()
     queue = deque([start_url])
     links = set()
 
     with tqdm(total=max_pages, desc="Crawling Pages", unit="page") as pbar:
         while queue and len(links) < max_pages:
             url = queue.popleft()
             if url in visited:
                 continue
             visited.add(url)
             
             try:
                 response = requests.get(url, timeout=10)
                 response.raise_for_status()
                 soup = BeautifulSoup(response.content, 'html.parser')
                 
                 for link in soup.find_all('a', href=True):
                     href = link['href']
                     full_url = urllib.parse.urljoin(url, href)
                     if domain in full_url and is_valid_fqdn(full_url) and full_url not in visited:
                         links.add(full_url)
                         queue.append(full_url)
                         pbar.update(1)
                         if len(links) >= max_pages:
                             break
             except requests.exceptions.RequestException:
                 continue
     
     logging.info(f"Total internal links found: {len(links)}")
     return list(links)
 
 def chunk_text(text, chunk_size=2000):
     chunks = []
     start_index = 0
     while start_index < len(text):
         end_index = start_index + chunk_size
         if end_index < len(text):
             sentence_end = max(text.rfind(punct, start_index, end_index) for punct in ['.', '!', '?'])
             if sentence_end > start_index:
                 end_index = sentence_end + 1
         chunks.append(text[start_index:end_index])
         start_index = end_index
     return chunks
 
 def summarize_chunk_with_llama(chunk, model_name="gemma3:1b"):
     try:
         os.environ["PYTHONIOENCODING"] = "utf-8"
         logging.info("Starting summarization for a chunk.")
         process = subprocess.Popen(
             ["ollama", "run", model_name],
             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
             text=True, encoding="utf-8"
         )
         try:
             stdout, stderr = process.communicate(
                 input=f"Summarize the following software documentation:\n{chunk}\nSummary:",
                   # Set timeout to prevent hanging
             )
         except subprocess.TimeoutExpired:
             process.kill()
             logging.error("Summarization timed out.")
             return None
         logging.info("Summarization completed for a chunk.")
 
         if process.returncode != 0 or "Error:" in stdout:
             logging.error(f"Error summarizing chunk: {stderr}")
             return None
         return stdout.strip()
     except Exception as e:
         logging.error(f"Summarization error: {e}")
         return None
 
 def main():
     search_query = input("Enter software documentation FQDN URL: ")
     base_urls = direct_fqdn_fetch(search_query)
     if not base_urls:
         logging.error("No valid FQDNs found.")
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
 
     all_summaries = []
     for base_url in base_urls:
         logging.info(f"Processing domain: {base_url}")
         domain = urllib.parse.urlparse(base_url).netloc
         internal_links = brute_force_crawl(domain, base_url)
         internal_links.insert(0, base_url)
 
         logging.info(f"Internal links found: {len(internal_links)}")
 def retrieve_relevant_text(query, top_k=3):
     """Retrieve most relevant text snippets based on the query"""
     query_embedding = embedding_model.encode([query], convert_to_numpy=True)
     distances, indices = index.search(query_embedding, top_k)
     results = [text_data[i] for i in indices[0] if i < len(text_data)]
     return results
 
         for url in tqdm(internal_links, desc="Processing Pages", unit="page"):
             scraped_text = scrape_text_from_url(url)
             if not scraped_text:
                 logging.warning(f"Failed to scrape content from {url}")
                 continue
 
             logging.info(f"Scraped text length from {url}: {len(scraped_text)} characters")
             text_chunks = chunk_text(scraped_text)
             logging.info(f"Generated {len(text_chunks)} chunks from {url}")
 def answer_question(query):
     """Use Ollama to generate answers based on retrieved text"""
     relevant_text = retrieve_relevant_text(query)
     context = '\n'.join(relevant_text)
     if not context:
         return "No relevant information found."
     
     prompt = f"Answer based on the documentation:\n{context}\n\nQuestion: {query}\nAnswer:"
     response = ollama.chat(model="gemma3:1b", messages=[{"role": "user", "content": prompt}])
     return response['message']['content']
 
             chunk_summaries = [summarize_chunk_with_llama(chunk) for chunk in tqdm(text_chunks, desc="Summarizing", unit="chunk")]
             chunk_summaries = list(filter(None, chunk_summaries))
 
             if not chunk_summaries:
                 logging.warning(f"No summaries generated for {url}")
                 continue
 def execute_code(code):
     """Execute Python code securely"""
     try:
         logging.info("Executing user-requested code.")
         result = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=5)
         return result.stdout or result.stderr
     except Exception as e:
         return str(e)
 
             combined_summary = "\n\n".join(chunk_summaries)
             all_summaries.append(f"Summary from {url}:\n{combined_summary}")
 
     if all_summaries:
         final_summary = "\n\n---\n\n".join(all_summaries)
         logging.info("\nFinal Documentation Summary:\n")
         print(final_summary)
 def main():
     base_url = input("Enter the documentation URL to index: ")
     if not is_valid_fqdn(base_url):
         logging.error("Invalid URL provided.")
         return
     
     text = scrape_text_from_url(base_url)
     if text:
         index_text(text)
     else:
         logging.error("No documentation summaries generated.")
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

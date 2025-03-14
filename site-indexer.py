import requests
from bs4 import BeautifulSoup
import subprocess
import os
import urllib.parse
import logging
from collections import deque
from tqdm import tqdm  # Progress bar for tasks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_valid_fqdn(url):
    parsed_url = urllib.parse.urlparse(url)
    return bool(parsed_url.netloc) and bool(parsed_url.scheme)

def direct_fqdn_fetch(url):
    if not is_valid_fqdn(url):
        logging.error("Invalid FQDN URL provided.")
        return None
    return [url]

def scrape_text_from_url(url):
    try:
        logging.info(f"Scraping content from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script_or_style in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script_or_style.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        return text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping URL {url}: {e}")
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
        return

    all_summaries = []
    for base_url in base_urls:
        logging.info(f"Processing domain: {base_url}")
        domain = urllib.parse.urlparse(base_url).netloc
        internal_links = brute_force_crawl(domain, base_url)
        internal_links.insert(0, base_url)

        logging.info(f"Internal links found: {len(internal_links)}")

        for url in tqdm(internal_links, desc="Processing Pages", unit="page"):
            scraped_text = scrape_text_from_url(url)
            if not scraped_text:
                logging.warning(f"Failed to scrape content from {url}")
                continue

            logging.info(f"Scraped text length from {url}: {len(scraped_text)} characters")
            text_chunks = chunk_text(scraped_text)
            logging.info(f"Generated {len(text_chunks)} chunks from {url}")

            chunk_summaries = [summarize_chunk_with_llama(chunk) for chunk in tqdm(text_chunks, desc="Summarizing", unit="chunk")]
            chunk_summaries = list(filter(None, chunk_summaries))

            if not chunk_summaries:
                logging.warning(f"No summaries generated for {url}")
                continue

            combined_summary = "\n\n".join(chunk_summaries)
            all_summaries.append(f"Summary from {url}:\n{combined_summary}")

    if all_summaries:
        final_summary = "\n\n---\n\n".join(all_summaries)
        logging.info("\nFinal Documentation Summary:\n")
        print(final_summary)
    else:
        logging.error("No documentation summaries generated.")

if __name__ == "__main__":
    main()

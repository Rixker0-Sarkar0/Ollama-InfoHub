import requests
from bs4 import BeautifulSoup
import subprocess
import os
import textwrap
import urllib.parse
from collections import deque

def is_valid_fqdn(url):
    """Checks if the URL is a fully qualified domain name (FQDN)."""
    parsed_url = urllib.parse.urlparse(url)
    return bool(parsed_url.netloc) and bool(parsed_url.scheme)

def direct_fqdn_fetch(url):
    """Directly hits the provided FQDN URL instead of performing a web search."""
    if not is_valid_fqdn(url):
        print("Invalid FQDN URL provided.")
        return None
    return [url]

def scrape_text_from_url(url):
    """Scrapes text content from a given URL, extracting only relevant documentation text."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script_or_style in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script_or_style.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error scraping URL {url}: {e}")
        return None

def brute_force_crawl(domain, start_url, max_pages=20):
    """Uses breadth-first crawling to fetch internal documentation links."""
    visited = set()
    queue = deque([start_url])
    links = set()
    
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
                    if len(links) >= max_pages:
                        break
        except requests.exceptions.RequestException:
            continue
    
    return list(links)

def chunk_text(text, chunk_size=80000):
    """Splits text into smaller chunks while maintaining readability."""
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

def summarize_chunk_with_llama(chunk, model_name="llama3:8b"):
    """Summarizes a text chunk using the Llama model running in the local Ollama environment."""
    try:
        os.environ["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            ["ollama", "run", model_name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8"
        )
        stdout, stderr = process.communicate(input=f"Summarize the following software documentation:\n{chunk}\nSummary:")
        if process.returncode != 0 or "Error:" in stdout:
            print(f"Error summarizing chunk: {stderr}")
            return None
        return stdout.strip()
    except Exception as e:
        print(f"Summarization error: {e}")
        return None

def main():
    search_query = input("Enter software documentation FQDN URL: ")
    base_urls = direct_fqdn_fetch(search_query)
    if not base_urls:
        print("No valid FQDNs found.")
        return
    
    all_summaries = []
    for base_url in base_urls:
        print(f"Processing domain: {base_url}")
        domain = urllib.parse.urlparse(base_url).netloc
        internal_links = brute_force_crawl(domain, base_url)
        internal_links.insert(0, base_url)  # Include the main page
        
        print(f"Internal links found: {len(internal_links)}")
        
        for url in internal_links:
            scraped_text = scrape_text_from_url(url)
            if not scraped_text:
                print(f"Failed to scrape content from {url}")
                continue
            
            print(f"Scraped text length from {url}: {len(scraped_text)} characters")
            text_chunks = chunk_text(scraped_text)
            print(f"Generated {len(text_chunks)} chunks from {url}")
            
            chunk_summaries = [summarize_chunk_with_llama(chunk) for chunk in text_chunks if chunk]
            chunk_summaries = list(filter(None, chunk_summaries))
            
            if not chunk_summaries:
                print(f"No summaries generated for {url}")
                continue
            
            combined_summary = "\n\n".join(chunk_summaries)
            all_summaries.append(f"Summary from {url}:\n{combined_summary}")
    
    if all_summaries:
        final_summary = "\n\n---\n\n".join(all_summaries)
        print("\nFinal Documentation Summary:\n", final_summary)
    else:
        print("No documentation summaries generated.")

if __name__ == "__main__":
    main()

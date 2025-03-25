# Ollama InfoHub

Ollama InfoHub is a powerful web indexing and search tool that integrates web scraping, semantic search, and retrieval-augmented generation (RAG) to provide intelligent and relevant search results.

## Features

- **Web Scraping & Indexing**: Utilizes `BeautifulSoup` and `requests` for efficient data extraction.
- **Semantic Search**: Powered by `sentence-transformers` for enhanced query relevance.
- **Google Search Integration**: Uses `googlesearch-python` for broader web search capabilities.
- **Efficient Data Storage**: Implements `faiss` for optimized text storage and retrieval.
- **Retrieval-Augmented Generation (RAG)**: Combines indexed data with AI-driven responses for better search insights.
- **PDF Content Search**: `pdffinder.py` enables searching within PDFs for relevant content.

## Installation

### Prerequisites

Ensure you have Python installed on your system. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Indexing Websites

To index a website and store embeddings for future searches, run:

```bash
python site-indexer.py <url>
```

### Performing a Semantic Search

To conduct a search using both indexed content and Google Search:

```bash
python web-search.py "your query here"
```

### Searching PDF Documents

To search for relevant content within indexed PDFs:

```bash
python pdffinder.py "your query here"
```

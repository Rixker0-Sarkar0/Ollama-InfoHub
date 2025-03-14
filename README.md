# Ollama InfoHub

## Overview
Ollama InfoHub is a powerful web indexing and search tool that integrates web scraping, semantic search, and retrieval-augmented generation (RAG). It enables efficient text retrieval and intelligent query responses using advanced NLP techniques.

## Features
- Web scraping with `BeautifulSoup` and `requests`.
- Semantic search using `sentence-transformers`.
- Google search integration via `googlesearch-python`.
- High-speed text indexing with `faiss`.
- Retrieval-augmented generation (RAG) for enhanced query responses.

## Installation

### Prerequisites
Ensure you have Python installed. Then, install dependencies:
```bash
pip install -r requirements.txt
```

### Clone the Repository
```bash
git clone Rixker0-Sarkar0/Ollama-InfoHub
cd Ollama-InfoHub-main
```

## Usage

### Index a Website
```bash
python site-indexer.py <url>
```
This will crawl and index the content of the given URL.

### Perform a Semantic Search
```bash
python web-search.py "your query here"
```
This runs a Google search and retrieves relevant indexed content.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.


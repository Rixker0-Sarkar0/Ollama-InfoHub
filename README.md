# Ollama InfoHub

Ollama InfoHub is a web indexing and search tool that integrates web scraping, semantic search, and retrieval-augmented generation (RAG) to provide intelligent search results.

## Features
- Web scraping and indexing using `BeautifulSoup` and `requests`.
- Semantic search powered by `sentence-transformers`.
- Google search integration with `googlesearch-python`.
- Efficient text storage and retrieval using `faiss`.
- Retrieval-augmented generation (RAG) for improved query responses.

## Installation
### Prerequisites
Ensure you have Python installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### Indexing Websites
To index a website and store embeddings for search:
```bash
python site-indexer.py <url>
```

### Performing a Semantic Search
To perform a semantic search using Google and indexed content:
```bash
python web-search.py "your query here"
```

## License
This project is licensed under the MIT License. See `LICENSE` for details.

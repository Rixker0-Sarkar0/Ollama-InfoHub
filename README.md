# Ollama Info Hub

This script integrates Google search, web scraping, and Ollama to function as an offline information repository. It retrieves relevant information from the web and processes it using Ollama's language model.

## Features
- Detects Ollama runtime before execution.
- Performs Google searches to gather relevant links.
- Scrapes web pages for text content.
- Pipes collected information into Ollama for processing.

## Requirements
Ensure you have the following installed:
- Python 3.x
- `googlesearch-python`
- `beautifulsoup4`
- `requests`
- Ollama installed and configured

Install dependencies using:
```bash
pip install googlesearch-python beautifulsoup4 requests
```

## Usage
1. Run the script:
```bash
python script.py
```
2. Enter a search query when prompted.
3. The script will fetch, scrape, and process the information using Ollama.

## Notes
- Ensure Ollama is installed and accessible from the command line.
- The script limits web scraping to the first few paragraphs to avoid excessive data processing.
- Modify the `num_results` parameter in `google_search()` to increase or decrease the number of search results.

## License
This project is open-source and available under the MIT License.


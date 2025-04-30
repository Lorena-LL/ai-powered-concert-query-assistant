# AI-powered Concert Query Assistant

This is an intelligent concert information assistant that allows users to query details about music events from ingested documents or online sources using RAG (Retrieval-Augmented Generation).

---

## Features

- Ingests and stores concert-related information from text files using semantic embeddings and metadata.
- Answers natural language questions based on the ingested data.
- Falls back to online search (using SerpAPI + Google Search) when local data doesn't cover the query.
- Extracts artists and locations using Named Entity Recognition.
- Summarizes long input text before storage.
- Uses LLMs and vector search for smart, context-aware answers.

---

## AI Concepts and Models Used

| Feature                      | Model/Library Used                            |
|-----------------------------|-----------------------------------------------|
| Embeddings (Semantic Search)| `all-MiniLM-L6-v2` via `sentence-transformers`|
| Named Entity Recognition     | `dbmdz/bert-large-cased-finetuned-conll03-english` (HuggingFace) |
| Summarization                | `facebook/bart-large-cnn`                     |
| Question Answering           | `google/flan-t5-base`                         |
| Vector Database              | `ChromaDB`                                    |
| Online Search Fallback       | `SerpAPI` + `requests`                        |

---

## Project Structure

```
Implementation/
├── .env                       # Needs to be created
├── requirements.txt 
├── main.py
├── function_logic.py
├── my_utils.py
├── concert_db/                # ChromaDB persistent storage
├── InputFiles/                # Ingested text files
│   ├── format1nr1.txt
│   ├── ...
└── .venv/                     # Python virtual environment
```

---

##  Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Implementation
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


### 4. Set Up SerpAPI Key

Create a `.env` file in the project root:
```
SERPAPI_KEY=your_serpapi_key_here
```

You can get your SerpAPI key from: https://serpapi.com/

---

## How to Run

### In `main.py` there are examples to run differnt functionalities of the code:
- To run the function of ingesting files:
```python
process_file("./InputFiles/format1nr2.txt") # in the folder InputFiles I left some text files to be ingested by the system 
```
- To see all the data contained in Chroma DB:
```python
get_all_documents_with_metadata()
```
- To answer a query only using RAG:
```python
rag_process_question(sent_querry4)
```
- To answer a query from RAG if possible and if not then search it online
```python
rag_or_online_process_question(sent_querry4)
```
---

## Design Choices

- **Semantic Filtering for Ingestion**: Uses cosine similarity against a concert keyword list to avoid ingesting unrelated content.
- **Metadata Extraction**: Stores artist names and locations with each document using NER to enable context-rich querying.
- **Summarization**: Compresses large documents using `facebook/bart-large-cnn` before embedding them, improving relevance.
- **Fallback Search**: If relevant data is not found in ChromaDB, online search fills the gap using SERP API and summarization.

---

## Future Improvements

- Add a simple web interface (e.g., Streamlit).
- Implement a natural language command interface so users can interact with the system more intuitively. For example, instead of selecting options manually, users could type things like:
  - "Ingest this document" or "Take in this file" while uploading a file
  - "Add the latest tour info from this text"
- More robust ingestion filtering.

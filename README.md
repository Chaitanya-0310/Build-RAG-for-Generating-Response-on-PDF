# Build RAG for Generating Response on PDF

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** system that enables intelligent question-answering over PDF documents. It leverages Google Gemini LLM, LangChain orchestration, and ChromaDB for semantic search to provide accurate, context-grounded responses.

## Architecture 
<img width="1275" height="693" alt="updated_RAG_doc" src="https://github.com/user-attachments/assets/91553a38-57f4-4260-a3ee-bfe7ea3b4727" />


## 🚀 Tech Stack

* **Environment:** Databricks (PySpark)
* **Orchestration:** LangChain
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace Embeddings (all-MiniLM-L6-v2)
* **LLM:** Google Gemini 2.5 Flash
* **Document Loading:** PyMuPDF (pymupdf)
* **API Key Management:** Google Generative AI

## 📋 Implementation Steps

### 1. Environment Setup & Dependencies
The project begins by initializing a Spark session and installing the necessary Python libraries.
* **Key Libraries:** `langchain`, `langchain-community`, `langchain-google-genai`, `langchain-chroma`, `chromadb`, `pymupdf`, `sentence-transformers`.
* **Note:** The Python kernel is restarted after installation in Databricks to ensure all packages are loaded correctly.

### 2. API Key Configuration
Google Gemini API key is securely obtained and set as an environment variable.
* **Method:** Uses `getpass()` for secure input without exposing credentials.
* **Storage:** Environment variable `GOOGLE_API_KEY` is set for Gemini configuration.

### 3. Document Loading
PyMuPDF (pymupdf) is used to extract text from PDF documents.
* **Source:** A sample invoice/bill PDF document loaded from Databricks workspace.
* **Action:** The loader extracts all pages and converts them into document objects with metadata.

### 4. Text Splitting (Chunking)
To optimize retrieval accuracy and handle context limitations, the document is split into overlapping chunks.
* **Method:** `RecursiveCharacterTextSplitter`
* **Parameters:**
    * `chunk_size = 1000`: Maximum characters per chunk.
    * `chunk_overlap = 200`: Overlap ensures context isn't lost between splits.
* **Metadata:** Each chunk is tagged with a chunk ID and source path for traceability.

### 5. Vector Embeddings & Storage
Text chunks are converted into vector representations and stored in a persistent vector database.
* **Embeddings Model:** `all-MiniLM-L6-v2` (lightweight, fast, production-proven).
* **Vector Store:** ChromaDB with persistent storage in `/tmp/chroma/simple_gemini_rag`.
* **Collection Name:** `invoice_rag` for organized storage.

### 6. LLM Initialization
Google Gemini API is configured for inference using LangChain wrapper.
* **Model:** `gemini-2.5-flash`
* **Configuration:**
    * Temperature set to 0 for deterministic responses.
    * Initialized via `ChatGoogleGenerativeAI` from `langchain-google-genai`.

### 7. RAG Pipeline Construction
The core RAG chain is built using LangChain's runnable composition:
1. **Retrieval:** Semantic search over ChromaDB returns top 5 relevant chunks.
2. **Context Formatting:** Retrieved documents are formatted with page metadata.
3. **Prompt Engineering:** Uses a strict prompt template instructing the LLM to answer *only* from provided context.
4. **Generation:** Gemini processes the prompt and generates a grounded response.
5. **Output Parsing:** Returns clean text response via `StrOutputParser`.

### 8. Query Interface
The `ask_with_citations()` function provides a simple interface to ask questions:
* Takes a natural language question as input.
* Returns AI-generated answers grounded in the PDF content.
* Tracks source metadata for answer provenance.

## ⚠️ Requirements
To run this notebook, you will need:
* A Databricks environment with notebook capabilities.
* A valid Google Gemini API key (get it from [Google AI Studio](https://aistudio.google.com)).
* Access to PDF files to load and analyze.
* Python packages as listed in the notebook's pip install commands.

## 🔧 How to Use

1. **Set up API Key:** Run the API key configuration cell and provide your Google Gemini API key when prompted.
2. **Load PDF:** Modify the `PDF_PATH` variable to point to your PDF file.
3. **Create Vector Store:** Run the embedding and ChromaDB cells to index your document.
4. **Ask Questions:** Use the `ask_with_citations()` function to query your PDF:
   ```python
   response = ask_with_citations("What is the total bill amount?")
   ```

## 📊 Example Queries
The notebook demonstrates queries like:
- "What is the total bill amount?"
- "What is the last date to pay the bill and what are the charges for late payment?"
- "Can you break down the total bill charges?"

<img width="1427" height="592" alt="Screenshot 2025-12-15 172555" src="https://github.com/user-attachments/assets/e9c15e4e-4b2f-405b-9b81-ed3773d13cab" />


## 🎯 Key Features
* **Semantic Search:** Uses embeddings for intelligent document retrieval.
* **Grounded Responses:** Gemini LLM only answers based on provided PDF content.
* **Persistent Storage:** ChromaDB saves embeddings for reuse across sessions.
* **Production-Ready:** Lightweight embedding model suitable for scalability.
* **Metadata Tracking:** Each response can be traced back to source chunks and pages.

# RAG-based PDF Q&A System with PySpark & LangChain

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline within a Databricks/PySpark environment. It allows users to upload a PDF document (e.g., a financial market recap) and ask natural language questions about its content using a locally loaded Large Language Model (LLM).

## 🚀 Tech Stack

* **Environment:** Databricks (PySpark)
* **Orchestration:** LangChain
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace Embeddings
* **LLM:** TinyLlama-1.1B-Chat-v1.0 (via Hugging Face Transformers)
* **Document Loading:** PyPDF

## 📋 Implementation Steps

### 1. Environment Setup & Dependencies
The project begins by initializing a Spark session and installing the necessary Python libraries.
* **Key Libraries:** `langchain`, `chromadb`, `sentence-transformers`, `torch`, `transformers`, `accelerate`.
* **Note:** The kernel is typically restarted after installation in Databricks to ensure all packages are loaded correctly.

### 2. Document Loading
We use `PyPDFLoader` to ingest the raw PDF data.
* **Source:** A JP Morgan Weekly Market Recap PDF.
* **Action:** The loader reads the file path and extracts the text content into document objects.

### 3. Text Splitting (Chunking)
To handle the LLM's context window limitations and improve retrieval accuracy, the document is split into smaller chunks.
* **Method:** `RecursiveCharacterTextSplitter`
* **Parameters:**
    * `chunk_size = 1000`: Maximum characters per chunk.
    * `chunk_overlap = 200`: Overlap ensures context isn't lost between breaks.

### 4. Vector Embeddings & Storage
The text chunks are converted into vector representations (embeddings) and stored in a vector database.
* **Embeddings:** `HuggingFaceEmbeddings` (converts text to numerical vectors).
* **Vector Store:** `Chroma` (stores vectors for fast similarity search).

### 5. LLM Initialization
Instead of using an external API (like OpenAI), this project loads an open-source model locally for inference.
* **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
* **Configuration:**
    * Loaded using `AutoModelForCausalLM`.
    * Optimized with `torch.float16` for memory efficiency on GPU.
    * Mapped to the available device (`device_map="auto"`).

### 6. The RAG Pipeline
The core logic is encapsulated in the `get_llm_answer(query)` function:
1.  **Retrieve:** Searches ChromaDB for the top 4 text chunks most similar to the user's query.
2.  **Context Construction:** Joins the retrieved chunks to form a "Reference Context."
3.  **Prompt Engineering:** Wraps the query and context in a structured prompt, instructing the AI to answer *only* based on the provided context.
4.  **Generation:** The `TinyLlama` model generates a response based on the engineered prompt.

## ⚠️ Requirements
To run this notebook, you will need:
* A Databricks environment (or local Jupyter setup with Spark).
* GPU support is recommended for the `transformers` model inference.
* Access to the specific PDF file path configured in the loader.

## 📝 Future Improvements
* Replace `TinyLlama` with a larger model (e.g., Mistral 7B) for higher accuracy if hardware permits.
* Implement a persistent directory for ChromaDB to avoid rebuilding embeddings on every run.
* Add streaming output for a better user experience.

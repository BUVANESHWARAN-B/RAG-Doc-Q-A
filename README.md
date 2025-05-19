# Advanced Document Q&A System (RAG)

A Streamlit application that allows users to upload a PDF document and ask questions about its content using a Retrieval-Augmented Generation (RAG) approach. This system leverages several state-of-the-art techniques for document processing and question answering.

## ✨ Features

* **PDF Document Upload:** Easily upload your PDF files through the Streamlit interface.
* **Accurate PDF Parsing:** Utilizes **LlamaParse** for robust and accurate extraction of text and structure from complex PDFs.
* **Semantic Text Chunking:** Employs **Semantic Chunker** with **HuggingFace `e5-small-v2` embeddings** to divide the document into semantically meaningful chunks based on content similarity.
* **Hybrid Retrieval:** Implements a **hybrid search** strategy combining **BM25** (sparse retrieval) and **FAISS** (dense vector retrieval) to fetch the most relevant document chunks for a given query.
* **Powerful LLM Integration:** Connects with **Google's Gemini 2.0 Flash** model via the LangChain framework to generate concise and accurate answers based on the retrieved context.
* **Chat History:** Maintains a history of questions asked and answers provided.
* **Source Display:** Ability to view the specific document chunks (sources) that were used by the LLM to generate the answer.
* **User-Friendly Interface:** Built with Streamlit for a simple and interactive web application experience.

## 🚀 Technologies Used

* **Framework:** Streamlit
* **PDF Parsing:** LlamaParse
* **Embedding Model:** `intfloat/e5-small-v2` (HuggingFace Embeddings)
* **Text Splitter:** LangChain SemanticChunker
* **Vector Store:** FAISS
* **Sparse Retriever:** BM25Retriever
* **Hybrid Retriever:** LangChain EnsembleRetriever
* **Large Language Model (LLM):** Google Gemini 2.0 Flash (`gemini-2.0-flash`)
* **Orchestration:** LangChain

## 🔧 Setup and Installation
1.  **Save the code:** Save the provided Python code as a file (e.g., `app.py`).
2.  Install all the Dependences using txt file uploaded
    ```bash
    pip install -r requirements.txt
    ```
3.**Set up API Keys:**
    *Create a .env file with the API Keys from Google and Llama_Parse
     In that file:
     
     LLAMA_PARSE_API_KEY = "Llama_parse_key"
     GOOGLE_API_KEY = "google_api_key"    
## ▶️ Running the Application

1.  Ensure your virtual environment is active (if used) and API keys are set up correctly (via environment variables or Streamlit secrets).
2.  Navigate to the directory containing your `app.py` file.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application will open in your web browser.

## 📝 Usage

1.  **Open the App:** Access the application via your web browser (usually `http://localhost:8501`).
2.  **Upload Document:** Use the file uploader in the sidebar to select a PDF file.
3.  **Process Document:** Click the "Process Document" button in the sidebar. Wait for the processing steps (Parsing, Chunking, Setting up Retrieval) to complete. A progress bar and status messages will guide you.
4.  **Ask Questions:** Once processing is complete, a text input field will appear in the main area. Enter your questions about the document content.
5.  **View Answers:** The system will retrieve relevant information and generate an answer using the Gemini model. The answer will appear in the chat history below the input field.
6.  **View Sources:** Click the expander for a Q&A entry in the chat history to see the answer. You can toggle the visibility of the source chunks used to generate that answer.
7.  **Clear Document:** To upload and process a new document, click the "Clear Document" button in the sidebar.

## 💻 Code Structure (Brief)

* **CSS Styling:** Custom CSS for visual appearance.
* **Session State:** Initializes and manages variables needed across user interactions (`processed_file`, `document_content`, `chunks`, `retriever`, `llm`, `chat_history`, `filename`, `show_sources`).
* `get_api_keys()`: Function to retrieve API keys (currently attempts secrets, falls back to hardcoded values - **modify this**).
* `parse_document()`: Handles PDF parsing using LlamaParse.
* `create_semantic_chunks()`: Implements semantic chunking using `e5-small-v2` embeddings and `SemanticChunker`.
* `create_hybrid_retriever()`: Sets up the BM25 and FAISS retrievers and combines them into an `EnsembleRetriever`.
* `setup_llm()`: Configures the Google Gemini 2.0 Flash LLM and creates an `LLMChain` with the defined prompt template.
* `process_query()`: Takes a user query, uses the retriever to find context, and feeds it to the LLM chain to get an answer.
* **Streamlit Layout:** Defines the sidebar (upload, processing) and the main content area (info, query input, chat history).

## 🛠️ Configuration Details

* **Semantic Chunking:**
    * Embedding Model: `intfloat/e5-small-v2`
    * Breakpoint Threshold Type: `percentile`
    * Breakpoint Threshold Amount: `90`
    * Minimum Chunk Size: `50`
    * Buffer Size: `0` (no overlap)
* **Hybrid Retrieval:**
    * Retrievers: BM25 (Sparse), FAISS (Dense)
    * FAISS Embedding Model: `intfloat/e5-small-v2`
    * Weights: BM25 (0.3), FAISS (0.7)
    * Number of Retrieved Documents (`k`): 3 for each retriever before ensembling.
* **LLM:**
    * Model: Google Gemini 2.0 Flash (`gemini-2.0-flash`)
    * Temperature: 0.1
    * Top P: 0.95
    * Max Output Tokens: 1024

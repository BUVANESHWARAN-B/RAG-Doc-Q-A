# Required installations:
# pip install streamlit llama-parse langchain langchain-experimental langchain-google-genai \
#             langchain-community sentence-transformers chromadb python-dotenv \
#             rank_bm25 transformers torch llama-index-core

import os
import streamlit as st
import tempfile
import time
from typing import List, Dict, Any, Optional

# Langchain/LlamaIndex Document Handling
from langchain_core.documents import Document as LangchainDocument
from llama_index.core.schema import Document as LlamaIndexDocument

# LlamaParse
from llama_parse import LlamaParse

# Langchain components
from langchain.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter # Removed
from langchain_experimental.text_splitter import SemanticChunker # Added for Semantic Chunking

# Utilities
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")
VECTOR_STORE_PATH   = "chroma_db_semantic" # Changed path slightly for semantic version
COLLECTION_NAME     = "semantic_rag_collection" # Changed collection name
EMBEDDING_MODEL     = "intfloat/e5-small-v2"

# --- Helper Functions ---

def get_embedding_function():
    """Initializes and returns the HuggingFace embedding function."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True}
    )

# --- PDF Parsing Function (no changes needed here) ---
def get_pdf_text(pdf_file_path: str) -> List[LangchainDocument]:
    """
    Load the PDF using LlamaParse and return a list of Langchain Document objects.
    Handles LlamaParse returning LlamaIndex Documents.
    """
    if not LLAMA_PARSE_API_KEY:
        st.error("LLAMA_PARSE_API_KEY not found in environment variables.")
        return []
    if not os.path.exists(pdf_file_path):
         st.error(f"PDF file not found at path: {pdf_file_path}")
         return []

    parser = LlamaParse(
        api_key=LLAMA_PARSE_API_KEY,
        result_type="markdown",
        verbose=True
    )
    try:
        raw_result = parser.load_data(pdf_file_path)
    except Exception as e:
        st.error(f"Error during PDF parsing with LlamaParse: {e}")
        return []

    docs: list[LangchainDocument] = []

    if isinstance(raw_result, list):
        for chunk in raw_result:
            if isinstance(chunk, LlamaIndexDocument):
                text_content = chunk.text
                if text_content and text_content.strip():
                     docs.append(LangchainDocument(page_content=text_content, metadata=chunk.metadata or {}))
            elif isinstance(chunk, LangchainDocument):
                 if chunk.page_content and chunk.page_content.strip():
                     docs.append(chunk)
            elif isinstance(chunk, str):
                 if chunk.strip():
                     docs.append(LangchainDocument(page_content=chunk))
            else:
                 st.warning(f"LlamaParse returned unexpected chunk type during list iteration: {type(chunk)}")
    elif isinstance(raw_result, LlamaIndexDocument):
         text_content = raw_result.text
         if text_content and text_content.strip():
             docs.append(LangchainDocument(page_content=text_content, metadata=raw_result.metadata or {}))
    else:
         st.error(f"LlamaParse returned completely unexpected result type: {type(raw_result)}")

    if not docs:
        st.warning("LlamaParse processing resulted in zero usable Langchain documents.")

    st.write(f"Successfully parsed PDF into {len(docs)} initial document sections.")
    return docs


# --- UPDATED create_text_chunks function using SemanticChunker ---
def create_text_chunks(doc_list: List[LangchainDocument]) -> List[LangchainDocument]:
    """
    Splits the text content of documents into semantic chunks using SemanticChunker.
    Takes List[LangchainDocument], returns List[LangchainDocument].
    """
    if not doc_list:
        st.warning("Received empty document list for chunking.")
        return []

    st.write("Initializing Semantic Chunker...")
    try:
        # Semantic chunking requires embeddings
        embeddings = get_embedding_function()
        if embeddings is None:
             st.error("Failed to initialize embedding function for Semantic Chunker.")
             return []

        semantic_chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile", # Choose threshold type ('percentile', 'standard_deviation', 'interquartile', 'gradient')
            # Adjust threshold value as needed (lower = more chunks for std dev)
            # Example using percentile:
            # breakpoint_threshold_type="percentile",
            # breakpoint_threshold_amount=95 # Split at the 95th percentile distance difference
        )

        st.write("Applying Semantic Chunking (this might take a bit longer)...")
        # Use split_documents which works directly with Langchain Documents
        # This preserves metadata from the original docs where possible
        split_chunks = semantic_chunker.split_documents(doc_list)

        if not split_chunks:
            st.warning("Semantic chunking resulted in zero chunks. Try adjusting threshold parameters.")
            return []

        st.write(f"Split documents into {len(split_chunks)} semantic text chunks.")
        # Log chunk sizes for analysis (optional)
        # chunk_sizes = [len(chunk.page_content) for chunk in split_chunks]
        # st.write(f"Chunk sizes (characters): Min={min(chunk_sizes)}, Max={max(chunk_sizes)}, Avg={sum(chunk_sizes)/len(chunk_sizes):.0f}")

        return split_chunks

    except Exception as e:
        st.error(f"Failed during semantic chunking: {e}")
        import traceback # Keep for debugging experimental features
        st.error(traceback.format_exc()) # Show detailed traceback
        return []
# --- END of updated function ---


# --- Vector Store Creation (no changes needed) ---
def create_and_persist_vector_store(text_chunks: List[LangchainDocument]) -> Optional[Chroma]:
    """Creates and persists a Chroma vector store from Langchain Documents."""
    if not text_chunks:
        st.error("Cannot create vector store: No text chunks provided.")
        return None
    try:
        embeddings = get_embedding_function()
        vector_store = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_STORE_PATH
        )
        vector_store.persist()
        st.write(f"Vector store persisted to '{VECTOR_STORE_PATH}'.")
        return vector_store
    except Exception as e:
        st.error(f"Failed to create/persist vector store: {e}")
        return None

# --- Hybrid Retriever Creation (no changes needed) ---
def create_hybrid_retriever(vector_store: Chroma, chunks: List[LangchainDocument]) -> Optional[EnsembleRetriever]:
    """Creates a hybrid retriever (BM25 + Vector Store) from Langchain Documents."""
    if not vector_store or not chunks:
         st.error("Cannot create hybrid retriever: Missing vector store or chunks.")
         return None
    try:
        dense_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 3
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )
        st.write("Hybrid retriever created successfully.")
        return ensemble_retriever
    except Exception as e:
        st.error(f"Failed to create hybrid retriever: {e}")
        return None

# --- LLM Setup (no changes needed) ---
def setup_llm_chain() -> Optional[LLMChain]:
    """Sets up the LLM and chain for answering queries."""
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found in environment variables.")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            top_p=0.9,
            convert_system_message_to_human=True
        )
        prompt_template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Provide the answer based *only* on the provided context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Failed to set up LLM chain: {e}")
        return None

# --- User Input Handling (no changes needed) ---
def handle_user_input(user_question: str):
    """Handles user query using the stored retriever and LLM chain."""
    if "retriever" not in st.session_state or st.session_state.retriever is None:
        st.error("Retriever not initialized. Please process a PDF first.")
        return
    if "llm_chain" not in st.session_state or st.session_state.llm_chain is None:
        st.error("LLM Chain not initialized.")
        return

    retriever = st.session_state.retriever
    chain = st.session_state.llm_chain

    try:
        with st.spinner("Retrieving relevant context..."):
            docs: List[LangchainDocument] = retriever.get_relevant_documents(user_question)
        if not docs:
            st.write("Reply: Could not find relevant context in the document for your question.")
            return
        context = "\n\n".join([doc.page_content for doc in docs])
        with st.spinner("Generating answer..."):
            response = chain.invoke({"context": context, "question": user_question})
        answer = response.get('text', 'Could not extract answer from LLM response.')
        st.write("Reply: ", answer)
    except Exception as e:
        st.error(f"Error processing question: {e}")


# --- Main Streamlit App (no changes needed in logic) ---
def main():
    st.set_page_config(page_title="Chat with PDF (Semantic)", layout="wide") # Updated title
    st.header("Chat with PDF using Gemini & Semantic Chunking 💁") # Updated header

    # Initialize session state
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "llm_chain" not in st.session_state:
        st.session_state.llm_chain = setup_llm_chain()

    with st.sidebar:
        st.title("Menu")
        st.write("Upload a PDF and click 'Process' to enable Q&A.")
        pdf_doc = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")

        if st.button("Process PDF", key="process_button") and pdf_doc is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_doc.getvalue())
                tmp_file_path = tmp_file.name

            st.session_state.processed = False
            st.session_state.retriever = None
            tmp_file_to_remove = tmp_file_path

            with st.spinner("Processing PDF... This may take a moment."):
                try:
                    # 1. Parse PDF -> Returns List[LangchainDocument]
                    st.write("Step 1: Parsing PDF...")
                    langchain_docs = get_pdf_text(tmp_file_path)
                    if not langchain_docs:
                        st.error("Stopping: Failed to parse PDF or PDF is empty.")
                        return

                    # 2. Chunk Text (NOW uses SemanticChunker) -> Takes/Returns List[LangchainDocument]
                    st.write("Step 2: Chunking text using Semantic Chunker...")
                    text_chunks = create_text_chunks(langchain_docs) # This function is updated
                    if not text_chunks:
                        st.error("Stopping: Failed to create semantic text chunks.")
                        return

                    # 3. Create/Persist Vector Store -> Takes List[LangchainDocument]
                    st.write("Step 3: Creating vector store...")
                    vector_store = create_and_persist_vector_store(text_chunks)
                    if not vector_store:
                        st.error("Stopping: Failed to create vector store.")
                        return

                    # 4. Create Hybrid Retriever -> Takes VectorStore and List[LangchainDocument]
                    st.write("Step 4: Creating hybrid retriever...")
                    st.session_state.retriever = create_hybrid_retriever(vector_store, text_chunks)
                    if st.session_state.retriever:
                        st.session_state.processed = True
                        st.success("PDF processed successfully! Ready for questions.")
                    else:
                        st.error("Stopping: Failed to create hybrid retriever.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                finally:
                    if 'tmp_file_to_remove' in locals() and os.path.exists(tmp_file_to_remove):
                        os.remove(tmp_file_to_remove)
                        st.write(f"Cleaned up temporary file.")

        if st.session_state.processed:
            st.sidebar.success("PDF is processed and ready.")
        else:
            st.sidebar.info("Please upload and process a PDF.")

    st.write("Ask a question about the processed PDF content.")
    user_question = st.text_input("Your Question:", key="user_question_input", disabled=not st.session_state.processed)

    if user_question:
        if st.session_state.processed and st.session_state.retriever and st.session_state.llm_chain:
            handle_user_input(user_question)
        elif not st.session_state.processed:
            st.warning("Please upload and process a PDF file using the sidebar first.")
        else:
             st.error("System not ready. Ensure PDF was processed and LLM is configured.")

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        st.error("FATAL: GOOGLE_API_KEY environment variable not set.")
    elif not LLAMA_PARSE_API_KEY:
         st.error("FATAL: LLAMA_PARSE_API_KEY environment variable not set.")
    else:
        main()
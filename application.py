import os
import streamlit as st
import tempfile
import time
from typing import List, Dict, Any

# Document processing
import llama_parse
from llama_parse import LlamaParse
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings

# Set page configuration
st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📚",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 2rem;
}
.subheader {
    font-size: 1.5rem;
    color: #4B5563;
    margin-bottom: 1rem;
}
.info-text {
    background-color: #F3F4F6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
}
.stButton>button {
    background-color: #1E3A8A;
    color: white;
    font-weight: bold;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>Document Q&A System</h1>", unsafe_allow_html=True)

# Initialize session state variables
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = False
if 'document_content' not in st.session_state:
    st.session_state.document_content = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_api_keys():
    """Get API keys from Streamlit secrets or environment variables"""
    # For deployment, use Streamlit secrets
    llama_parse_api_key = "llx-wP1kGssDnVVuuWZRkqpw8VeJtqmPbiGWxd4VsoWOaEar9ER8"
    google_api_key = "AIzaSyCmIGe4Yl69P0FqnSgoGDYo8Sd_44fRhsM"
        
    return llama_parse_api_key, google_api_key

def parse_document(file_path, llama_parse_api_key):
    """Parse PDF document using LlamaParse"""
    try:
        parser = LlamaParse(
            api_key=llama_parse_api_key,
            result_type="markdown"
        )
        
        # Parse the document using LlamaParse
        result = parser.load_data(file_path)
        
        # Debug info
        st.write(f"LlamaParse result type: {type(result)}")
        
        # Convert to LangChain Document objects if needed
        documents = []
        
        # Check if result is a list
        if isinstance(result, list):
            for item in result:
                # If already a Document, use it directly
                if hasattr(item, 'page_content'):
                    documents.append(item)
                # If a dictionary, convert to Document
                elif isinstance(item, dict) and 'text' in item:
                    documents.append(Document(page_content=item['text']))
                # If string, convert to Document
                elif isinstance(item, str):
                    documents.append(Document(page_content=item))
        else:
            # If it's a single item
            if hasattr(result, 'page_content'):
                documents.append(result)
            elif isinstance(result, dict) and 'text' in result:
                documents.append(Document(page_content=result['text']))
            elif isinstance(result, str):
                documents.append(Document(page_content=result))
        
        # If no documents were created, extract text manually
        if not documents:
            text = str(result)
            documents = [Document(page_content=text)]
            
        st.write(f"Created {len(documents)} document objects")
        return documents
    
    except Exception as e:
        st.error(f"Error in parse_document: {str(e)}")
        st.write(f"Result type: {type(result) if 'result' in locals() else 'Not available'}")
        raise

def create_semantic_chunks(documents):
    """Create semantic chunks using SemanticChunker"""
    try:
        # Initialize OpenAI embeddings with text-embedding-3-large model
        embeddings = HuggingFaceEmbeddings(
         model_name="intfloat/e5-small-v2",
         model_kwargs={"trust_remote_code": True}
        )
        
        # Create semantic chunker with percentile threshold
        semantic_chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile"  # Using percentile for breakpoints
        )
        
        # Extract document text
        if not documents:
            raise ValueError("No documents provided for chunking")
        
        # Debug info
        st.write(f"Number of documents for chunking: {len(documents)}")
        
        # Extract text content from documents
        text_contents = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                text_contents.append(doc.page_content)
            elif isinstance(doc, dict) and 'text' in doc:
                text_contents.append(doc['text'])
            elif isinstance(doc, str):
                text_contents.append(doc)
                
        if not text_contents:
            raise ValueError("Could not extract text content from documents")
            
        st.write(f"Extracted {len(text_contents)} text segments")
        
        # Join text contents
        full_text = "\n\n".join(text_contents)
        
        # Create chunks based on semantic similarity
        chunks = semantic_chunker.create_documents([full_text])
        st.write(f"Created {len(chunks)} semantic chunks")
        
        return chunks
    
    except Exception as e:
        st.error(f"Error in create_semantic_chunks: {str(e)}")
        raise

def create_hybrid_retriever(chunks):
    """Create a hybrid retriever with BM25 and FAISS"""
    try:
        # Create embeddings with text-embedding-3-large model
        embeddings = HuggingFaceEmbeddings(
          model_name="intfloat/e5-small-v2",
          model_kwargs={"trust_remote_code": True}
        )
        
        # Create FAISS index (dense retriever)
        texts = [doc.page_content for doc in chunks]
        faiss_vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create BM25 retriever (sparse retriever)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 3
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        
        return ensemble_retriever
    
    except Exception as e:
        st.error(f"Error in create_hybrid_retriever: {str(e)}")
        raise

def setup_llm(google_api_key):
    """Set up the LLM and chain for answering queries"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key,
        temperature=0.2,
        top_p=0.9,
    )
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def process_query(query, retriever, llm_chain):
    """Process user query and generate response"""
    # Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(query)
    
    # Join document contents
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Generate response
    
    response = llm_chain.run(context=context, question=query)
    return response, relevant_docs

# Sidebar for document upload
with st.sidebar:
    st.markdown("<h2 class='subheader'>Upload Document</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    # Process the uploaded file
    if uploaded_file is not None:
        st.info("Processing document... This may take a minute...")
        progress_bar = st.progress(0)
        
        # Get API keys
        llama_parse_api_key, openai_api_key, google_api_key = get_api_keys()
        
        if not google_api_key:
            st.error("Missing Google API key. Please set the required API key.")
        else:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name
                
                # Update progress
                progress_bar.progress(20)
                
                try:
                    # Parse document
                    st.info("Parsing document with LlamaParse...")
                    document = parse_document(temp_file_path, llama_parse_api_key)
                    progress_bar.progress(50)
                    
                    # Create semantic chunks using SemanticChunker
                    st.info("Creating semantic chunks...")
                    chunks = create_semantic_chunks(document)
                    progress_bar.progress(70)
                    
                    # Create hybrid retriever
                    st.info("Setting up retrieval system...")
                    retriever = create_hybrid_retriever(chunks)
                    progress_bar.progress(90)
                    
                    # Setup LLM
                    llm_chain = setup_llm(google_api_key)
                    
                    # Store in session state
                    st.session_state.document_content = document
                    st.session_state.chunks = chunks
                    st.session_state.retriever = retriever
                    st.session_state.llm = llm_chain
                    st.session_state.processed_file = True
                    st.session_state.filename = uploaded_file.name
                    
                    progress_bar.progress(100)
                    st.success(f"Successfully processed {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)
            except Exception as e:
                st.error(f"Error handling file: {str(e)}")
    
    # Display document info if processed
    if st.session_state.processed_file:
        st.markdown("<h3>Document Information</h3>", unsafe_allow_html=True)
        st.write(f"Filename: {st.session_state.filename}")
        st.write(f"Number of document segments: {len(st.session_state.document_content)}")
        st.write(f"Number of semantic chunks: {len(st.session_state.chunks)}")
        
        if st.button("Clear Document"):
            st.session_state.processed_file = False
            st.session_state.document_content = None
            st.session_state.chunks = None
            st.session_state.retriever = None
            st.session_state.llm = None
            st.session_state.chat_history = []

# Main content area
if not st.session_state.processed_file:
    st.markdown("<div class='info-text'>Upload a PDF document using the sidebar to get started.</div>", unsafe_allow_html=True)
    
    # Display sample questions and information
    st.markdown("<h2 class='subheader'>About this RAG System</h2>", unsafe_allow_html=True)
    st.markdown("""
    This document Q&A system uses:
    
    - **LlamaParse** for accurate PDF parsing
    - **Semantic chunking** with OpenAI's text-embedding-3-large model using percentile-based similarity
    - **Hybrid search** combining BM25 (sparse) and FAISS (dense) retrieval
    - **Google's Gemini** for generating accurate answers
    
    Upload your document to start querying!
    """)
else:
    # Query input
    st.markdown("<h2 class='subheader'>Ask questions about your document</h2>", unsafe_allow_html=True)
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Generating answer..."):
            start_time = time.time()
            
            # Process query
            response, relevant_docs = process_query(query, st.session_state.retriever, st.session_state.llm)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add to chat history
            st.session_state.chat_history.append({
                "query": query,
                "response": response,
                "docs": relevant_docs,
                "time": processing_time
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("<h2 class='subheader'>Chat History</h2>", unsafe_allow_html=True)
        
        for i, exchange in enumerate(st.session_state.chat_history):
            # Create expandable sections for each Q&A
            with st.expander(f"Q: {exchange['query']}", expanded=(i == len(st.session_state.chat_history) - 1)):
                st.markdown(f"**Answer:**\n{exchange['response']}")
                st.markdown(f"*Processed in {exchange['time']:.2f} seconds*")
                
                # Show sources in a nested expander
            with st.expander("View Sources"):
                for j, doc in enumerate(exchange['docs']):
                    st.markdown(f"**Source {j+1}**")
                    st.markdown(doc.page_content)
                    st.markdown("---")
import os
import streamlit as st
import tempfile
import time
from typing import List, Dict, Any
import pickle 
import llama_parse
from llama_parse import LlamaParse
from langchain_experimental.text_splitter import SemanticChunker
from langchain.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY       = os.getenv("GOOGLE_API_KEY")
LLAMA_PARSE_API_KEY  = os.getenv("LLAMA_PARSE_API_KEY")

from llama_parse import LlamaParse


def get_pdf_text(pdf_file) -> list[Document]:
    """
    Load the PDF and return a list of Document objects.
    LlamaParse.load_data sometimes returns:
      - a single Document
      - a list of Document
      - a list of strings (if result_type="text")
    We canonicalize all of them into List[Document].
    """
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_PARSE_API_KEY"),
        result_type="text"
    )
    raw = parser.load_data(pdf_file)

    # Normalize to list
    docs: list[Document] = []
    if isinstance(raw, Document):
        docs = [raw]
    elif isinstance(raw, list):
        for chunk in raw:
            if isinstance(chunk, Document):
                docs.append(chunk)
            elif isinstance(chunk, str):
                docs.append(Document(page_content=chunk))
            else:
                raise ValueError(f"Unexpected chunk type: {type(chunk)}")
    else:
        raise ValueError(f"Unexpected parser output: {type(raw)}")
    return docs
    

def create_semantic_chunks(doc: List[Document]) -> List[Document]:
    """embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        model_kwargs={"trust_remote_code": True}
    )
    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        buffer_size=1,
        breakpoint_threshold_amount=0.30,
        add_start_index=True
    )


    # now split it into semantically coherent chunks
    chunks = semantic_chunker.split_documents(doc)"""
    full_text = "\n\n".join(d.page_content for d in doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(full_text)
    fine_chunks = [Document(page_content=chunk) for chunk in chunks]
    return fine_chunks
    # sanity check
    if not chunks:
        raise ValueError("No chunks produced—check that your PDF actually contains text!")


def get_vector_store(text_chunks: List[Document]) -> Chroma:   
    # 1) Prepare your embedding function
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        model_kwargs={"trust_remote_code": True}
    )

    # 2) Create an *empty* Chroma store
    vector_store = Chroma(
        persist_directory="C:\\Users\\buvan\\Internship",
        collection_name="chroma_index",
        embedding_function=embeddings
    )

    # 3) Index your chunks
    vector_store.add_documents(text_chunks)

    # 4) Persist it to disk
    vector_store.persist()

    return vector_store

def hybrid_retriever(chunks: List[Document]) -> EnsembleRetriever:  
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        model_kwargs={"trust_remote_code": True}
    )

    # fresh store pointing at your persisted index
    chroma_store = Chroma(
        persist_directory="C:\\Users\\buvan\\Internship",
        collection_name="chroma_index",
        embedding_function=embeddings
    )
    dense_retriever = chroma_store.as_retriever(search_kwargs={"k": 3})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

def setup_llm(GOOGLE_API_KEY) -> LLMChain:
    """Set up the LLM and chain for answering queries"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=GOOGLE_API_KEY,
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


def user_input(user_question: str):
    embeddings = HuggingFaceEmbeddings(
          model_name="intfloat/e5-small-v2",
          model_kwargs={"trust_remote_code": True}
        )
    new_db = Chroma(
        collection_name="chroma_index",
        persist_directory="C:\\Users\\buvan\\Internship",
        embedding_function=embeddings
    )
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.(user_question)

    chain = setup_llm(GOOGLE_API_KEY)

    
    context = "\n\n".join([doc.page_content for doc in docs])
    response = chain({"context": context, "question": user_question},
    return_only_outputs=True
   )


    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",type=["pdf"])
        if st.button("Submit & Process") and pdf_doc: 
            with st.spinner("Processing..."):
                raw_docs = get_pdf_text(pdf_doc)
                text_chunks = create_semantic_chunks(raw_docs)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
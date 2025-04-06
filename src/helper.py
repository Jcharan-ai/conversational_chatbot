from dotenv import load_dotenv
import os
from langchain.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.logger import logger
import streamlit as st

def load_env():
    """Load environment variables from .env file."""
 
    load_dotenv()
    os.environ["HF_API_KEY"] = st.secrets["HF_API_KEY"]
    
    
def create_documents(docs):
    """Create documents from uploaded files."""
    
    loaders = {
        "txt": TextLoader,
        "pdf": PyMuPDFLoader,
        "docx": UnstructuredWordDocumentLoader,
    }

    documents = []
    for doc in docs:
        os.makedirs("./temp", exist_ok=True)
        tempdocs = os.path.join("./temp", doc.name)
        with open(tempdocs, "wb") as f: 
            f.write(doc.getbuffer())
        file_extension = f.name.split(".")[-1]
        if file_extension in loaders:
            loader = loaders[file_extension](f"./temp/" + doc.name)
            documents.extend(loader.load())
            logger.info(f"Loaded {len(documents)} documents from {doc.name}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    return docs

def create_embeddings(docs):
    """Create embeddings for the documents."""
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    return vectorstore
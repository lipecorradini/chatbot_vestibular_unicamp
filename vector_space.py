import csv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

def split_text(file_path, chunk_size=1000, chunk_overlap=50):
    """
    Splits the text from the given file into chunks of specified size with overlap.
    
    Args:
        file_path (str): Path to the input text file.
        chunk_size (int): Number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
    
    Returns:
        List[str]: List of text chunks.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    
    return chunks

def split_tables(file_path):
    """
    Splits the tables text file into chunks where each line is a separate chunk.
    
    Args:
        file_path (str): Path to the tables text file.
    
    Returns:
        List[str]: List of table chunks, each corresponding to a line in the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Strip whitespace and skip empty lines
    chunks = [line.strip() for line in lines if line.strip()]
    
    return chunks

def generate_embeddings_huggingface(documents, model_name="all-MiniLM-L6-v2"):
    """
    Generates embeddings for each document using Hugging Face and adds them to FAISS.
    
    Args:
        documents (List[Document]): List of LangChain Document objects.
        model_name (str): Hugging Face model name for embeddings.
    
    Returns:
        FAISS: FAISS vector store containing embeddings and metadata.
    """
    # Initialize the Hugging Face embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Create FAISS vector store from documents
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store

def add_embeddings_to_faiss(vector_store, documents, model_name="all-MiniLM-L6-v2"):
    """
    Adds new documents to an existing FAISS vector store.
    
    Args:
        vector_store (FAISS): Existing FAISS vector store.
        documents (List[Document]): List of new LangChain Document objects to add.
        model_name (str): Hugging Face model name for embeddings.
    
    Returns:
        FAISS: Updated FAISS vector store with new embeddings.
    """
    # Initialize the Hugging Face embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Add documents to the existing vector store
    vector_store.add_documents(documents, embeddings)
    
    return vector_store

def main():
    # Define file paths
    text_input_file = "./data/text/extracted_text.txt"
    tables_input_file = "./data/text/tables.txt"
    faiss_index_path = "./data/faiss_index"
    
    # Step 1: Split the main text into chunks
    print("Splitting main text into chunks...")
    text_chunks = split_text(text_input_file)
    print(f"Total text chunks created: {len(text_chunks)}")
    
    # Convert text chunks to Document objects with metadata
    text_documents = [Document(page_content=chunk, metadata={"source": "text"}) for chunk in text_chunks]
    
    # Step 2: Split the tables text into chunks (each line as a chunk)
    print("Splitting tables text into chunks...")
    table_chunks = split_tables(tables_input_file)
    print(f"Total table chunks created: {len(table_chunks)}")
    
    # Convert table chunks to Document objects with metadata
    table_documents = [Document(page_content=chunk, metadata={"source": "table"}) for chunk in table_chunks]
    
    # Combine all documents
    all_documents = text_documents + table_documents
    print(f"Total documents to embed: {len(all_documents)}")
    
    # Step 3: Generate embeddings and create FAISS vector store
    print("Generating embeddings with Hugging Face and creating FAISS vector store...")
    vector_store = generate_embeddings_huggingface(all_documents)
    
    # Save the FAISS index for later use
    vector_store.save_local(faiss_index_path)
    print(f"Embeddings generated and FAISS index saved locally at '{faiss_index_path}'.")
    
    print("All chunks have been processed and added to the FAISS index.")

if __name__ == "__main__":
    main()

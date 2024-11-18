from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

def split_text(file_path, chunk_size=1000, chunk_overlap=50):
    """
    Splits the text from the given file into chunks suitable for embedding.

    Args:
        file_path (str): Path to the text file.
        chunk_size (int): Maximum number of characters in each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    
    return chunks

def generate_embeddings_huggingface(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Generates embeddings for each text chunk using Hugging Face and stores them in FAISS.

    Args:
        chunks (List[str]): List of text chunks.
        model_name (str): Hugging Face model name for embeddings.

    Returns:
        FAISS: FAISS vector store containing embeddings and metadata.
    """
    # Initialize the Hugging Face embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Convert chunks to LangChain Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Create FAISS vector store from documents
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store

def main():
    input_file = "extracted_text.txt"
    
    # Step 1: Split the text into chunks
    print("Splitting text into chunks...")
    chunks = split_text(input_file)
    print(f"Total chunks created: {len(chunks)}")
    
    # Step 2: Generate embeddings and create FAISS vector store using Hugging Face
    print("Generating embeddings with Hugging Face and creating FAISS vector store...")
    vector_store = generate_embeddings_huggingface(chunks)
    
    # Save the FAISS index for later use
    vector_store.save_local("faiss_index_huggingface")
    print("Embeddings generated and FAISS index saved locally.")

if __name__ == "__main__":
    main()

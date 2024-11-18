from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_faiss_vector_store(index_path, model_name="all-MiniLM-L6-v2"):
    """
    Loads the FAISS vector store from the specified path.

    Args:
        index_path (str): Path to the saved FAISS index.
        model_name (str): Hugging Face model name used for embeddings.

    Returns:
        FAISS: Loaded FAISS vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Enable dangerous deserialization if you trust the source
    vector_store = FAISS.load_local(
        folder_path=index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vector_store


def retrieve_relevant_chunks(query, vector_store, top_k=5):
    """
    Retrieves the top-K most relevant text chunks for a given query.

    Args:
        query (str): User input query.
        vector_store (FAISS): FAISS vector store containing embeddings.
        top_k (int): Number of top similar chunks to retrieve.

    Returns:
        List[Document]: List of retrieved documents.
    """
    results = vector_store.similarity_search(query, k=top_k)
    return results

if __name__ == "__main__":
    # Load the FAISS vector store
    vector_store = load_faiss_vector_store("./data/faiss_index")
    
    # Example query
    user_query = "Quais conteúdos serão abordados na prova?"
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(user_query, vector_store, top_k=5)
    
    print("Retrieved Chunks:")
    for i, doc in enumerate(relevant_chunks, 1):
        print(f"\nChunk {i}:\n{doc.page_content}")

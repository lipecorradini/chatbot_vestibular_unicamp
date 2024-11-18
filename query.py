# Install necessary packages if not already installed
# pip install langchain transformers faiss-cpu groq

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq
import os

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


def generate_response(retrieved_chunks, user_query, model_name="llama3-groq-70b-8192-tool-use-preview"):
    """
    Generates a response using the Groq API based on retrieved chunks and user query.

    Args:
        retrieved_chunks (List[Document]): Retrieved text chunks.
        user_query (str): User input query.
        model_name (str): Groq model name.

    Returns:
        str: Generated response from the model.
    """
    # Initialize the Groq client
    client = Groq()
    
    # Construct the context from retrieved chunks
    context = "\n\n".join([f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_chunks)])
    
    # Construct the messages for the API
    messages = [
        {
            "role": "system",
            "content": "Considere a conversa, o contexto e a pergunta dada para dar uma resposta. Caso você não saiba uma resposta, fale 'Me desculpe, mas não tenho uma resposta para esta pergunta' em vez de tentar gerar uma resposta imprecisa. Responda a pergunta passo-a-passo."
        },
        {
            "role": "system",
            "content": f"Contexto:\n{context}"
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    # Call the Groq API
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        top_p=0.65,
        stream=True,
        stop=None,
    )

    print('-----------------------------')

    generated_response = ''
    for chunk in completion:
        generated_response += (chunk.choices[0].delta.content or "")

    return generated_response


if __name__ == "__main__":
    # Load the FAISS vector store
    vector_store = load_faiss_vector_store("./data/faiss_index")
    
    # Example query
    user_query = "Como funciona a prova de habilidades específicas?"
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(user_query, vector_store, top_k=5)
    
    # print("Retrieved Chunks:")
    # for i, doc in enumerate(relevant_chunks, 1):
        # print(f"\nChunk {i}:\n{doc.page_content}")
    
    # Generate response using Groq API
    response = generate_response(relevant_chunks, user_query)
    
    print("\nGenerated Response:")
    print(response)

# Install necessary packages if not already installed
# pip install langchain transformers faiss-cpu groq

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq

def load_faiss_vector_store(index_path, model_name="all-MiniLM-L6-v2"):
    """
    Carrega a vector store para realizar a busca por similaridade
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name) # instanciando o huggingface embeddings
    
    # carregando a bvector store localmente
    vector_store = FAISS.load_local(
        folder_path=index_path,
        embeddings=embeddings,
    )
    
    return vector_store


def retrieve_relevant_chunks(query, vector_store, top_k=10):
    """
    Obtém os top-k chunks para a query passada pelo usuário, por meio da busca de similaridade
    """
    results = vector_store.similarity_search(query, k=top_k)
    return results


def generate_response(retrieved_chunks, user_query, model_name="llama3-groq-70b-8192-tool-use-preview"):
    """
    Generates a response using the Groq API based on retrieved chunks and user query.

    Gera a resposta utilizando a API do Groq baseada nos chunks obtidos e na pergunta passada pelo usuário. 
    O modelo utilizado por padrão foi o llama3 com 70 bilhões de parâmetros
    """
    # inicializando o groq
    client = Groq()
    
    # construindo o contexto a partir dos chunks
    context = "\n\n".join([f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_chunks)])
    
    # construindo a mensagem para a chamada da api
    messages = [
        {
            "role": "system",
            "content": "Considere a conversa, o contexto e a pergunta dada para dar uma resposta. Caso você não saiba uma resposta, fale 'Me desculpe, mas não tenho uma resposta para esta pergunta' em vez de tentar gerar uma resposta imprecisa. Responda apenas o que foi perguntado de maneira sucinta."
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
    
    # chamada da api
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0, # selecionar as respostas com menos 'variação'
        max_tokens=1024,
        top_p=0.65,
        stream=True,
        stop=None,
    )

    # obtendo a resposta e retornando para o usuário
    generated_response = ''
    for chunk in completion:
        generated_response += (chunk.choices[0].delta.content or "")

    return generated_response


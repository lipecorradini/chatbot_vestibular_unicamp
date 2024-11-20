import streamlit as st

from query import load_faiss_vector_store, retrieve_relevant_chunks, generate_response

def main():
    st.set_page_config(page_title="Chatbot Vestibular Unicamp 2025", layout="wide") 
    
    # título da página
    st.title("📚 Chatbot Vestibular Unicamp 2025") 
    
    st.write("""
        Digite a sua pergunta sobre o Vestibular 2025 da Unicamp abaixo. 
    """)
    
    # pergunta do usuário
    user_query = st.text_input("Digite sua pergunta:", "")
    
    # Caso o botão 'submit' seja clicado, processar a pergunta
    if st.button("Submit"):
        with st.spinner("Processando sua dúvida..."):
            index_path = "../data/faiss_index"
            vector_store = load_faiss_vector_store(index_path) # carregar a vector store
            
            top_k = 15
            relevant_chunks = retrieve_relevant_chunks(user_query, vector_store, top_k=top_k) # obter os 15 chunks mais relevantes
            
            response = generate_response(relevant_chunks, user_query) # obter resposta para o usuário
            
            st.subheader("💬 Resposta Gerada:")
            st.write(response) # escrever resposta na tela

if __name__ == "__main__":
    main()

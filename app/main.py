import streamlit as st

from query import load_faiss_vector_store, retrieve_relevant_chunks, generate_response

def main():
    st.set_page_config(page_title="Chatbot Vestibular Unicamp 2025", layout="wide")
    st.title("📚 Chatbot Vestibular Unicamp 2025")
    
    st.write("""
        Digite a sua pergunta sobre o Vestibular 2025 da Unicamp abaixo. 
    """)
    
    # User input
    user_query = st.text_input("Digite sua pergunta:", "")
    
    if st.button("Submit"):
        if user_query.strip() == "":
            st.warning("Por favor, insira uma pergunta válida.")
        else:
            with st.spinner("Processando sua dúvida..."):
                index_path = "../data/faiss_index"
                vector_store = load_faiss_vector_store(index_path)
                
                top_k = 15
                relevant_chunks = retrieve_relevant_chunks(user_query, vector_store, top_k=top_k)
                
                # Teste: mostrar chunks similares
                st.subheader("🔍 Chunks Similares:")
                for i, doc in enumerate(relevant_chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(doc.page_content)
                    st.markdown("---")
                
                response = generate_response(relevant_chunks, user_query)
                
                st.subheader("💬 Resposta Gerada:")
                st.write(response)

if __name__ == "__main__":
    main()

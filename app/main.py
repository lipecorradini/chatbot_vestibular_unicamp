# app/main.py

import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq
from langchain.docstore.document import Document
import os

# Import utility functions
from query import load_faiss_vector_store, retrieve_relevant_chunks, generate_response

def main():
    st.set_page_config(page_title="FAISS Retrieval System", layout="wide")
    st.title("📚 FAISS-Based Text and Table Retrieval System")
    
    st.write("""
        Enter your query below, and the system will retrieve the most relevant information from the text and tables, 
        then generate a response using the Groq API.
    """)
    
    # User input
    user_query = st.text_input("Enter your query:", "")
    
    if st.button("Submit"):
        if user_query.strip() == "":
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Processing your query..."):
                # Load the FAISS vector store
                index_path = "../data/faiss_index"
                vector_store = load_faiss_vector_store(index_path)
                
                # Retrieve relevant chunks
                top_k = 5  # You can make this configurable
                relevant_chunks = retrieve_relevant_chunks(user_query, vector_store, top_k=top_k)
                
                # Display retrieved chunks
                st.subheader("🔍 Retrieved Chunks:")
                for i, doc in enumerate(relevant_chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(doc.page_content)
                    st.markdown("---")
                
                # Generate response
                response = generate_response(relevant_chunks, user_query)
                
                # Display generated response
                st.subheader("💬 Generated Response:")
                st.write(response)

if __name__ == "__main__":
    main()

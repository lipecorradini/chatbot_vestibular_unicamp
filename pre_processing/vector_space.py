from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

def split_text(file_path, chunk_size=1000, chunk_overlap=50):
    """
    transforma o texto em chunks de tamanho e overlaps passados como parâmetro
    """

    # abrindo arquivo para leitura
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # definindo o recursive text splitter.
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], # generalizacao dos separadores
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # separando o texto
    chunks = text_splitter.split_text(text)
    
    return chunks

def split_tables(file_path):
    """
    divide as tabelas, considerando cada linha como um chunk diferente
    """
    # abrindo arquivo para leitura
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # lista de chunks
    chunks = [line for line in lines]
    
    return chunks

def generate_embeddings_huggingface(documents, model_name="all-MiniLM-L6-v2"):
    """
    gerando os embeddings, com um encoder pré-definido, e guardando na FAISS vector store.
    """
    # gerando embeddings 
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # alocando na vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store



def main():

    # caminho de arquivos
    text_input_file = "../data/text/extracted_text.txt"
    tables_input_file = "../data/text/tables.txt"
    faiss_index_path = "../data/faiss_index"
    
    # dividindo texto em chunks
    text_chunks = split_text(text_input_file)
    
    # convertendo chunks em documentos
    text_documents = [Document(page_content=chunk, metadata={"source": "text"}) for chunk in text_chunks]
    
    # dividindo as tabelas em chunks
    table_chunks = split_tables(tables_input_file)
    
    # transformando tabela e documentos
    table_documents = [Document(page_content=chunk, metadata={"source": "table"}) for chunk in table_chunks]
    
    # unindo documentos para gerar a vector store
    all_documents = text_documents + table_documents
    
    # gerando embeddings
    print("Generating embeddings with Hugging Face and creating FAISS vector store...")
    vector_store = generate_embeddings_huggingface(all_documents)
    
    # salvando o vector store localmente (./data/faiss_index)
    vector_store.save_local(faiss_index_path)
    

if __name__ == "__main__":
    main()

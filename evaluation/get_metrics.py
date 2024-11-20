import pandas as pd
import ast
from datasets import Dataset
from ragas import EvaluationDataset
from ragas.metrics import FactualCorrectness, Faithfulness, SemanticSimilarity
from ragas import evaluate

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.query import load_faiss_vector_store, retrieve_relevant_chunks, generate_response

def safe_parse_list(x):
    """
    garantindo que cada chunk é uma lista de strings. alguns não estavam formatados corretamente antes
    """
    if isinstance(x, str): # se for string, trata e transforma em lista
        x = x.strip()
        if x.startswith('[') and x.endswith(']'):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return [x]
        else:
            return [x]

    elif isinstance(x, list): # se for lista, so retorna
        return x

    else: # se nao, transforma em string e coloca em uma lista
        return [str(x)]


def generate_answers(input_csv_path='evaluation/data/questions.csv', output_csv_path='evaluation/data/generated_dataset.csv'):
    """
    para cada pergunta definida, gera uma resposta no do modelo, e constrói o dataset para ser avaliado pelo RAGA
    """
    # carregar a vector store
    index_path = "./data/faiss_index"  
    vector_store = load_faiss_vector_store(index_path)

    # carregando as questões
    questions = pd.read_csv(input_csv_path, encoding='utf-8')

    # lista para guardar respostas do modelo e os chunks obtidos por cada recuperação de contexto
    answers = []
    all_chunks = []

    for idx, row in questions.iterrows():
        question = row['perguntas']

        # obter chunks mais relevantes
        relevant_chunks = retrieve_relevant_chunks(question, vector_store, 10)

        # gerar resposta
        response = generate_response(relevant_chunks, question)

        # salvar resposta
        answers.append(response)

        # extrair os chunks e colocá-los numa lista
        chunks_text = [doc.page_content for doc in relevant_chunks]
        all_chunks.append(chunks_text)

    # construindo dataframe 
    retrieved_info = pd.DataFrame({
        'user_input': questions['perguntas'],
        'reference': questions['verdades'], 
        'response': answers,
        'retrieved_contexts': all_chunks,    # lista de strings, terá que ser tratada
    })

    # salvando dataframe como arquivo csv
    retrieved_info.to_csv(output_csv_path, index=False, encoding='utf-8')



def calculate_metrics(generated_dataset_path='evaluation/data/generated_dataset.csv'):
    """
    calculando métricas definidas utilizando a biblioteca RAGAS.
    """
    # carregando o dataset
    generated_df = pd.read_csv(generated_dataset_path, encoding='utf-8')
   
    # tratando os contextos para se adaptar ao tipo exigido pelo ragas
    generated_df['retrieved_contexts'] = generated_df['retrieved_contexts'].apply(safe_parse_list)

    # obtendo dataset do tipo huggingface (exigido pelo RAGAs)
    hf_dataset = Dataset.from_pandas(generated_df)

    # criando dataset de avaliação
    eval_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)

    # Iinicializando o wrapper do gpt 4o
    model_name = "gpt-4o"  # Alternative model
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=model_name))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
   
    # definindo as métricas que serão avaliadas
    metrics = [
        FactualCorrectness(llm=evaluator_llm), 
        Faithfulness(llm=evaluator_llm),
        SemanticSimilarity(embeddings=evaluator_embeddings)
    ]

    # realizando a avaliação
    results = evaluate(dataset=eval_dataset, metrics=metrics)

    # converter resultados em dataframe
    results_df = results.to_pandas()

    # salvando em csv 
    results_df.to_csv('./data/evaluation_results.csv', index=False, encoding='utf-8')



if __name__ == "__main__":
   
    # gerando respostas
    generate_answers(input_csv_path='./evaluation/data/questions.csv', output_csv_path='evaluation/data/generated_dataset.csv')
    
    # calculando métricas
    calculate_metrics(generated_dataset_path='evaluation/data/generated_dataset.csv')
    
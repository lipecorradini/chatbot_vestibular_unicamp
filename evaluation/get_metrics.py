import os
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
    Safely parse a string representation of a list into an actual list.
    If parsing fails, return a list containing the original string.
    """
    if isinstance(x, str):
        x = x.strip()
        # Check if the string starts with '[' and ends with ']'
        if x.startswith('[') and x.endswith(']'):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                # If parsing fails, return the string in a list
                return [x]
        else:
            # If not list-like, wrap the string in a list
            return [x]
    elif isinstance(x, list):
        return x
    else:
        # For any other type, convert to string and wrap in a list
        return [str(x)]

def generate_answers(input_csv_path='./evaluation/questions.csv', output_csv_path='evaluation/generated_dataset.csv'):
    """
    Generates answers and retrieves relevant contexts for each question.

    Args:
        input_csv_path (str): Path to the CSV file containing questions and ground truths.
        output_csv_path (str): Path where the generated dataset will be saved.
    """
    # Load the FAISS vector store
    index_path = "./data/faiss_index"  
    vector_store = load_faiss_vector_store(index_path)

    # Load questions and ground truths
    questions = pd.read_csv(input_csv_path, encoding='utf-8')

    answers = []
    all_chunks = []

    for idx, row in questions.iterrows():
        question = row['perguntas']
        ground_truth = row['verdades']

        # Retrieve relevant chunks from the vector store
        relevant_chunks = retrieve_relevant_chunks(question, vector_store)

        # Generate response using the retrieved chunks and the question
        response = generate_response(relevant_chunks, question)

        # Append the generated answer
        answers.append(response)

        # Extract the text from each relevant chunk as a list of strings
        chunks_text = [doc.page_content for doc in relevant_chunks]
        all_chunks.append(chunks_text)

    # Construct the dataset with your specified field names
    retrieved_info = pd.DataFrame({
        'user_input': questions['perguntas'],
        'reference': questions['verdades'],  # Remains a plain string
        'response': answers,
        'retrieved_contexts': all_chunks,    # List of strings
    })

    # Save the generated dataset to a CSV file
    retrieved_info.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Generated dataset saved to {output_csv_path}")



def calculate_metrics(generated_dataset_path='evaluation/generated_dataset.csv'):
    """
    Calculates evaluation metrics for the generated dataset using gpt-3.5-turbo.

    Args:
        generated_dataset_path (str): Path to the generated dataset CSV file.
    """
    # Load the generated dataset
    try:
        generated_df = pd.read_csv(generated_dataset_path, encoding='utf-8')
        print("DataFrame loaded successfully")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Safely parse 'retrieved_contexts' from string to list
    generated_df['retrieved_contexts'] = generated_df['retrieved_contexts'].apply(safe_parse_list)
    print("Parsed 'retrieved_contexts' to lists.")

    # Verify the parsing by printing the first few entries
    print(generated_df[['retrieved_contexts']].head())

    # Convert the DataFrame to a HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(generated_df)

    # Create an EvaluationDataset from the HuggingFace Dataset
    eval_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)
    print("Evaluation Dataset created successfully")

    # Initialize the LLM and Embeddings wrappers with the gpt-3.5-turbo model
    try:
        model_name = "gpt-4o"  # Alternative model
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=model_name))
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    except Exception as e:
        print(f"Error initializing LLM or Embeddings: {e}")
        return

    # Define the metrics to evaluate
    metrics = [
        FactualCorrectness(llm=evaluator_llm), 
        Faithfulness(llm=evaluator_llm),
        SemanticSimilarity(embeddings=evaluator_embeddings)
    ]
    print("Metrics defined")

    # Perform the evaluation
    try:
        results = evaluate(dataset=eval_dataset, metrics=metrics)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    # Convert results to a pandas DataFrame and display
    try:
        results_df = results.to_pandas()
        print(results_df.head())
    except Exception as e:
        print(f"Error converting results to DataFrame: {e}")
        return

    # Optionally, save the results to a CSV file
    try:
        results_df.to_csv('./evaluation_results.csv', index=False, encoding='utf-8')
        print("Evaluation metrics saved to evaluation_results.csv")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")


if __name__ == "__main__":
    # Step 1: Generate the dataset with answers and contexts
    generate_answers(input_csv_path='./evaluation/questions.csv', output_csv_path='evaluation/generated_dataset.csv')
    
    # Step 2: Calculate the evaluation metrics
    calculate_metrics(generated_dataset_path='evaluation/generated_dataset.csv')
    
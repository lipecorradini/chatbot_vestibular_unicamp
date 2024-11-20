import pandas as pd

def return_results():
    ''' 
    mostra os resultados das métricas
    '''
    # lendo o arquivo .csv com os resultados
    results = pd.read_csv('./evaluation_results.csv')
    
    # retirando os valores que não conseguiram ser computados
    fact_isna = results['factual_correctness'].isna().sum()
    faith_isna = results['faithfulness'].isna().sum()
    semantic_isna = results['semantic_similarity'].isna().sum()

    # transformando valores nao computados em 0
    results.fillna(0, inplace=True)
    print(results.columns)

    # obtendo a soma de cada uma das métricas
    factual = sum(results['factual_correctness'])
    faithfulness = sum(results['faithfulness'])
    similarity = sum(results['semantic_similarity'])

    # devolvendo as métricas para o usuário
    print(f"Factual Correctness: {factual / (len(results) - fact_isna)}")
    print(f"Faithfulness: {faithfulness / (len(results) - faith_isna)}")
    print(f"Semantic Similarity: {similarity / (len(results) - semantic_isna)}")


if __name__ == "__main__":
    return_results()
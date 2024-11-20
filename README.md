# Chatbot para auxiliar alunos com perguntas em relação ao vestibular da Unicamp 2025

Esse projeto foi desenvolvido como uma das etapas do proceso de estágio na Neuralmind. O objetivo do projeto foi desenvolver um chatbot que, utilizando Retrieval-Augmented Generation (RAG), consiga responder dúvidas acerca da [Resolução GR-029/2024](https://www.pg.unicamp.br/norma/31879/0), documento contendo informações gerais acerca do vestibular. 

---


## Uso

O chatbot está disponível em 
[Inserir link do Deploy quando tiver terminado]()

---

## Pipeline

### 1. Coleta e Processamento de Dados
Como primeiro passo para o desenvolvimento do projeto, foi preciso obter os dados referentes à Resolução. Para isso, foi utilizada a biblioteca **requests**, e para análise dos elementos HTML, foi utilizada a biblioteca **Beautiful Soup**. Essa análise foi relevante principalmente no quesito do **tratamento de tabelas**, onde o BeautifulSoup permitiu separar as tabelas do texto original, para tratamento posterior.
No processamento das tabelas, como muitas apresentavam características que limitavam sua compreensão como texto, como células mescladas nos headers, as transformamos em arquivos *csv* com auxílio do *ChatGPT*, e depois as transformamos em texto corrido e as guardamos em um arquivo .txt.


### 2. Criação do Índice de Busca 
Para a separação do texto em chunks, separamos em duas abordagens distintas: uma para o texto completo, e a outra para as tabelas que foram extraídas do texto. 
Para definir os chunks, utilizamos o Recursive Character Text Splitter, do langchain, e para as tabelas, consideramos apenas cada linha da tabela como um chunk distinto.
Já para a geração das Embeddings, foi utilizado o modelo "all-MiniLM-L6-v2", a partir do HuggingFaceEmbeddings.
Além disso, optamos por utilizar a FAISS vector store para (//colocar função da FAISS VECTOR STORE), já que (//colocar vantagens da FAISS VECTOR STORE).
Por fim, unimos as representações vetoriais de ambas as fontes de texto e as guardamos em um arquivo separado, para ser consultado posteriormente.

### 3. Recuperação de Contexto
Para a etapa da recuperação do contexto, obtemos a pergunta do usuário como input, e realizamos uma busca de similaridade na vector store, de modo a retornar os k chunks mais relevantes em relação à pergunta. 

### 4. Geração de Respostas
Nesta etapa, o modelo LLaMA 3 (70B), acessado via a API Groq, é utilizado para gerar respostas baseadas nos chunks recuperados e na pergunta do usuário. O prompt foi elaborado combinando os chunks em um contexto estruturado e incluindo instruções para guiar o modelo, buscando respostas mais precisas. A resposta é construída a partir das saídas retornadas pelo modelo e apresentada ao usuário como resultado final.

---
## Avaliação do modelo
Para avaliar as respostas geradas pelo modelo, me baseei nas perguntas mais frequentes do vestibular da Unicamp //adicionar link, e dentre essas, selecionei e adaptei algumas para que fossem mais condizentes com o conteúdo apresentado na resolução. 
Para a avaliação, foi utilizada a biblioteca RAGAs, de modo que foi gerado um dataset contendo a pergunta, a resposta correta, resposta gerada pelo RAG e o contexto obtido pela busca. 
Com isso, foi possível avaliar a corretude dos fatos, a fidelidade dos fatos e a similaridade semântica com a resposta ideal. Tais métricas foram obtidas e salvas no arquivo "evaluation_results.csv".

As métricas obtidas pela análise foram:
Factual Correctness: 0.1361904761904762
Faithfulness: 0.6978991596638655
Semantic Similarity: 0.8360901115111442

---

## Melhorias e trabalhos futuros
Como melhorias, os principais pontos seriam trabalhar mais extensamente em testes para configurar parâmetros fundamentais para a avaliação do modelo. Algumas decisões, como o tamanho dos chunks e a quantidade de chunks extraídos por tabela, mostraram-se mais empíricas do que fundamentadas em dados que comprovem sua efetividade. Assim, realizar experimentos sistemáticos para analisar o impacto da variação desses parâmetros seria essencial para otimizar o desempenho da aplicação.

Além disso, aplicar técnicas mais sofisticadas, como o re-ranking, traria maior robustez ao modelo, permitindo que os resultados retornados pela busca fossem ordenados com base em sua relevância em relação à pergunta do usuário. Isso ajudaria a reduzir a probabilidade de incluir informações irrelevantes no contexto passado ao modelo.

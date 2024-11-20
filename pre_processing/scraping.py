import requests
from bs4 import BeautifulSoup
import csv
import os

def extract_text_from_url(url, output_file):
    
    """
    Função para extrair o texto e separar as tabelas para tratamento separado
    """
    # request para obter o conteúdo html da página
    response = requests.get(url, verify=False)
    response.raise_for_status()

    # definindo o beautifulsoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # identificando as tabelas presentes no conteúdo
    tables = soup.find_all('table')

    # remover as tabelas do conteúdo para tratamento separado
    for table in tables:
        table.decompose()

    # obtendo o texto restante do conteúdo da página
    text = soup.get_text(separator='\n')

    # limpando o texto extraído
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)

    # escrevendo num txt o texto completo
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(clean_text)


def process_tables(csv_directory, output_file):
    """
    Obtém as tabelas já transformadas em arquivos .csv e as transforma em texto corrido, criando um arquivo .txt
    """
    # arquivos para serem processados (na pasta data/tables)
    csv_filenames = [f"{i}.csv" for i in range(1, 7)] 
    
    # lista para armazenar linhas formatadas
    all_formatted_lines = []
    
    # iterando pelos arquivos das tabelas
    for csv_filename in csv_filenames:

        # obtendo o caminho completo para o arquivo
        csv_path = os.path.join(csv_directory, csv_filename)
        
        # abrir o arquivo como leitor
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            
            reader = csv.reader(csvfile)
            
            # obtem as colunas da tabela
            headers = next(reader)
            
            #  iterando pelas linhas do arquivo
            for row_num, row in enumerate(reader, start=1):
                
                # tratando células vazias
                if not any(cell.strip() for cell in row):
                    continue
                
                # retirar espaços do início e final de células textuais
                row = [cell.strip() for cell in row]
                
                # transformando texto em célula formatada
                formatted_line = ""
                for idx, (header, value) in enumerate(zip(headers, row)):
                    # consideramos para todas as tabelas a primeira coluna como identificadora e as demais como atributos
                    # por isso, as transformamos em texto com "Para __nome__ temos __atributo__ __valor__"
                    if idx == 0:
                        formatted_line += f"Para {header} {value}"
                    else:
                        formatted_line += f" o {header} é {value}, "
                
                # quebrando linha no final
                formatted_line += "\n"

                # adicionando linha formatada na lista de linhas                
                all_formatted_lines.append(formatted_line)
    
    # escrevendo todas as linhas num arquivo .txt único
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(all_formatted_lines)
    

if __name__ == "__main__":

    # definindo url e caminho dos arquivos
    url = "https://www.pg.unicamp.br/norma/31879/0"
    text_output_file = '../data/text/extracted_text.txt'
    tables_output_file = '../data/text/tables.txt'
    tables_directory = '../data/tables'

    # extrair textos e tabelas
    tables = extract_text_from_url(url, text_output_file)
    
    # processar tabelas separadamente
    process_tables(tables_directory, tables_output_file)

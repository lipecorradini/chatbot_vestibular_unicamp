import requests
from bs4 import BeautifulSoup
import csv
import os

def extract_text_from_url(url, output_file):
   
    # Make a GET request to fetch the raw HTML content
    response = requests.get(url, verify=False)
    response.raise_for_status()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Identify all table elements and print their classes
    print('Classes of each table:')
    tables = soup.find_all('table')
    for table in tables:
        print(table.get('class'))

    # Extract and store the HTML of each table in a list
    table_html_list = [str(table) for table in tables]

    # Remove all table elements from the soup to exclude them from the text
    for table in tables:
        table.decompose()

    # Extract text from the modified soup (tables have been removed)
    text = soup.get_text(separator='\n')

    # Clean up the extracted text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)

    # Write the cleaned text to the specified output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(clean_text)

    print(f"Text successfully extracted and saved to '{output_file}'.")

    # Store the table HTML list as needed
    print(f"Number of tables extracted: {len(table_html_list)}")



def process_tables(csv_directory, output_file):
    """
    Processes CSV files numbered from 1.csv to 6.csv in the specified directory.
    Converts each row into a formatted text line and writes all lines to output_file.
    
    Parameters:
    - csv_directory: Path to the directory containing CSV files.
    - output_file: Path to the output text file (e.g., 'tables.txt').
    """
    # List of CSV filenames to process
    csv_filenames = [f"{i}.csv" for i in range(1, 7)]  # ['1.csv', '2.csv', ..., '6.csv']
    
    # Initialize a list to hold all formatted lines
    all_formatted_lines = []
    
    for csv_filename in csv_filenames:
        csv_path = os.path.join(csv_directory, csv_filename)
        
        # Check if the CSV file exists
        if not os.path.isfile(csv_path):
            print(f"Warning: '{csv_filename}' does not exist in '{csv_directory}'. Skipping.")
            continue
        
        print(f"Processing '{csv_filename}'...")
        
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            try:
                headers = next(reader)  # Read the header row
            except StopIteration:
                print(f"Warning: '{csv_filename}' is empty. Skipping.")
                continue  # Empty CSV file
            
            # Clean headers by stripping whitespace
            headers = [header.strip() for header in headers]
            
            # Iterate through each data row
            for row_num, row in enumerate(reader, start=1):
                # Skip empty rows
                if not any(cell.strip() for cell in row):
                    print(f"Skipping empty row {row_num} in '{csv_filename}'.")
                    continue
                
                # Ensure the row has the same number of columns as headers
                if len(row) != len(headers):
                    print(f"Warning: Row {row_num} in '{csv_filename}' has {len(row)} cells; expected {len(headers)}. Skipping.")
                    continue
                
                # Strip whitespace from each cell
                row = [cell.strip() for cell in row]
                
                # Construct the formatted line
                formatted_line = ""
                for idx, (header, value) in enumerate(zip(headers, row)):
                    if idx == 0:
                        # First column uses "Para"
                        formatted_line += f"Para {header} {value}"
                    else:
                        # Subsequent columns use "temos"
                        formatted_line += f" temos {header}:{value}"
                
                # Add a newline character at the end
                formatted_line += "\n"
                
                # Append the formatted line to the list
                all_formatted_lines.append(formatted_line)
    
    # Write all formatted lines to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(all_formatted_lines)
    
    print(f"All tables have been processed and saved to '{output_file}'.")

if __name__ == "__main__":
    url = "https://www.pg.unicamp.br/norma/31879/0"
    text_output_file = './data/text/extracted_text.txt'
    tables_output_file = './data/text/tables.txt'
    tables_directory = './data/tables'

    # Extract text and tables from the URL
    tables = extract_text_from_url(url, text_output_file)
    
    # Process the extracted tables and save to tables.txt
    process_tables(tables_directory, tables_output_file)

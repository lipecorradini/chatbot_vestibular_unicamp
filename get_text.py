import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url, output_file):
    try:
        # Specify the CA bundle
        response = requests.get(url, verify=False)
        response.raise_for_status()

        # Parse HTML and extract text
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator='\n')

        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(clean_text)

        print(f"Text successfully extracted and saved to '{output_file}'.")

    except requests.exceptions.SSLError as ssl_err:
        print(f"SSL error: {ssl_err}")
        print("Consider updating your CA certificates or check the website's SSL configuration.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
    except Exception as ex:
        print(f"An error occurred: {ex}")

if __name__ == "__main__":
    url = "https://www.pg.unicamp.br/norma/31879/0"
    output_file = "extracted_text.txt"
    extract_text_from_url(url, output_file)

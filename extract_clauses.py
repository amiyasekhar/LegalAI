import openai
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from docx import Document
import docx2txt
import glob

# Load environment variables from the .env file
load_dotenv()

# Fetch the API key from the environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# OpenAI GPT generation function
def generate_constituent_parts(contract):
    custom_prompt = f"""
    You are a helpful legal assistant who extracts legal clauses from a contract. 
    You are to return only the legal clauses in order of apperance.
    Please do not include any of the following parts in your answer:
        - Title
        - Preamble
        - Parties
        - Recitals
        - Words of Agreement
        - Definitions
        - Business Provisions
        - General Provisions
        - Signature Block
        - Exhibits and Schedules
    
    Do not include anything that isn't a legal clausse.
    
    After each clause, on the next line after the clause, add a "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-".
 
    
    Please extract only the clauses from the following contract text: {contract}
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": custom_prompt}
        ]
    )
    # Get the generated response content
    generated_answer = response.choices[0].message.content
    return generated_answer

def docx_to_formatted_txt_with_right_spacing(directory):
    """
    Converts a Word document (.docx) into formatted plain text with proper indentation and spacing.
    
    Args:
        docx_path (str): Path to the input .docx file.
        output_txt_path (str): Path to save the output .txt file.
    
    Returns:
        str: Formatted text content extracted from the document.
    """
    if os.path.isfile(directory) and directory.endswith('.docx'):
        docx_files = [directory]
    else:
        docx_files = glob.glob(os.path.join(directory, "*.docx"))
    
    print("Glob: ", docx_files)

    if not docx_files:
        raise FileNotFoundError(f"No .docx files found in the directory: {directory}")

        
    docx_path = glob.glob(directory)
    print("Glob: ", docx_path)
    


    for file_name in docx_path:
        with open(file_name, 'rb') as infile:
            filename = f"{file_name[:-5]}.txt"
            with open(filename, 'w', encoding='utf-8') as outfile:
                doc = docx2txt.process(infile)
                outfile.write(doc)
                return doc  # Return the entire text content as a string

    


contract_text = docx_to_formatted_txt_with_right_spacing('/Users/amiyasekhar/CLM/contracts/1.Contract.docx')
print("The contract text: ", contract_text)
constituent_parts = generate_constituent_parts(contract_text)
print(constituent_parts)
with open('broken_down_contract.txt', 'w') as file:
    file.write(constituent_parts)

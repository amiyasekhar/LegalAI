import openai
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from docx import Document

# Load environment variables from the .env file
load_dotenv()

# Fetch the API key from the environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# OpenAI GPT generation function
def generate_constituent_parts(contract):
    custom_prompt = f"""
    You are a helpful legal assistant who extracts the constituent parts of a contract. 
    Here are the constituent parts of a contract in the order they typically appear:

        1. Title: Reflects the nature of the contract (e.g., Sales Agreement, Employment Contract).
        2. Preamble: Introduces the contract, stating the name of the agreement, parties involved, and the date.
        3. Parties: Specific identification of individuals or entities entering the contract.
        4. Recitals: Often beginning with "Whereas," provides background information and context for the agreement.
        5. Words of Agreement: Usually starts with "Therefore," indicating that the parties agree to the terms.
        6. Definitions: Explains key terms used throughout the contract.
        7. Body: The main part of the contract, containing several important elements including but not limited to:
            a. Offer: The proposal made by one party to another.
            b. Acceptance: Agreement to the terms of the offer.
            c. Consideration: What each party is giving or receiving in the agreement.
            d. Capacity: Indication that all parties have the legal capacity to enter the contract.
            e. Legality: Assurance that the contract's terms do not violate any laws.
            f. Mutuality of Obligation: Indication of a common intention between parties.
            g. Agreement (Meeting of the Minds): Demonstration of mutual agreement.
            h. Conditions and General Provisions: Specific terms, rights, and obligations.
            i. Scope: Detailed description of work, products, or services covered.
            j. Representations and Warranties: Statements of fact and promises made by parties.
            k. Covenants: Promises to perform specific actions.
        8. Business Provisions: Deals with the following but not limited to:
            a. Payment terms
            b. Performance requirements
            c. Closing date
            d. Duration of contract
            e. Time allowed for performance
            f. Warranties
            g. Conditions
        9. General Provisions (Boilerplate Clauses): Standard provisions and clauses and clause types that are not mentioned here. You must account for them. 
        10. Signature Block: Where parties sign to indicate their agreement.
        11. Exhibits and Schedules: Additional documents or information referenced in the main body.

    Note that a contract general follows the abovementioned format but may not necessarily match it exactly. 

    You, as the helpful legal assistant, are to help me do the following TASKS. 
    
        1. extract all the constituent parts - except for what could be deemed as General Provisions, Business Provisions, and the Body - of the contract I give you, and give me an answer strictly in the following format. DO NOT OMIT TEXT - your answer must be in the format of what is inside the triple quotes:

        '''
            Constituent Part Type: *the type of constituent it is, for example, a title, a recital, etc*
            Constituent Part: *text of the constituent part*
        '''

        2. Insert a "****End of Task 1****", and on the next line give me the contract text excluding everything but the General Provisions, Business Provisions, and the Body. DO NOT OMIT TEXT

    Based on this, perform the TASKS on the following contract: {contract}
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

def docx_to_formatted_txt_with_right_spacing(docx_path, output_txt_path):
    """
    Converts a Word document (.docx) into formatted plain text with proper indentation and spacing.
    
    Args:
        docx_path (str): Path to the input .docx file.
        output_txt_path (str): Path to save the output .txt file.
    
    Returns:
        str: Formatted text content extracted from the document.
    """
    if not os.path.isfile(docx_path):
        raise FileNotFoundError(f"The file {docx_path} does not exist.")
    
    # Load the Word document
    doc = Document(docx_path)
    output_content = ""  # String to store the full text content

    # Open the output text file for writing
    with open(output_txt_path, "w", encoding="utf-8") as txt_file:
        for para in doc.paragraphs:
            # Extract the text from each paragraph
            paragraph_text = para.text.strip()
            
            # Add indents or spaces based on paragraph format
            indent = para.paragraph_format.left_indent
            if indent:
                indent_spaces = int(indent.pt / 5)  # Approximate spaces for indentation
                paragraph_text = " " * indent_spaces + paragraph_text
            
            # Write paragraph text to the output text file (including empty lines)
            txt_file.write(paragraph_text + "\n")
            output_content += paragraph_text + "\n"

        # Ensure to preserve spacing between paragraphs
        txt_file.write("\n")
        output_content += "\n"

    print(f"Formatted Word document text has been written to {output_txt_path}.")
    return output_content  # Return the entire text content as a string

contract_text = docx_to_formatted_txt_with_right_spacing('/Users/amiyasekhar/CLM/contracts/1.Contract.docx', 'WORD_TO_TXT_5891_1.Contract.docx.txt')
constituent_parts = generate_constituent_parts(contract_text)
print(constituent_parts)
with open('broken_down_contract.txt', 'w') as file:
    file.write(constituent_parts)
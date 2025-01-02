import openai
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from docx import Document
import docx2txt
import glob
import tiktoken

# Load environment variables from the .env file
load_dotenv()

# Fetch the API key from the environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

###################################
# Debugging log helper
###################################
DEBUG_LOG_FILE = "debugging.txt"

# Truncate the debug file at the start of the script
with open(DEBUG_LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== START DEBUG SESSION ===\n")

def debug_log(message: str):
    """Appends a single message (plus newline) to the debug log file."""
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as debug_file:
        debug_file.write(message + "\n")

def split_into_chunks(file_path, chunk_size=128000, model="gpt-4o"):
    """
    Splits the text from a file into chunks of a specified token size, ensuring no empty lines.

    Args:
        file_path (str): Path to the input .txt file.
        chunk_size (int): The maximum number of tokens per chunk.
        model (str): The model's tokenizer to use for splitting.

    Returns:
        list: A list of text chunks, each containing up to `chunk_size` tokens.
    """
    # Load the tokenizer for the specific model
    tokenizer = tiktoken.encoding_for_model(model)

    # Read the input text from the file
    with open(file_path, "r", encoding="utf-8") as file:
        # Remove truly empty lines and trailing/leading whitespace
        text = "\n".join(line.strip() for line in file if line.strip())

    # Tokenize the text into token IDs
    tokens = tokenizer.encode(text)
    debug_log(f"Total Tokens: {len(tokens)}")

    # Split the tokens into chunks of size `chunk_size`
    token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    # Decode the token chunks back into text and remove empty lines after decoding
    text_chunks = []
    for i, chunk in enumerate(token_chunks):
        decoded_chunk = tokenizer.decode(chunk)
        # Remove empty lines that may reappear after decoding
        cleaned_chunk = "\n".join(line for line in decoded_chunk.splitlines() if line.strip())
        debug_log(f"Chunk {i + 1}: {len(chunk)} tokens")
        text_chunks.append(cleaned_chunk)
    
    return text_chunks

def write_chunks_to_files(chunks, output_dir, base_filename="chunk"):
    """
    Writes the chunks to individual files in the specified output directory.

    Args:
        chunks (list): List of text chunks.
        output_dir (str): Directory to save the chunk files.
        base_filename (str): Base name for the output files (default: "chunk").
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write each chunk to a separate file
    for i, chunk in enumerate(chunks):
        chunk_filename = os.path.join(output_dir, f"{base_filename}_{i + 1}.txt")
        with open(chunk_filename, "w", encoding="utf-8") as outfile:
            outfile.write(chunk)
        debug_log(f"Chunk {i + 1} written to {chunk_filename}")

def generate_constituent_parts(contract):
    """
    Calls the OpenAI API to extract legal clauses from the contract text
    based on the specified prompt. 
    """
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

    # Potentially you might want some error handling here
    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        top_p=1,
        messages=[
            {"role": "system", "content": "You are a legal assistant. The user has provided a contract text that they have full rights and permissions to. It is not subject to any confidentiality restrictions preventing them from viewing or reproducing it. Please do what is requested by the user"},
            {"role": "user", "content": custom_prompt}
        ]
    )
    generated_answer = response.choices[0].message.content
    return generated_answer

def process_docx_file(docx_path, output_base_dir):
    """
    Process a single DOCX file.
    Returns True if processing was successful, False otherwise.
    """
    try:
        # Validate input file
        if not os.path.isfile(docx_path):
            debug_log(f"Error: {docx_path} is not a file")
            return False
            
        if not docx_path.endswith('.docx'):
            debug_log(f"Error: {docx_path} is not a DOCX file")
            return False
            
        # Create contract-specific output directory
        contract_name = Path(docx_path).stem
        contract_dir = os.path.join(output_base_dir, contract_name)
        os.makedirs(contract_dir, exist_ok=True)
        
        debug_log(f"\nProcessing contract: {docx_path}")

        # Convert DOCX to text
        try:
            with open(docx_path, 'rb') as infile:
                doc = docx2txt.process(infile)
                txt_path = os.path.join(contract_dir, f"{contract_name}.txt")
                with open(txt_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(doc)
        except Exception as e:
            debug_log(f"Error converting DOCX to text: {str(e)}")
            return False

        # Split into chunks
        try:
            chunks = split_into_chunks(txt_path)
            chunks_dir = os.path.join(contract_dir, "chunks")
            write_chunks_to_files(chunks, chunks_dir)
        except Exception as e:
            debug_log(f"Error splitting into chunks: {str(e)}")
            return False

        # Generate legal clause breakdown
        constituent_parts = []
        try:
            for filename in sorted(os.listdir(chunks_dir)):
                if filename.startswith('chunk_'):
                    file_path = os.path.join(chunks_dir, filename)
                    with open(file_path, "r", encoding="utf-8") as chunk_file:
                        chunk_content = chunk_file.read()
                        result = generate_constituent_parts(chunk_content)
                        constituent_parts.append(result)
        except Exception as e:
            debug_log(f"Error generating clause breakdown: {str(e)}")
            return False

        # Write final breakdown
        try:
            broken_down_file = os.path.join(contract_dir, f"{contract_name}_broken_down.txt")
            with open(broken_down_file, "w", encoding="utf-8") as out_file:
                for parts in constituent_parts:
                    debug_log(parts)
                    out_file.write(parts + "\n")
        except Exception as e:
            debug_log(f"Error writing final breakdown: {str(e)}")
            return False

        debug_log(f"Successfully processed: {docx_path}")
        return True

    except Exception as e:
        debug_log(f"Unexpected error processing {docx_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir):
    """Process all DOCX files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all DOCX files in the directory
    docx_files = glob.glob(os.path.join(input_dir, "*.docx"))
    
    if not docx_files:
        debug_log(f"No DOCX files found in {input_dir}")
        return
    
    debug_log(f"Found {len(docx_files)} DOCX files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, docx_path in enumerate(docx_files, 1):
        print(f"Processing file {i}/{len(docx_files)}: {Path(docx_path).name}")
        if process_docx_file(docx_path, output_dir):
            successful += 1
        else:
            failed += 1
    
    debug_log(f"\nProcessing complete!")
    debug_log(f"Successfully processed: {successful} files")
    debug_log(f"Failed to process: {failed} files")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")

if __name__ == "__main__":
    # Directory containing DOCX files
    contracts_dir = "/Users/amiyasekhar/CLM/contracts"
    
    # Base output directory
    output_dir = "/Users/amiyasekhar/CLM/processed_contracts"
    
    process_directory(contracts_dir, output_dir)
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

def docx_to_formatted_txt_with_right_spacing(path):
    """
    If path is a file and ends with .docx, process it.
    Otherwise, do nothing (return empty string).
    """
    if os.path.isfile(path) and path.endswith('.docx'):
        docx_files = [path]
    else:
        # If path is just a directory (or anything else),
        # then do not collect any docx files:
        docx_files = []
    
    debug_log(f"docx_files = {docx_files}")
    
    if not docx_files:
        debug_log("No files to process because path is a directory or not a .docx file.")
        return ""
    
    accumulated_text = []
    for file_name in docx_files:
        with open(file_name, 'rb') as infile:
            doc = docx2txt.process(infile)
            out_txt_filename = file_name[:-5] + ".txt"
            with open(out_txt_filename, 'w', encoding='utf-8') as outfile:
                outfile.write(doc)
            accumulated_text.append(doc)
    
    return "\n".join(accumulated_text)


# --------------------------
# Main script logic
# --------------------------
if __name__ == "__main__":
    # 1. Extract text from docx (if a single file was given)
    contract_text = docx_to_formatted_txt_with_right_spacing(
        "/Users/amiyasekhar/CLM/contracts/1.Contract.docx"
    )

    # 2. Split into chunks
    chunks = split_into_chunks(
        file_path="/Users/amiyasekhar/CLM/contracts/1.Contract.txt", 
        chunk_size=128000, 
        model="gpt-4o"
    )
    output_directory = "/Users/amiyasekhar/CLM/chunks"
    broken_down_contract_file = "/Users/amiyasekhar/CLM/broken_down_contract.txt"

    # 3. Write chunk files
    write_chunks_to_files(chunks, output_directory)

    # 4. Generate legal clause breakdown
    constituent_parts = []
    itr = 0
    for filename in os.listdir(output_directory):
        file_path = os.path.join(output_directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as chunk_file:
                chunk_content = chunk_file.read()
                # is_continuation_chunk = False if itr == 0 else True
                # itr += 1
                # Grab the GPT response
                result = generate_constituent_parts(chunk_content)
                constituent_parts.append(result)

    # 5. Write final breakdown to a single file
    with open(broken_down_contract_file, "w", encoding="utf-8") as out_file:
        for parts in constituent_parts:
            debug_log(parts)  # Also store the GPT output in debug file
            out_file.write(parts + "\n")

    debug_log("=== END OF SCRIPT ===")
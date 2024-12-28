import tiktoken
import os

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
    print(f"Total Tokens: {len(tokens)}")  # Debugging: Print total number of tokens

    # Split the tokens into chunks of size `chunk_size`
    token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    # Decode the token chunks back into text and remove empty lines after decoding
    text_chunks = []
    for i, chunk in enumerate(token_chunks):
        decoded_chunk = tokenizer.decode(chunk)
        # Remove empty lines that may reappear after decoding
        cleaned_chunk = "\n".join(line for line in decoded_chunk.splitlines() if line.strip())
        print(f"Chunk {i + 1}: {len(chunk)} tokens")  # Debugging: Print token count per chunk
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
        print(f"Chunk {i + 1} written to {chunk_filename}")

# Example usage
input_text_file = "/Users/amiyasekhar/CLM/WORD_TO_TXT_5891_1.Contract.docx.txt"
output_directory = "/Users/amiyasekhar/CLM/chunks"

# Get the chunks from the input file
chunks = split_into_chunks(input_text_file, chunk_size=128000, model="gpt-4o")

# Write the chunks to individual files in the output directory
write_chunks_to_files(chunks, output_directory)

print(f"All chunks have been written to {output_directory}")
import os
from docx import Document

def extract_lines_from_word(doc_path):
    # Open the Word document
    doc = Document(doc_path)
    
    # Gather all lines (paragraphs) in a list
    lines = []
    for paragraph in doc.paragraphs:
        line = paragraph.text.strip()
        if line:  # Skip empty lines
            lines.append(line)
    return lines

def save_lines_to_txt(lines, txt_path):
    # Write each line to a text file
    with open(txt_path, 'w') as f:
        for line in lines:
            f.write(line + "\n")

if __name__ == "__main__":
    # Directory containing the Word files
    contract_directory = '/Users/amiyasekhar/CLM/contracts'  # Update this path as needed
    output_directory = '/Users/amiyasekhar/CLM/word_txt_outputs'  # Directory for saving txt files

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through each Word document in the contract directory
    for file_name in os.listdir(contract_directory):
        if file_name.endswith(".docx"):  # Process only .docx files
            # Full path to the Word document
            doc_path = os.path.join(contract_directory, file_name)
            
            # Extract lines from the Word document
            lines = extract_lines_from_word(doc_path)
            
            # Define the path for the output .txt file
            txt_file_name = os.path.splitext(file_name)[0] + "_LINE_BY_LINE.txt"
            txt_path = os.path.join(output_directory, txt_file_name)
            
            # Save extracted lines to the text file
            save_lines_to_txt(lines, txt_path)
            print(f"Processed {file_name} and saved to {txt_file_name}")
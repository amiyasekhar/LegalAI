from docx import Document

def docx_to_txt(input_file, output_file):
    try:
        # Load the .docx file
        doc = Document(input_file)
        # Extract all text
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        # Write text to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Successfully converted '{input_file}' to '{output_file}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python docx_to_txt.py input.docx output.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    docx_to_txt(input_file, output_file)

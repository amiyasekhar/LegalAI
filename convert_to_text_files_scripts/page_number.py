import fitz  # PyMuPDF
from docx2pdf import convert
import os

def convert_word_to_pdf(word_path, pdf_output_path):
    # Convert the Word document to PDF using docx2pdf (cross-platform)
    convert(word_path, pdf_output_path)
    print(f"Converted Word document to PDF: {pdf_output_path}")

def extract_last_text_on_each_page(pdf_path):
    # Open the PDF
    doc = fitz.open(pdf_path)
    last_text_per_page = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")  # Extract plain text from the page
        
        # Split the page text into lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if lines:  # If the page has text
            # Get the last group of text on the page
            last_group_of_text = lines[-1]
            last_text_per_page.append({last_group_of_text: page_num + 1})

    doc.close()
    return last_text_per_page

# Main function to convert Word to PDF and extract last group of text on each page
def process_word_document(word_path, pdf_output_path):
    # Convert the Word document to PDF
    convert_word_to_pdf(word_path, pdf_output_path)
    
    # Extract the last group of text from each page
    last_text_per_page = extract_last_text_on_each_page(pdf_output_path)
    
    return last_text_per_page

'''
# Example usage
word_path = '/Users/amiyasekhar/CLM/contracts/5891-european-union-and-argentina-transition-mineral-agreement-2023.docx'  # Replace with your actual Word file path
pdf_output_path = 'temp.pdf'  # Temporary PDF output path

# Process the Word document and get the last text on each page
last_text_per_page = process_word_document(word_path, pdf_output_path)

# Print the result
print(f"Last group of text on each page: {last_text_per_page}")

# Clean up by removing the temporary PDF file if needed
if os.path.exists(pdf_output_path):
    os.remove(pdf_output_path)
'''
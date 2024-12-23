from docx import Document
from page_number import process_word_document
import os

def docx_to_formatted_txt_with_right_spacing(docx_path, output_txt_path):
    # Load the Word document
    doc = Document(docx_path)
    output_content = ""  # String to store everything written to the txt file
    temp_pdf_output_path = 'temp.pdf'
    phrase_page_dict = process_word_document(docx_path, temp_pdf_output_path)
    
    if os.path.exists(temp_pdf_output_path):
        os.remove(temp_pdf_output_path)

    with open(output_txt_path, "w") as txt_file:
        for para in doc.paragraphs:
            # Extract the text from each paragraph
            paragraph_text = para.text.strip()
            
            # Add indents or spaces based on paragraph format
            indent = para.paragraph_format.left_indent
            if indent:
                indent_spaces = int(indent.pt / 5)  # Approximate spaces for indentation
                paragraph_text = " " * indent_spaces + paragraph_text

            # Check if any phrase in the paragraph matches a phrase from the phrase_page_dict
            for phrase_dict in phrase_page_dict:
                phrase = list(phrase_dict.keys())[0]
                page_number = phrase_dict[phrase]

                if phrase in paragraph_text:
                    # Add the page number after the paragraph
                    paragraph_text += f"\n|--- PAGE {page_number} ---|"
                    
                    # Remove the matched dictionary from the list
                    phrase_page_dict.remove(phrase_dict)
                    break  # Stop after finding the first matching phrase

            # Print the constructed paragraph text (for visualization)
            print(f"Line: {paragraph_text if paragraph_text.strip() else '[Empty Line]'}")

            # Write paragraph text to the text file
            txt_file.write(paragraph_text + "\n")
            
            # Append it to the output_content string
            output_content += paragraph_text + "\n"

        # Optionally add blank lines between paragraphs
        txt_file.write("\n")
        output_content += "\n"

    print(f"Formatted Word document text has been written to {output_txt_path}.")
    
    return output_content  # Return the entire content as a string

# Example usage
# docx_path = '../contracts/5891-european-union-and-argentina-transition-mineral-agreement-2023.docx'  # Replace with your actual Word document path
# output_txt_path = 'WORD_TO_TXT_5891-european-union-and-argentina-transition-mineral-agreement-2023.txt'  # Output .txt file path

# output_content = docx_to_formatted_txt_with_right_spacing(docx_path, output_txt_path)
# print(f"Word document content stored in string format:\n{output_content}")
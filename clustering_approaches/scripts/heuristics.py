# heuristics.py

import re
from convert_to_txt.extract_word_text import docx_to_formatted_txt_with_right_spacing

def preprocess_contract(contract):
    cleaned_paragraphs = []
    temp_paragraph = ""

    for line in contract.split("\n"):
        cleaned_line = line.strip()

        # If the line contains a page marker (e.g., "|--- PAGE 1 ---|"), skip it
        if cleaned_line.startswith("|--- PAGE") and cleaned_line.endswith("---|"):
            # If we already have accumulated text in temp_paragraph, add it to the paragraphs
            if temp_paragraph:
                cleaned_paragraphs.append(temp_paragraph.strip())
                temp_paragraph = ""  # Reset for the next paragraph
        else:
            # Append the cleaned line to temp_paragraph (handle unfinished sentences across pages)
            temp_paragraph += " " + cleaned_line

    # Add the last paragraph if any
    if temp_paragraph:
        cleaned_paragraphs.append(temp_paragraph.strip())
    return cleaned_paragraphs

# Heuristic function to identify clauses
def is_clause(paragraph):
    paragraph = paragraph.strip()
    if len(paragraph) == 0:  # Check if the paragraph is empty
        return False
    return paragraph[0].isdigit() or re.match(r'^\d+\.', paragraph)


# Heuristic function for recital based on keywords
def is_recital(paragraph):
    return "whereas" in paragraph.lower()

# Heuristic function for preamble (as a fallback if it's not a clause or recital)
def classify_paragraph(paragraph):
    if is_clause(paragraph):
        return "clause"
    elif is_recital(paragraph):
        return "recital"
    else:
        return "preamble"

# Example usage
if __name__ == "__main__":
    contract_path = '/Users/amiyasekhar/CLM/contracts/5886-united-states-and-japan-transition-mineral-agreement-2023.docx'
    output_path = 'WORD_TO_TXT_5886-united-states-and-japan-transition-mineral-agreement-2023.txt'
    output_file = 'RESULTS_heuristics.txt'
    
    # Step 1: Extract and process the document into paragraphs
    contract_from_word = docx_to_formatted_txt_with_right_spacing(contract_path, output_path)
    
    # Step 2: Preprocess the paragraphs (clean up and remove page markers)
    contract_paragraphs = preprocess_contract(contract_from_word)
    
    with open(output_file, 'w') as output:
        for para in contract_paragraphs:
            category = classify_paragraph(para)
            output.write(f"Paragraph: {para}\nCategory: {category}\n\n")
            print(f"Paragraph: {para}\nCategory: {category}\n")
    
    
    print(f"Structured contract has been written to {output_file}.")
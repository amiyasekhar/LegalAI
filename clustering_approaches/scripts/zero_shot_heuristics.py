# zero_shot_heuristics.py

from transformers import pipeline
import re
from convert_to_txt.extract_word_text import docx_to_formatted_txt_with_right_spacing

# Load LegalBERT pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model="nlpaueb/legal-bert-base-uncased")

# Preprocess contract to clean and remove page markers
def preprocess_contract(contract):
    cleaned_paragraphs = []
    temp_paragraph = ""

    for line in contract.split("\n"):
        cleaned_line = line.strip()

        # If the line contains a page marker (e.g., "|--- PAGE 1 ---|"), skip it
        if cleaned_line.startswith("|--- PAGE") and cleaned_line.endswith("---|"):
            if temp_paragraph:
                cleaned_paragraphs.append(temp_paragraph.strip())
                temp_paragraph = ""  # Reset for the next paragraph
        else:
            temp_paragraph += " " + cleaned_line

    if temp_paragraph:
        cleaned_paragraphs.append(temp_paragraph.strip())
    return cleaned_paragraphs

# Write the structured contract to a file
def write_structured_contract_to_file(structured_contract, output_file):
    with open(output_file, 'w') as f:
        for category, paragraphs in structured_contract.items():
            f.write(f"=========== {category.capitalize()} ===========\n\n")
            if paragraphs:
                for para in paragraphs:
                    f.write(f"{para}\n\n")
            else:
                f.write(f"[No {category.capitalize()} Found]\n\n")
            f.write("================================\n\n")
    print(f"Structured contract has been written to {output_file}")

# Function to categorize paragraph using zero-shot classification
def zero_shot_classify(paragraph):
    if paragraph.strip():  # Ensure the paragraph is not empty
        labels = ["preamble", "recital", "clause"]
        result = classifier(paragraph, labels)
        return result['labels'][0]
    else:
        return "other"  # Default to 'other' if the paragraph is empty

# Heuristic function to identify clauses
def is_clause(paragraph):
    paragraph = paragraph.strip()
    return len(paragraph) > 0 and (paragraph[0].isdigit() or re.match(r'^\d+\.', paragraph))

# Heuristic function for recital based on keywords
def is_recital(paragraph):
    return "whereas" in paragraph.lower()

# Apply zero-shot learning and heuristics
def zero_shot_and_apply_heuristics(paragraphs):
    structured_output = {"preamble": [], "recital": [], "clause": [], "other": []}  # Added "other" for fallback
    
    for para in paragraphs:
        if is_clause(para):
            structured_output["clause"].append(para + "\n\n")
        elif is_recital(para):
            structured_output["recital"].append(para + "\n\n")
        else:
            category = zero_shot_classify(para + "\n\n")
            # Ensure the label is valid and exists in the dictionary, otherwise default to "other"
            if category in structured_output:
                structured_output[category].append(para + "\n\n")
            else:
                structured_output["other"].append(para + "\n\n")  # Default to "other" if label is unexpected
    
    return structured_output

if __name__ == "__main__":
    contract_path = '/Users/amiyasekhar/CLM/contracts/5886-united-states-and-japan-transition-mineral-agreement-2023.docx'
    output_path = 'WORD_TO_TXT_5886-united-states-and-japan-transition-mineral-agreement-2023.txt'
    output_file = 'RESULTS_zero_shot_heuristics.txt'
    
    # Step 1: Extract and process the document into paragraphs
    contract_from_word = docx_to_formatted_txt_with_right_spacing(contract_path, output_path)
    
    # Step 2: Preprocess the paragraphs (clean up and remove page markers)
    contract_paragraphs = preprocess_contract(contract_from_word)
    
    # Step 3: Apply zero-shot classification and heuristics
    structured_contract = zero_shot_and_apply_heuristics(contract_paragraphs)

    # Step 4: Write the structured contract to a file
    write_structured_contract_to_file(structured_contract, output_file)
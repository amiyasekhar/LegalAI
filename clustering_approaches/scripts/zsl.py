# zero_shot_learning.py

from transformers import pipeline
from convert_to_txt.extract_word_text import docx_to_formatted_txt_with_right_spacing

# Load LegalBERT pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model="nlpaueb/legal-bert-base-uncased")

# Preprocess contract to clean and remove page markers
def preprocess_contract(contract):
    cleaned_paragraphs = []
    temp_paragraph = ""

    for line in contract.split("\n"):
        cleaned_line = line.strip()

        # If the line contains a page marker (e.g., "|--- PAGE X ---|"), skip it
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
    labels = ["preamble", "recital", "clause"]
    if not paragraph.strip():  # Skip empty paragraphs
        return "other"
    
    result = classifier(paragraph, labels)
    return result['labels'][0]  # Return the top predicted label

# Process paragraphs and classify them
def process_paragraphs(contract_paragraphs):
    structured_contract = {"preamble": [], "recital": [], "clause": [], "other": []}
    
    for para in contract_paragraphs:
        if para.strip():  # Check if the paragraph is not empty
            category = zero_shot_classify(para)
            structured_contract[category].append(para)
            print(f"Paragraph: {para}\nCategory: {category}\n")
        else:
            print("Skipping empty paragraph.")
    
    return structured_contract

# Main sequence for execution when run directly
if __name__ == "__main__":
    contract_path = '/Users/amiyasekhar/CLM/contracts/5886-united-states-and-japan-transition-mineral-agreement-2023.docx'
    output_path = 'WORD_TO_TXT_5886-united-states-and-japan-transition-mineral-agreement-2023.txt'
    output_file = 'RESULTS_zsl.txt'
    
    # Step 1: Extract and process the document into paragraphs
    contract_from_word = docx_to_formatted_txt_with_right_spacing(contract_path, output_path)
    
    # Step 2: Preprocess the paragraphs (clean up and remove page markers)
    contract_paragraphs = preprocess_contract(contract_from_word)
    
    # Step 3: Process and classify paragraphs
    structured_contract = process_paragraphs(contract_paragraphs)
    
    # Step 4: Write the structured contract to a file
    write_structured_contract_to_file(structured_contract, output_file)

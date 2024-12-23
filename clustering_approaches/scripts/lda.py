from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from convert_to_txt.extract_word_text import docx_to_formatted_txt_with_right_spacing
import numpy as np

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

# Apply LDA to the paragraphs
def apply_lda(paragraphs):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(paragraphs)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)
    return lda.transform(X)

# Function to assign sections based on LDA results
def assign_sections(paragraphs, lda_results):
    structured_contract = {
        'preamble': [],
        'recital': [],
        'clauses': []
    }

    for i, probs in enumerate(lda_results):
        # Get the topic with the highest probability
        topic = np.argmax(probs)

        if topic == 0:
            structured_contract['preamble'].append(paragraphs[i])
        elif topic == 1:
            structured_contract['recital'].append(paragraphs[i])
        else:
            structured_contract['clauses'].append(paragraphs[i])

    return structured_contract

# Write the structured contract to a file
def write_structured_contract_to_file(structured_contract, output_file):
    with open(output_file, 'w') as f:
        f.write("=========== Preamble ===========\n\n")
        if structured_contract['preamble']:
            for preamble in structured_contract['preamble']:
                f.write(f"Paragraph: {preamble}\n\n")
        else:
            f.write("[No Preamble Found]\n\n")

        f.write("=========== Recital ============\n\n")
        if structured_contract['recital']:
            for recital in structured_contract['recital']:
                f.write(f"Paragraph: {recital}\n\n")
        else:
            f.write("[No Recital Found]\n\n")

        f.write("=========== Clauses ============\n\n")
        if structured_contract['clauses']:
            for clause in structured_contract['clauses']:
                f.write(f"Paragraph: {clause}\n\n")
        else:
            f.write("[No Clauses Found]\n\n")
        
        f.write("================================\n")

    print(f"Structured contract has been written to {output_file}")

# Main script logic
if __name__ == "__main__":
    contract_path = '/Users/amiyasekhar/CLM/contracts/5886-united-states-and-japan-transition-mineral-agreement-2023.docx'
    output_path = 'WORD_TO_TXT_5886-united-states-and-japan-transition-mineral-agreement-2023.txt'
    output_file = 'RESULTS_lda.txt'
    
    # Step 1: Extract and process the document into paragraphs
    contract_from_word = docx_to_formatted_txt_with_right_spacing(contract_path, output_path)
    
    # Step 2: Preprocess the paragraphs (clean up and remove page markers)
    contract_paragraphs = preprocess_contract(contract_from_word)
    
    # Step 3: Apply LDA and get the probability distributions for each paragraph
    lda_results = apply_lda(contract_paragraphs)
    print("LDA Results:", lda_results)

    # Step 4: Assign paragraphs to sections based on LDA results
    structured_contract = assign_sections(contract_paragraphs, lda_results)

    # Step 5: Write the structured contract to a file
    write_structured_contract_to_file(structured_contract, output_file)

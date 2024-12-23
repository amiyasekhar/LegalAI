# clustering_heuristics.py

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import numpy as np
from convert_to_txt.extract_word_text import docx_to_formatted_txt_with_right_spacing

# Load LegalBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Preprocess raw contract text file
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

# Function to get paragraph embeddings
def get_embeddings(paragraphs):
    embeddings = []
    for para in paragraphs:
        inputs = tokenizer(para, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

# Heuristic function to identify clauses
def is_clause(paragraph):
    """
    Determines if the paragraph is a clause based on whether the first character is a digit.
    """
    paragraph = paragraph.strip()
    if len(paragraph) == 0:  # Check if the paragraph is empty
        return False
    return paragraph[0].isdigit()

# Heuristic function for recital based on keywords
def is_recital(paragraph):
    return "whereas" in paragraph.lower()

# Function to cluster paragraphs and apply heuristics
def cluster_and_apply_heuristics(paragraphs):
    embeddings = get_embeddings(paragraphs)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(embeddings)
    labels = kmeans.labels_
    
    structured_output = {"preamble": [], "recital": [], "clauses": []}
    
    for i, para in enumerate(paragraphs):
        if is_clause(para):
            structured_output["clauses"].append(para)
        elif is_recital(para):
            structured_output["recital"].append(para)
        else:
            structured_output["preamble"].append(para)
    
    return structured_output


if __name__ == "__main__":
    contract_path = '/Users/amiyasekhar/CLM/contracts/5886-united-states-and-japan-transition-mineral-agreement-2023.docx'
    output_path = 'WORD_TO_TXT_5886-united-states-and-japan-transition-mineral-agreement-2023.txt'
    output_file = 'RESULTS_clustering_heuristics.txt'
    
    # Step 1: Extract and process the document into paragraphs
    contract_from_word = docx_to_formatted_txt_with_right_spacing(contract_path, output_path)
    
    # Step 2: Preprocess the paragraphs (clean up and remove page markers)
    contract_paragraphs = preprocess_contract(contract_from_word)
    
    # Step 3: Cluster and apply heuristics
    structured_contract = cluster_and_apply_heuristics(contract_paragraphs)

    # Step 4: Write the structured contract to a file
    with open(output_file, "w") as f:
        f.write("Structured Contract Output\n")
        f.write("=" * 40 + "\n\n")

        # Writing Preamble
        f.write("Preamble:\n")
        f.write("=" * 20 + "\n")
        for para in structured_contract["preamble"]:
            f.write(para.strip() + "\n\n")

        # Writing Recital
        f.write("Recital:\n")
        f.write("=" * 20 + "\n")
        for para in structured_contract["recital"]:
            f.write(para.strip() + "\n\n")

        # Writing Clauses
        f.write("Clauses:\n")
        f.write("=" * 20 + "\n")
        for para in structured_contract["clauses"]:
            f.write(para.strip() + "\n\n")
    
    print(f"Structured contract has been written to {output_file}.")
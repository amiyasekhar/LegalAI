# lda_clustering.py

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
from convert_to_txt.extract_word_text import docx_to_formatted_txt_with_right_spacing
import re

# Load LegalBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

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

# Function to get paragraph embeddings using LegalBERT
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
    paragraph = paragraph.strip()
    if len(paragraph) == 0:  # Check if the paragraph is empty
        return False
    return paragraph[0].isdigit() or re.match(r'^\d+\.', paragraph)

# Heuristic function for recital based on keywords
def is_recital(paragraph):
    return "whereas" in paragraph.lower()

# LDA to detect topic changes
def apply_lda(paragraphs):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(paragraphs)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)
    return lda.transform(X)

# Combine LDA topic detection and clustering
def lda_and_clustering(paragraphs):
    lda_results = apply_lda(paragraphs)
    embeddings = get_embeddings(paragraphs)
    
    # Combine LDA topics with embeddings for clustering
    combined_features = np.hstack([lda_results, embeddings])
    
    kmeans = KMeans(n_clusters=3, random_state=42).fit(combined_features)
    labels = kmeans.labels_

    structured_output = {"preamble": [], "recital": [], "clauses": []}
    
    for i, para in enumerate(paragraphs):
        if is_clause(para):
            structured_output["clauses"].append(para)
        elif is_recital(para):
            structured_output["recital"].append(para)
        else:
            if labels[i] == 0:
                structured_output["preamble"].append(para)
            elif labels[i] == 1:
                structured_output["recital"].append(para)
            else:
                structured_output["clauses"].append(para)
    
    return structured_output

def write_structured_contract_to_file(structured_contract, output_file):
    with open(output_file, 'w') as f:
        f.write("=========== Preamble ===========\n\n")
        if structured_contract['preamble']:
            for preamble in structured_contract['preamble']:
                f.write(preamble + "\n\n")
        else:
            f.write("[No Preamble Found]\n\n")

        f.write("=========== Recital ============\n\n")
        if structured_contract['recital']:
            for recital in structured_contract['recital']:
                f.write(recital + "\n\n")
        else:
            f.write("[No Recital Found]\n\n")

        f.write("=========== Clauses ============\n\n")
        if structured_contract['clauses']:
            for clause in structured_contract['clauses']:
                f.write(clause + "\n\n")
        else:
            f.write("[No Clauses Found]\n\n")
        
        f.write("================================\n")

    print(f"Structured contract has been written to {output_file}")

# Example usage
if __name__ == "__main__":
    contract_path = '/Users/amiyasekhar/CLM/contracts/5886-united-states-and-japan-transition-mineral-agreement-2023.docx'
    output_path = 'WORD_TO_TXT_5886-united-states-and-japan-transition-mineral-agreement-2023.txt'
    output_file = 'RESULTS_lda_clustering.txt'
    
    # Step 1: Extract and process the document into paragraphs
    contract_from_word = docx_to_formatted_txt_with_right_spacing(contract_path, output_path)
    
    # Step 2: Preprocess the paragraphs (clean up and remove page markers)
    contract_paragraphs = preprocess_contract(contract_from_word)
    
    structured_contract = lda_and_clustering(contract_paragraphs)
    write_structured_contract_to_file(structured_contract, output_file)
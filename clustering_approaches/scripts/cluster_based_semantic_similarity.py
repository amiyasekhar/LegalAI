import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import numpy as np
from convert_to_txt.extract_word_text import docx_to_formatted_txt_with_right_spacing

# Load LegalBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Function to get paragraph embeddings using LegalBERT
def get_embeddings(paragraphs):
    embeddings = []
    for para in paragraphs:
        inputs = tokenizer(para, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

# Function to cluster paragraphs using KMeans
def cluster_paragraphs(paragraphs, n_clusters=6):
    embeddings = get_embeddings(paragraphs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    return kmeans.labels_

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

# Write clustered results to a file in a readable format
def write_clustered_results_to_file(paragraphs, labels, output_file):
    """
    Writes the paragraphs and their corresponding cluster labels to a file in the format:
    
    Cluster 0:
    Para 1:
    
    Para 2:
    
    _________
    
    Cluster 1:
    Para 1:
    
    Para 2:
    
    _________
    """
    with open(output_file, "w") as f:
        cluster_dict = {}
        for para, label in zip(paragraphs, labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(para)

        for label in sorted(cluster_dict.keys()):
            f.write(f"Cluster {label}:\n")
            for i, para in enumerate(cluster_dict[label], 1):
                f.write(f"Para {i}:\n")
                f.write(para.strip() + "\n\n")  # Write each paragraph with a space after it
            f.write("_________\n\n")  # Line between clusters

        print(f"Clustered results have been written to {output_file}.")


if __name__ == "__main__":
    # Example paths
    contract_path = '/Users/amiyasekhar/CLM/contracts/5886-united-states-and-japan-transition-mineral-agreement-2023.docx'
    output_path = 'WORD_TO_TXT_5886-united-states-and-japan-transition-mineral-agreement-2023.txt'
    output_file = 'RESULTS_cluster_based_semantic_similarity.txt'
    
    # Step 1: Extract and process the document into paragraphs
    contract_from_word = docx_to_formatted_txt_with_right_spacing(contract_path, output_path)
    
    # Step 2: Preprocess the paragraphs (clean up and remove page markers)
    cleaned_paragraphs = preprocess_contract(contract_from_word)
    
    # Step 3: Cluster the paragraphs with 6 clusters (matching the 6 sections)
    labels = cluster_paragraphs(cleaned_paragraphs, n_clusters=6)
    
    # Step 4: Write the results to a file in a readable format
    write_clustered_results_to_file(cleaned_paragraphs, labels, output_file)
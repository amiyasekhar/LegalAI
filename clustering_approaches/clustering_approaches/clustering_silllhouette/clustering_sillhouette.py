# contract_clause_clustering.py

"""
Contract Clause Clustering Script

This script performs unsupervised clustering on contract clauses extracted from Word documents.
It processes the text, converts it into numerical features, clusters the clauses, and visualizes the results.

Dependencies:
- Python 3.x
- nltk
- scikit-learn
- sentence-transformers
- matplotlib
- seaborn
- pandas
- umap-learn

Ensure all dependencies are installed before running the script:
pip install nltk scikit-learn sentence-transformers matplotlib seaborn pandas umap-learn
"""

import re
import nltk
import numpy as np  # Added import for NumPy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer, models
# import umap
import logging

# -----------------------------------
# 1. Configure Logging
# -----------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# -----------------------------------
# 2. Data Extraction (Simulated)
# -----------------------------------
# For demonstration purposes, we'll simulate the extracted_children list.
# In practice, replace this with your actual data extraction logic.

def parse_line(line):
    """
    Parses a line and extracts child_number, type, and content.

    Returns:
        dict: A dictionary with keys 'child_number', 'type', and 'content'.
              Returns None if the line doesn't match the expected pattern.
    """
    # Regular expression to match the pattern
    match = re.match(r'Child\s+(\d+)\s+\((p|tbl)\):\s*(.*)', line)
    if match:
        child_number = int(match.group(1))
        child_type = match.group(2)
        content = match.group(3).strip()
        return {
            "child_number": child_number,
            "type": child_type,
            "content": content
        }
    else:
        return None

def extract_dataset(input_file_path):
    """
    Reads the input file, parses each line, and returns the dataset as a list.

    Args:
        input_file_path (str): Path to the input text file.

    Returns:
        list: A list of dictionaries containing the extracted data.
    """
    dataset = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parsed = parse_line(line)
            if parsed:
                dataset.append(parsed)
            else:
                # Optionally, handle lines that don't match the pattern
                print(f"Warning: Line skipped due to unmatched format:\n{line.strip()}")
    
    # Optionally, sort the dataset by child_number to maintain order
    dataset.sort(key=lambda x: x["child_number"])
    
    return dataset

input_file = 'test.txt'  # Replace with your actual input file name

# Extract the dataset
extracted_children = extract_dataset(input_file)

# -----------------------------------
# 3. Text Preprocessing
# -----------------------------------
def preprocess_text(text):
    """
    Preprocesses the input text by:
    - Detecting and handling empty or placeholder paragraphs.
    - Lowercasing.
    - Removing punctuation, digits, and special characters.
    - Removing stop words.
    - Lemmatizing words.

    Parameters:
    - text (str): The input text to preprocess.

    Returns:
    - str: The cleaned and preprocessed text, or an empty string if the text is a placeholder.
    """
    # Define patterns that indicate empty or placeholder paragraphs
    placeholder_patterns = [
        r'^\*{3,}.*\*{3,}$',  # Matches texts like "***THIS IS AN EMPTY PARA***"
        r'^---+$',             # Matches texts like "-----"
        r'^\s*$',              # Matches empty strings or strings with only whitespace
        # Add more patterns as needed
    ]

    # Check if the text matches any placeholder pattern
    for pattern in placeholder_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            logging.info(f"Detected placeholder or empty paragraph: '{text.strip()}'")
            return ""  # Return an empty string or handle as desired

    # Proceed with preprocessing

    # Lowercase the text
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Tokenize the text into words
    tokens = text.split()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Reconstruct the cleaned text
    clean_text = ' '.join(tokens)

    return clean_text

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Apply preprocessing to all clauses
cleaned_texts = [preprocess_text(child['content']) for child in extracted_children]

# -----------------------------------
# 4. Filter Out Placeholders
# -----------------------------------
# Remove clauses that were identified as placeholders (empty strings)
filtered_texts = [text for text in cleaned_texts if text.strip() != ""]
filtered_children = [child for child, text in zip(extracted_children, cleaned_texts) if text.strip() != ""]

# -----------------------------------
# 5. Feature Extraction
# -----------------------------------
# Option 1: TF-IDF Vectorization
# tfidf_vectorizer = TfidfVectorizer(max_features=500)
# tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_texts)

# Option 2: Embedding-Based Representations using Sentence Transformers
# Uncomment the following lines to use embeddings instead of TF-IDF

'''
model = SentenceTransformer("Stern5497/sbert-legal-xlm-roberta-base")  # Corrected model name
embeddings = model.encode(filtered_texts, convert_to_tensor=False)
embeddings = np.array(embeddings)  # Keep embeddings as NumPy array for compatibility
'''
# Load the BERT model
word_embedding_model = models.Transformer("nlpaueb/bert-base-uncased-contracts")

# Add a pooling layer to generate sentence embeddings
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)

# Combine the models into a SentenceTransformer
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Encode the filtered texts to get embeddings
embeddings = model.encode(filtered_texts, convert_to_tensor=False)
embeddings = np.array(embeddings)  # Ensure embeddings are in NumPy array format

# -----------------------------------
# 6. Dimensionality Reduction (Optional)
# -----------------------------------
# Using PCA to reduce dimensionality
pca = PCA(n_components=50, random_state=42)
reduced_embeddings = pca.fit_transform(embeddings)

# Alternatively, using UMAP for dimensionality reduction
# Uncomment the following lines to use UMAP
# umap_reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
# umap_embeddings = umap_reducer.fit_transform(reduced_embeddings)

# -----------------------------------
# 7. Clustering - Determining Optimal K Using Silhouette Score
# -----------------------------------
def compute_silhouette_scores(data, max_k=10):
    """
    Computes silhouette scores for different values of K and plots the results.

    Parameters:
    - data (array-like): The feature matrix.
    - max_k (int): The maximum number of clusters to test.

    Returns:
    - optimal_k (int): The optimal number of clusters based on the highest silhouette score.
    """
    scores = []
    K = range(2, max_k+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
        print(f'For n_clusters = {k}, the Silhouette Score is {score:.4f}')

    # Plot the Silhouette Scores
    plt.figure(figsize=(8, 4))
    plt.plot(K, scores, 'bo-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Various K')
    plt.xticks(K)
    plt.show()

    # Determine the optimal K as the one with the highest silhouette score
    optimal_k = K[scores.index(max(scores))]
    print(f'\nOptimal number of clusters based on Silhouette Score: {optimal_k}')

    return optimal_k

# Compute and plot Silhouette Scores using reduced_embeddings
# optimal_k = compute_silhouette_scores(reduced_embeddings, max_k=10)
optimal_k = 3

# Apply K-Means Clustering with optimal_k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(reduced_embeddings)
labels = kmeans.labels_

# Alternatively, use Agglomerative Clustering
# agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
# labels = agg_clustering.fit_predict(reduced_embeddings)

# Alternatively, use DBSCAN
# dbscan = DBSCAN(eps=0.5, min_samples=2)
# labels = dbscan.fit_predict(reduced_embeddings)

# -----------------------------------
# 8. Evaluation
# -----------------------------------
sil_score = silhouette_score(reduced_embeddings, labels)
print(f'Silhouette Score for K={optimal_k}: {sil_score:.4f}')

# -----------------------------------
# 9. Visualization
# -----------------------------------
# Using t-SNE for 2D visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(reduced_embeddings)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=tsne_embeddings[:,0], y=tsne_embeddings[:,1], hue=labels, palette='tab10', s=100)
plt.title(f'K-Means Clusters Visualized with t-SNE (K={optimal_k})')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Cluster')
plt.show()

# Alternatively, using UMAP for visualization
# Uncomment the following lines if you wish to visualize using UMAP

# plt.figure(figsize=(10, 7))
# sns.scatterplot(x=umap_embeddings[:,0], y=umap_embeddings[:,1], hue=labels, palette='tab10', s=100)
# plt.title(f'K-Means Clusters Visualized with UMAP (K={optimal_k})')
# plt.xlabel('UMAP Dimension 1')
# plt.ylabel('UMAP Dimension 2')
# plt.legend(title='Cluster')
# plt.show()

# -----------------------------------
# 10. Analyzing Clusters
# -----------------------------------
# Create a DataFrame for easy analysis
df_clusters = pd.DataFrame({
    'Clause': [child['content'] for child in filtered_children],
    'Cleaned_Clause': filtered_texts,
    'Cluster': labels
})

# Display sample clauses from each cluster
with open("clusters_ch.txt", 'w') as file:

    for cluster in range(optimal_k):
        print(f"\n--- Cluster {cluster} ---")
        file.write(f"\n--- Cluster {cluster} ---\n")
        sample_clauses = df_clusters[df_clusters['Cluster'] == cluster]['Clause'] #.head(5)
        for idx, clause in enumerate(sample_clauses, 1):
            print(f"{idx}. {clause[:len(clause)]}...")  # Print first 200 characters
            file.write(f"{idx}. {clause[:len(clause)]}\n")
        file.write("\n")

# -----------------------------------
# 11. Optional: Topic Modeling with LDA
# -----------------------------------

# Since TF-IDF Vectorization is commented out, LDA cannot be performed.
# To enable LDA, you need to uncomment the TF-IDF Vectorizer section and ensure tfidf_matrix is defined.

# def display_topics(model, feature_names, no_top_words):
#     """
#     Displays the top words for each topic in the LDA model.
#
#     Parameters:
#     - model: Trained LDA model.
#     - feature_names (list): List of feature names from the vectorizer.
#     - no_top_words (int): Number of top words to display for each topic.
#     """
#     for topic_idx, topic in enumerate(model.components_):
#         print(f"\nTopic {topic_idx + 1}:")
#         top_features = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
#         print(", ".join(top_features))

# # Initialize LDA
# lda = LatentDirichletAllocation(n_components=optimal_k, random_state=42)
# lda.fit(tfidf_matrix)

# # Display topics
# display_topics(lda, tfidf_vectorizer.get_feature_names_out(), 10)

# -----------------------------------
# 12. Save Clustering Results (Optional)
# -----------------------------------
# Save the DataFrame to a CSV file
df_clusters.to_csv('clustered_contract_clauses.csv', index=False)
print("\nClustering results saved to 'clustered_contract_clauses.csv'.")

# -----------------------------------
# 13. Additional Tips
# -----------------------------------
"""
- Experiment with different numbers of clusters (K) to see how it affects the grouping.
- Try different feature extraction methods, such as using sentence embeddings.
- Collaborate with legal experts to interpret and validate the clusters.
- Enhance preprocessing by handling specific legal terminology or phrases.
- Integrate this script into a larger pipeline that processes multiple contracts.
"""
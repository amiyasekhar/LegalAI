# contract_clause_clustering.py

"""
Contract Clause Clustering Script

This script performs unsupervised clustering on contract clauses extracted from Word documents.
It processes the text, converts it into numerical features using Sentence Transformers, applies
dimensionality reduction, clusters the clauses, evaluates the results, and visualizes the clusters.

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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
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

# Example structure of extracted_children
extracted_children = [
    {"child_number": 1, "type": "p", "content": "MASTER SERVICES AGREEMENT"},
    {"child_number": 2, "type": "p", "content": "***THIS IS AN EMPTY PARA***"},
    {"child_number": 3, "type": "p", "content": "This Master Services Agreement (\"MSA\") is executed on this [] (hereinafter referred to as \"Effective Date\")."},
    {"child_number": 4, "type": "p", "content": "***THIS IS AN EMPTY PARA***"},
    {"child_number": 5, "type": "p", "content": "By and Between"},
    {"child_number": 6, "type": "p", "content": "***THIS IS AN EMPTY PARA***"},
    {"child_number": 7, "type": "p", "content": "ABC India Private Limited, a company incorporated under the provisions of the Companies Act, 1956, having its registered office at [], hereinafter referred to as \"ABC\" (which expression shall unless repugnant to the context or contrary to the meaning thereof mean and include its successors and permitted assigns) of the FIRST PART."},
    {"child_number": 8, "type": "p", "content": "***THIS IS AN EMPTY PARA***"},
    {"child_number": 9, "type": "p", "content": "AND"},
    {"child_number": 10, "type": "p", "content": "***THIS IS AN EMPTY PARA***"},
    {"child_number": 11, "type": "p", "content": "XYZ Limited, a company incorporated under the provisions of the Companies Act, [], having its registered office at [], hereinafter referred to as \"BSL\" (which expression shall unless repugnant to the context or contrary to the meaning thereof mean and include its successors and permitted assigns) of the SECOND PART."},
    {"child_number": 12, "type": "p", "content": "***THIS IS AN EMPTY PARA***"},
    {"child_number": 13, "type": "p", "content": "\"ABC\" and \"BSL\" hereto shall hereinafter be collectively referred to as the \"Parties\" and individually as a \"Party\"."},
    {"child_number": 14, "type": "p", "content": "***THIS IS AN EMPTY PARA***"},
    {"child_number": 15, "type": "p", "content": "WHEREAS"},
    {"child_number": 16, "type": "p", "content": "ABC is a wholly owned subsidiary of MQZ Private Limited which is, one of the leading international companies in Information and Communications Technology (ICT)."},
    {"child_number": 17, "type": "p", "content": "BSL is engaged in the business of IT resource and staffing company which has 18+ years of rich experience with extensive and successful large global program delivery experience."},
    {"child_number": 18, "type": "p", "content": "ABC is currently providing IT services and undertaking offshore technical consultancy services to FGPQ Automotive Germany GmbH (\"SEG\") and similar other clients on several enterprise level platforms."},
    {"child_number": 19, "type": "p", "content": "Based on the representations of the BSL, ABC is keen to engage BSL for providing Contractual Services (defined below) and BSL has agreed to accept an engagement of ABC on the terms and conditions hereinafter contained."},
    {"child_number": 20, "type": "p", "content": "Parties agrees to use this MSA for similar clients to the extent as possible."},
    {"child_number": 21, "type": "p", "content": "***THIS IS AN EMPTY PARA***"},
    {"child_number": 22, "type": "p", "content": "NOW, THEREFORE, in consideration of the mutual promises herein, the Parties agree as follows:"},
    {"child_number": 23, "type": "p", "content": "DEFINITIONS"},
    {"child_number": 24, "type": "p", "content": "In this MSA, unless the context otherwise requires, the following expressions have the following meanings:"},
    {"child_number": 25, "type": "p", "content": "\"Applicable Data Protection Laws\" EU General Data Protection Laws, Information Technology Act, 2000, The Information Technology (Reasonable Security Practices and Procedures and Sensitive Personal Data or Information) Rules 2011 or any other statute, laws, regulations, ordinances, rules, determination, judgments, rule of law, orders, decrees, policies, guidelines to the extent applicable to the Parties of this MSA."},
    {"child_number": 26, "type": "p", "content": "\"Applicable Laws\" means all Indian statutes, laws, regulations, ordinances, rules, determination, judgments, rule of law, orders, decrees, policies, guidelines, permits, approvals, concessions, grants, franchises, licenses, requirements, or any similar form of decision of, or any provision or condition of any permit, license or other operating authorization issued by any governmental authority having or asserting jurisdiction over the matter or matters in question, whether now or hereafter in effect, to the extent applicable to the Parties to this MSA.\""},
    {"child_number": 27, "type": "p", "content": "\"Client Contract\" means contractual arrangements made by ABC with SEG and such other clients."},
    {"child_number": 28, "type": "p", "content": "\"Confidential Information\" means this MSA and all information obtained by one Party from the other, pursuant to this MSA, which is expressly marked as confidential or which is manifestly confidential or which is confirmed in writing to be confidential within 7 (seven) days of its disclosure, and each Party's confidential information shall include without limitation all operational, business, commercial and financial information, business or trade secrets, personnel information, its products and/or prospective products, ideas, marketing information, technical and commercial know-how, tools, specifications, inventions, processes and initiatives that have been disclosed to the other Party, or the other Party becomes aware of, in the course of the MSA."},
    {"child_number": 29, "type": "p", "content": "\"Consultants\" means those employees and sub-contractors of BSL engaged from time to time in providing the Contractual Services and any employees of any such sub-contractors who are so engaged;"},
    {"child_number": 30, "type": "p", "content": "\"Contractual Services\" means the services to be provided by BSL pursuant to the engagement with SEG."},
    {"child_number": 31, "type": "p", "content": "\"Developed Works\" means any and all works of authorship and materials developed, written or prepared by BSL, its employees, agents or sub-contractors in the course of providing the Contractual Services (whether individually, collectively or jointly with the ABC and on whatever media)."},
    {"child_number": 32, "type": "p", "content": "\"Force Majeure Event\" shall mean any unforeseen event which arises after the date of the MSA, which obstructs the MSA, and/or either Party from executing part or all of the its obligations under the MSA, and which by the exercise of reasonable diligence of the said Party is unable to prevent, including without limitation:"},
    {"child_number": 33, "type": "p", "content": "Acts of God (such as, but not limited to, fires, explosions, storm, earthquakes, drought, tidal waves and floods), except to the extent that such an act of God is caused, or its effects contributed to, by the Party claiming force majeure, and/or"},
    {"child_number": 34, "type": "p", "content": "War, hostilities (whether declared or not), invasion, act of public or enemies, mobilization, requisition, or embargoes or other import restrictions, acts of terrorism, rebellion, revolution, insurrection, or military or usurped power, or civil war, and/or"},
    {"child_number": 35, "type": "p", "content": "contamination by radioactivity from any nuclear fuel, or from any nuclear waste from the combustion of nuclear fuel, radio-active toxic explosive, or other hazardous properties of any explosive nuclear assembly or nuclear component of such assembly, and/or"},
    {"child_number": 36, "type": "p", "content": "Riot, commotion, strikes, go slows, industrial disturbances, sabotage, lock outs or disorder, except where solely restricted to employees of the Party, or breakdown of or injury to any facilities used for production, and/or"},
    {"child_number": 37, "type": "p", "content": "change in governmental law and/or policies, government prohibitions, lockdown, epidemic, pandemic, strike and/or emergency, and/or"},
    {"child_number": 38, "type": "p", "content": "cyber-attack, cyber terrorism, and/or malicious damage."},
    {"child_number": 39, "type": "p", "content": "\"Intellectual Property Rights\" means patents, copyright and related rights, trademarks, trade names and domain names, rights in get-up, rights in goodwill or to sue for passing off, rights in designs, rights in computer software, database rights, rights in confidential information (including know-how and trade secrets) and any other intellectual property rights, in each case whether registered or unregistered and including all applications (or rights to apply) for, and renewals or extensions of, such rights and all similar or equivalent rights or forms of protection which may now or in the future subsist in any part of the world;"},
    {"child_number": 40, "type": "p", "content": "INTERPRETATION"},
    {"child_number": 41, "type": "p", "content": "In this MSA:"},
    {"child_number": 42, "type": "p", "content": "any reference to a Party to this MSA includes a reference to his successors in title and permitted assigns;"},
    {"child_number": 43, "type": "p", "content": "the headings to the clauses are for ease of reference only and shall not affect the interpretation or construction of this MSA."},
    {"child_number": 44, "type": "p", "content": "SCOPE OF MSA"},
    {"child_number": 45, "type": "p", "content": "This MSA sets out overall framework for the provision of Contractual Services to be provided by BSL to ABC on a back to back arrangement pursuant to Client Contract."},
    {"child_number": 46, "type": "p", "content": "The Parties shall, from time to time during the Term of this MSA, enter into individual agreements (\"Purchase Order\") to set out the scope of Contractual Services to be performed by BSL under this MSA or Client Contract and consequent to any instructions given by SEG or any other similar clients. For the purpose of this MSA, SEG and such other clients shall be deemed to be as Principal Contractor."},
    {"child_number": 47, "type": "p", "content": "Purchase Order shall be concluded on the basis of the template attached to this MSA as \"Schedule A\"."},
    {"child_number": 48, "type": "p", "content": "Unless otherwise agreed in writing as a specific variation to this MSA, this MSA shall apply to the provision of all Contractual Services performed by BSL under each Purchase Order."},
    {"child_number": 49, "type": "p", "content": "TERM"},
    {"child_number": 50, "type": "p", "content": "This MSA shall commence with effect from Effective Date and shall continue until terminated in accordance with Clause 15 (\"Term\")."},
    {"child_number": 51, "type": "p", "content": "REPRESENTATION AND WARRANTIES"},
    {"child_number": 52, "type": "p", "content": "Each Party represents that:"},
    {"child_number": 53, "type": "p", "content": "It has the legal right and authority to enter into this MSA and is not barred by any agency or under Applicable Laws."},
    {"child_number": 54, "type": "p", "content": "All the information and disclosures made in respect to this MSA are true and accurate."},
    {"child_number": 55, "type": "p", "content": "It has taken all necessary authorizations and approvals for the purpose of execution of this MSA."},
    {"child_number": 56, "type": "p", "content": "***THIS IS AN EMPTY PARA***"},
    {"child_number": 57, "type": "p", "content": "BSL represents and warrants to ABC that:"},
    {"child_number": 58, "type": "p", "content": "it guarantees the functionality of the Developed Work under the Purchase Order."},
    {"child_number": 59, "type": "p", "content": "in the event of any defect in Developed Work, BSL shall modify the Developed Work in accordance terms of the Purchase Order without any additional charge."},
    {"child_number": 60, "type": "p", "content": "BSL and the Consultants will have the necessary skill and expertise to provide the Contractual Services on the terms set out in this MSA."},
    {"child_number": 61, "type": "p", "content": "the Developed Works will, so far as they do not comprise pre-existing material originating from the ABC, its employees, agents, third party or contractors, be original works of authorship and the use or possession thereof by the ABC or BSL will not subject the ABC or BSL to any claim for infringement of any proprietary rights of any third party;"},
    {"child_number": 62, "type": "p", "content": "the Contractual Services will be provided in a timely and professional manner with reasonable skill and care and on a best effort basis."},
    {"child_number": 63, "type": "p", "content": "It shall, neither by itself nor through any persons employed or acting on its behalf (including employees, directors, agents, Consultants, or approved subcontractors) (i) give, offer, or promise to give, directly or indirectly, or accept, receive or agree to accept or receive anything of value (including, but not limited to, money, services, product samples, commissions, contributions, fees, gifts, bribes, rebates, payoffs, travel expenses, entertainment, influence payments, kickbacks or any other payment), regardless of form, to any person (including the professionals) to secure a business advantage, to obtain or retain business or to direct business to or away from any entity."},
    {"child_number": 64, "type": "p", "content": "it owns all Intellectual Property Rights or has the right to use and assign such Intellectual Property Rights in the Developed Works. If BSL recognizes or must recognize that such breach is imminent there is a duty to inform ABC without any delay. In this context, ABC shall be provided with all information concerning the assertion of claims by third parties, in particular concerning the type and scope of the alleged infringement of Intellectual Property Rights."},
    {"child_number": 65, "type": "p", "content": "to the extent necessary it shall ensure its employees, subcontractors and Consultants are fully aware of the terms of this MSA that relate to them and that they comply with the same."},
    {"child_number": 66, "type": "p", "content": "any facts, opinions or property provided by BSL contained in any Developed Works, is accurate, is not misleading or defamatory, and otherwise complies with all Applicable Laws."},
    {"child_number": 67, "type": "p", "content": "in implementing Purchase Order BSL shall also use technical solutions that are produced on the basis of generally offered network platforms of BSL and third parties. If technical modifications are made to individual features of the Developed Work these changes must also be implemented in this MSA. BSL shall inform ABC to the extent technically possible, avoid any disadvantages for ABC. The conversion of Developed Work by BSL is generally free of charge for ABC."},
    {"child_number": 68, "type": "p", "content": "upon ABC request BSL shall permit ABC to sublet its IT Infrastructure even in part or any other transfer of use to third parties."},
    {"child_number": 69, "type": "p", "content": "for the purpose of providing Contractual Services, it shall adhere to all the provisions or guidelines set out in the Client Contract."},
    {"child_number": 70, "type": "p", "content": " ABC represents to BSL that it shall:"},
    {"child_number": 71, "type": "p", "content": "ensure that its employees and any sub-contractors co-operate fully and promptly with"}
]

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

model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models
embeddings = model.encode(filtered_texts, convert_to_tensor=False)
embeddings = np.array(embeddings)  # Convert to NumPy array for compatibility

# -----------------------------------
# 6. Dimensionality Reduction (Optional)
# -----------------------------------
# Using PCA to reduce dimensionality
pca = PCA(n_components=50, random_state=42)
reduced_embeddings = pca.fit_transform(embeddings)

# Alternatively, using UMAP for dimensionality reduction (for visualization purposes)
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
        logging.info(f'For n_clusters = {k}, the Silhouette Score is {score:.4f}')

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
    logging.info(f'\nOptimal number of clusters based on Silhouette Score: {optimal_k}')

    return optimal_k

# Compute and plot Silhouette Scores
optimal_k = compute_silhouette_scores(reduced_embeddings, max_k=10)

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
for cluster in range(optimal_k):
    print(f"\n--- Cluster {cluster} ---")
    sample_clauses = df_clusters[df_clusters['Cluster'] == cluster]['Clause'].head(5)
    for idx, clause in enumerate(sample_clauses, 1):
        print(f"{idx}. {clause[:200]}...")  # Print first 200 characters

# -----------------------------------
# 11. Optional: Topic Modeling with LDA
# -----------------------------------
# Since we're using Sentence Transformers and embeddings, LDA (which relies on TF-IDF) is not applicable.
# You can consider other topic modeling approaches compatible with embeddings or skip this step.

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
- Try different feature extraction methods, such as using different Sentence Transformer models.
- Collaborate with legal experts to interpret and validate the clusters.
- Enhance preprocessing by handling specific legal terminology or phrases.
- Integrate this script into a larger pipeline that processes multiple contracts.
"""

'''
umap_reducer = umap.UMAP(n_neighbors=15, n_components=50, random_state=42)
reduced_embeddings = umap_reducer.fit_transform(embeddings)
'''
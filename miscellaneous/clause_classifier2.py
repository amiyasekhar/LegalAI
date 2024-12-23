import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load Spacy's pre-trained model for English
nlp = spacy.load('en_core_web_sm')

# Example pre-defined clause types (for simplicity). In practice, this would be a trained model.
CLAUSE_TYPES = {
    'confidentiality': 'Confidentiality',
    'termination': 'Termination',
    'indemnity': 'Indemnity',
}

# Sample clauses for training
TRAINING_CLAUSES = {
    'The parties agree to keep all information confidential and not disclose it to any third party.': 'confidentiality',
    'Either party may terminate this agreement with a 30-day written notice.': 'termination',
    'The supplier agrees to indemnify the client against any claims arising from the breach.': 'indemnity',
}

# Initialize and train a simple model to classify clauses
def train_clause_classifier(clauses):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(clauses.keys())
    y_train = list(clauses.values())

    # Simple Logistic Regression classifier (you can use more complex models as needed)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    return vectorizer, classifier

# Extract clauses from the contract text
def extract_clauses(contract_text):
    # Simple clause extraction based on splitting by periods and line breaks
    doc = nlp(contract_text)
    clauses = []
    for sent in doc.sents:
        clause = sent.text.strip()
        if len(clause) > 20:  # Skip very short sentences
            clauses.append(clause)
    return clauses

# Classify clauses using the trained classifier
def classify_clauses(clauses, vectorizer, classifier):
    clause_types = []
    X_test = vectorizer.transform(clauses)
    predictions = classifier.predict(X_test)

    for i, clause in enumerate(clauses):
        clause_types.append((i + 1, clause, CLAUSE_TYPES.get(predictions[i], 'Unknown')))
    
    return clause_types

# Main function to process a contract and classify clauses
def classify_contract(contract_text):
    # Step 1: Train the classifier
    vectorizer, classifier = train_clause_classifier(TRAINING_CLAUSES)

    # Step 2: Extract clauses
    clauses = extract_clauses(contract_text)

    # Step 3: Classify clauses
    classified_clauses = classify_clauses(clauses, vectorizer, classifier)

    # Step 4: Print the results in the desired format
    print(f"{'Clause Number':<15} | {'Clause Content':<100} | {'Clause Type':<20}")
    print("-" * 140)
    for clause_num, content, clause_type in classified_clauses:
        print(f"{clause_num:<15} | {content:<100} | {clause_type:<20}")

# Example contract text (this would be your actual contract input)
contract_text = """
The parties agree to keep all information confidential and not disclose it to any third party.
Either party may terminate this agreement with a 30-day written notice.
The supplier agrees to indemnify the client against any claims arising from the supplier’s breach of this agreement.
"""

# Run the classifier on the contract
classify_contract(contract_text)

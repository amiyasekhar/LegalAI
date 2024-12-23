# clause_classifier.py

import nltk
import pandas as pd
import numpy as np
import re

# For preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# For feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# For model training and evaluation
from sklearn import svm
from sklearn.metrics import classification_report

# Download NLTK data files (only need to run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample data
data = {
    'CL_no': [1, 2, 3, 4, 5],
    'Clause_Text': [
        'The parties agree to keep all information confidential and not disclose it to any third party without prior consent.',
        'Either party may terminate this agreement with a 30-day written notice.',
        'The supplier agrees to indemnify the client against any claims arising from the supplier’s breach of this agreement.',
        'All disputes arising out of this agreement shall be settled through arbitration.',
        'The agreement shall be governed by the laws of the State of California.'
    ],
    'COT': ['Confidentiality', 'Termination', 'Indemnity', 'Dispute Resolution', 'Governing Law']
}

df = pd.DataFrame(data)

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a string
    text = ' '.join(tokens)
    return text

# Preprocess the clauses
df['Processed_Text'] = df['Clause_Text'].apply(preprocess_text)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['Processed_Text'])
y = df['COT']

# Since the dataset is small, we'll use all data for training
# In practice, split the data into training and testing sets
# For demonstration, we will train and test on the same data

# Initialize the classifier
classifier = svm.SVC(kernel='linear')

# Train the classifier
classifier.fit(X, y)

# Evaluate the model
y_pred = classifier.predict(X)
print("Classification Report:\n")
print(classification_report(y, y_pred))

# Function to classify a new clause
def classify_clause(clause_text):
    # Preprocess
    processed_text = preprocess_text(clause_text)
    # Transform
    vector = tfidf_vectorizer.transform([processed_text])
    # Predict
    prediction = classifier.predict(vector)
    return prediction[0]

# Example usage
new_clause = 'The client shall provide written notice at least 60 days prior to terminating the agreement.'
predicted_cot = classify_clause(new_clause)
print(f'Predicted COT: {predicted_cot}')

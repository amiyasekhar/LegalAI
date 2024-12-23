from transformers import BertForSequenceClassification, RobertaForSequenceClassification, T5ForConditionalGeneration, Trainer, TrainingArguments, BertTokenizer, RobertaTokenizer, T5Tokenizer
from datasets import Dataset, DatasetDict
import torch
import numpy as np
import pandas as pd

# Set the device to MPS if available, otherwise fall back to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load dataset
df1 = pd.read_csv("Clauses 1.csv")
df2 = pd.read_csv("Clauses 2.csv")
combined_df = pd.concat([df1, df2], ignore_index=True)

# Count unique labels in the 'Clause Heading' column to determine num_labels
label_column = 'Clause Heading'
unique_labels = combined_df[label_column].unique()
num_labels = len(unique_labels)

# Create a label-to-id mapping for Clause Heading labels
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Map labels to integers
combined_df['labels'] = combined_df[label_column].map(label2id)

# Split into 80% training and 20% validation for each label
train_df_list = []
validation_df_list = []

for label, group in combined_df.groupby(label_column):
    train_size = int(0.8 * len(group))
    train_df_list.append(group[:train_size])
    validation_df_list.append(group[train_size:])

# Concatenate all training and validation data
train_df = pd.concat(train_df_list).reset_index(drop=True)
validation_df = pd.concat(validation_df_list).reset_index(drop=True)

# Convert DataFrames to Hugging Face Datasets with labels
train_dataset = Dataset.from_pandas(train_df[['Clause Content', 'labels']])
validation_dataset = Dataset.from_pandas(validation_df[['Clause Content', 'labels']])

# Combine into DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset
})

# Load models and tokenizers
legal_bert_model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=num_labels).to(device)
legal_roberta_model = RobertaForSequenceClassification.from_pretrained("saibo/legal-roberta-base", num_labels=num_labels).to(device)

# Initialize tokenizers
legal_bert_tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_roberta_tokenizer = RobertaTokenizer.from_pretrained("saibo/legal-roberta-base")

# Use 't5-small' as fallback if 'doctable/t5-small-finetuned-legal' is unavailable
try:
    legal_t5_model = T5ForConditionalGeneration.from_pretrained("doctable/t5-small-finetuned-legal").to(device)
    legal_t5_tokenizer = T5Tokenizer.from_pretrained("doctable/t5-small-finetuned-legal")
except Exception as e:
    print("Could not load 'doctable/t5-small-finetuned-legal', using 't5-small' as fallback.")
    legal_t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    legal_t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Preprocessing functions for BERT/Roberta and T5
def preprocess_function_bert_roberta(examples, tokenizer):
    encoding = tokenizer(examples['Clause Content'], truncation=True, padding='max_length', max_length=512)
    encoding["labels"] = examples["labels"]
    return encoding

def preprocess_function_t5(examples):
    inputs = ["classify: " + text for text in examples['Clause Content']]
    targets = [id2label[label] for label in examples['labels']]
    model_inputs = legal_t5_tokenizer(inputs, max_length=512, padding='max_length', truncation=True)
    labels = legal_t5_tokenizer(targets, max_length=10, padding='max_length', truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize data for BERT and RoBERTa models
bert_roberta_dataset = dataset.map(lambda x: preprocess_function_bert_roberta(x, legal_bert_tokenizer), batched=True)

# Tokenize data for T5 model
t5_dataset = dataset.map(preprocess_function_t5, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Fine-tune each model
def train_model(model, dataset, tokenizer=None):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer
    )
    trainer.train()
    return trainer

# Train Legal-BERT and Legal-RoBERTa for initial classification
bert_trainer = train_model(legal_bert_model, bert_roberta_dataset, legal_bert_tokenizer)
roberta_trainer = train_model(legal_roberta_model, bert_roberta_dataset, legal_roberta_tokenizer)

# Train T5 model for legal clause classification
t5_trainer = train_model(legal_t5_model, t5_dataset, legal_t5_tokenizer)

### Step 2: Ensemble Inference with Confidence Scores

# Inference function to get predictions and confidence scores from each model
def get_predictions(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding='max_length', max_length=512).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    softmax = torch.nn.functional.softmax(logits, dim=-1)
    confidence, predicted_class = torch.max(softmax, dim=1)
    return predicted_class.item(), confidence.item()

# Select a clause from the validation set for testing
sample_clause = validation_df.sample(1).iloc[0]
clause = sample_clause['Clause Content']
actual_clause_type = sample_clause[label_column]  # Actual clause type

# Run inference on each model
bert_pred, bert_conf = get_predictions(legal_bert_model, legal_bert_tokenizer, clause)
roberta_pred, roberta_conf = get_predictions(legal_roberta_model, legal_roberta_tokenizer, clause)

# Get T5 classification as a text output (e.g., "governing law", "termination")
def get_t5_prediction(text):
    inputs = legal_t5_tokenizer("classify: " + text, return_tensors="pt", padding='max_length', max_length=512).to(device)
    outputs = legal_t5_model.generate(**inputs)
    return legal_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

t5_pred_text = get_t5_prediction(clause)

# Mapping T5 predictions to a label
t5_mapping = {label: idx for idx, label in id2label.items()}
t5_pred = t5_mapping.get(t5_pred_text, -1)

# Assign confidence for T5 if it matches any label
t5_conf = 0.9 if t5_pred != -1 else 0.0

# Final Step: Ensemble Decision Logic
predictions = [bert_pred, roberta_pred, t5_pred]
confidences = [bert_conf, roberta_conf, t5_conf]

final_pred_id = predictions[np.argmax(confidences)] if max(confidences) > 0 else -1
final_pred = id2label.get(final_pred_id, "Unknown")

# Print the results
print(f"Clause: '{clause}'")
print(f"Actual Clause Type: {actual_clause_type}")
print(f"Legal-BERT Prediction: {id2label[bert_pred]} (Confidence: {bert_conf})")
print(f"Legal-RoBERTa Prediction: {id2label[roberta_pred]} (Confidence: {roberta_conf})")
print(f"Legal-T5 Prediction: {t5_pred_text} (Mapped to Class: {id2label.get(t5_pred, 'Unknown')} with Confidence: {t5_conf})")
print(f"Final Prediction: {final_pred}")
import os
import torch
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, RobertaTokenizer
from datasets import Dataset, DatasetDict

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

legal_bert_tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_roberta_tokenizer = RobertaTokenizer.from_pretrained("saibo/legal-roberta-base")

# Preprocessing function for tokenization
def preprocess_function(examples, tokenizer):
    encoding = tokenizer(examples['Clause Content'], truncation=True, padding='max_length', max_length=512)
    encoding["labels"] = examples["labels"]
    return encoding

# Tokenize data for BERT and RoBERTa models
bert_dataset = dataset.map(lambda x: preprocess_function(x, legal_bert_tokenizer), batched=True)
roberta_dataset = dataset.map(lambda x: preprocess_function(x, legal_roberta_tokenizer), batched=True)

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
bert_trainer = train_model(legal_bert_model, bert_dataset, legal_bert_tokenizer)
roberta_trainer = train_model(legal_roberta_model, roberta_dataset, legal_roberta_tokenizer)

# Save the fine-tuned models
legal_bert_model.save_pretrained("./fine_tuned_legal_bert")
legal_roberta_model.save_pretrained("./fine_tuned_legal_roberta")
legal_bert_tokenizer.save_pretrained("./fine_tuned_legal_bert_tokenizer")
legal_roberta_tokenizer.save_pretrained("./fine_tuned_legal_roberta_tokenizer")

### Inference on New Files

# Load the fine-tuned models and tokenizers for inference
legal_bert_model = BertForSequenceClassification.from_pretrained("./fine_tuned_legal_bert").to(device)
legal_roberta_model = RobertaForSequenceClassification.from_pretrained("./fine_tuned_legal_roberta").to(device)
legal_bert_tokenizer = BertTokenizer.from_pretrained("./fine_tuned_legal_bert_tokenizer")
legal_roberta_tokenizer = RobertaTokenizer.from_pretrained("./fine_tuned_legal_roberta_tokenizer")

# Helper function to run predictions on a single paragraph
def get_predictions(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    softmax = torch.nn.functional.softmax(logits, dim=-1)
    confidence, predicted_class = torch.max(softmax, dim=1)
    return predicted_class.item(), confidence.item()

# Directory paths for contract files and output results
directory_path = '/Users/amiyasekhar/CLM/MAIN_STUIFF_contract_parsing/results_3'  # Directory containing contract text files
output_directory = '/Users/amiyasekhar/CLM/'  # Directory to store results

for file_name in os.listdir(directory_path):
    # Open the debug file once outside the loop
    debug_file_path = "debugging_cc_for_real.txt"
    with open(debug_file_path, 'w') as deb:
        for file_name in os.listdir(directory_path):
            deb.write(f"We are going through the file {file_name}\n")

            if file_name.endswith(".txt"):
                file_path = os.path.join(directory_path, file_name)
                output_file = os.path.join(output_directory, f"results_{file_name}")

                with open(file_path, 'r') as file, open(output_file, 'w') as out_file:
                    out_file.write(f"Results for {file_name}\n")

                    for line in file:
                        stripped_line = line.strip()
                        deb.write(f"Stripped line: {stripped_line}\n\n")

                        # Skip lines that are empty, contain "***THIS IS AN EMPTY PARA***", or match the pattern "Child X (type):"
                        if not stripped_line or stripped_line == "***THIS IS AN EMPTY PARA***" or (stripped_line.startswith("Child") and stripped_line.endswith(":")):
                            continue

                        # Process the valid line
                        paragraph_text = stripped_line
                        print(f"Processing line: {paragraph_text}")
                        bert_pred, bert_conf = get_predictions(legal_bert_model, legal_bert_tokenizer, paragraph_text, device)
                        roberta_pred, roberta_conf = get_predictions(legal_roberta_model, legal_roberta_tokenizer, paragraph_text, device)

                        # Ensemble decision logic
                        predictions = [bert_pred, roberta_pred]
                        print(f"Predictions for line{paragraph_text}: {predictions}")
                        confidences = [bert_conf, roberta_conf]
                        final_pred_id = predictions[np.argmax(confidences)] if max(confidences) > 0 else -1
                        final_pred = id2label.get(final_pred_id, "Unknown")

                        # Log results
                        out_file.write(f"Clause: '{paragraph_text}'\n")
                        out_file.write(f"Legal-BERT Prediction: {id2label[bert_pred]} (Confidence: {bert_conf})\n")
                        out_file.write(f"Legal-RoBERTa Prediction: {id2label[roberta_pred]} (Confidence: {roberta_conf})\n")
                        out_file.write(f"Final Prediction: {final_pred}\n")
                        out_file.write("="*80 + "\n\n")
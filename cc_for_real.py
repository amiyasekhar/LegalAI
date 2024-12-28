import os
import torch
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, RobertaTokenizer
from datasets import Dataset, DatasetDict
from concurrent.futures import ThreadPoolExecutor

# Directory paths for contract files and output results
contract_directory_path = './word_txt_outputs'
output_directory = './'
debug_file_path = "./debugging_cc_for_real.txt"
bert_model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")
if bert_model:
    print("We've Bert")
else:
    print("No bert")

# Helper function to write to a file in append mode
def write_to_file(filename, text, mode='a'):
    with open(filename, mode) as file:
        file.write(text)

# Clear the debug file if it exists
if os.path.exists(debug_file_path):
    os.remove(debug_file_path)
    print(f"Deleted: {debug_file_path}")

# Delete existing output files in the output directory
for file_name in os.listdir(output_directory):
    file_path = os.path.join(output_directory, file_name)
    if file_name.endswith(".txt"):
        os.remove(file_path)
        print(f"Deleted: {file_path}")

# Set the device to MPS if available, otherwise fall back to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Paths for models and tokenizers
bert_model_path = "./fine_tuned_legal_bert"
bert_tokenizer_path = "./fine_tuned_legal_bert_tokenizer"
roberta_model_path = "./fine_tuned_legal_roberta"
roberta_tokenizer_path = "./fine_tuned_legal_roberta_tokenizer"

# Load dataset to create label mappings regardless of training
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

# Check if the models already exist
bert_model_exists = os.path.exists(bert_model_path) and os.path.exists(bert_tokenizer_path)
roberta_model_exists = os.path.exists(roberta_model_path) and os.path.exists(roberta_tokenizer_path)

if not (bert_model_exists and roberta_model_exists):
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
    legal_bert_model.save_pretrained(bert_model_path)
    legal_roberta_model.save_pretrained(roberta_model_path)
    legal_bert_tokenizer.save_pretrained(bert_tokenizer_path)
    legal_roberta_tokenizer.save_pretrained(roberta_tokenizer_path)

else:
    # Load the fine-tuned models and tokenizers for inference
    legal_bert_model = BertForSequenceClassification.from_pretrained(bert_model_path).to(device)
    legal_roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_path).to(device)
    legal_bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
    legal_roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_tokenizer_path)

# Helper function to run predictions on a single paragraph
def get_predictions(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    softmax = torch.nn.functional.softmax(logits, dim=-1)
    confidence, predicted_class = torch.max(softmax, dim=1)
    return predicted_class.item(), confidence.item()

def extract_clauses_from_file(filepath, delimiter):
    """
    Reads the entire file, then splits the text by a specific delimiter.
    Returns a list of clauses, stripped of leading/trailing whitespace.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split on the delimiter:
    raw_clauses = text.split(delimiter)
    
    # Strip whitespace and discard any empty strings:
    clauses = [clause.strip() for clause in raw_clauses if clause.strip()]
    return clauses

contract_file = "broken_down_contract.txt"
delimiter_str = "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"
clauses_list = extract_clauses_from_file(contract_file, delimiter_str)
output_file = os.path.join(output_directory, "results_clauses_list")

for clause in clauses_list:
    bert_pred, bert_conf = get_predictions(legal_bert_model, legal_bert_tokenizer, clause, device)
    roberta_pred, roberta_conf = get_predictions(legal_roberta_model, legal_roberta_tokenizer, clause, device)
    predictions = [bert_pred, roberta_pred]
    confidences = [bert_conf, roberta_conf]
    final_pred_id = predictions[np.argmax(confidences)] if max(confidences) > 0 else -1
    final_pred = id2label.get(final_pred_id, "Unknown")
    write_to_file(output_file, f"Clause: '{clause}'\n", 'a')
    write_to_file(output_file, f"Legal-BERT Prediction: {id2label[bert_pred]} (Confidence: {bert_conf})\n", 'a')
    write_to_file(output_file, f"Legal-RoBERTa Prediction: {id2label[roberta_pred]} (Confidence: {roberta_conf})\n", 'a')
    write_to_file(output_file, f"Final Prediction: {final_pred}\n", 'a')
    write_to_file(output_file, "=" * 80 + "\n\n", 'a')
# Directory paths for contract files and output results
'''
for file_name in os.listdir(contract_directory_path):
    write_to_file(debug_file_path, f"We are going through the file {file_name}\n", 'a')

    if file_name.endswith(".txt"):
        file_path = os.path.join(contract_directory_path, file_name)
        output_file = os.path.join(output_directory, f"results_{file_name}")

        write_to_file(output_file, f"Results for {file_name}\n", 'a')

        with open(file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                write_to_file(debug_file_path, f"Stripped line: {stripped_line}\n\n", 'a')

                if not stripped_line or stripped_line == "***THIS IS AN EMPTY PARA***" or (stripped_line.startswith("Child") and stripped_line.endswith(":")):
                    continue

                # Process the valid line
                paragraph_text = stripped_line
                bert_pred, bert_conf = get_predictions(legal_bert_model, legal_bert_tokenizer, paragraph_text, device)
                roberta_pred, roberta_conf = get_predictions(legal_roberta_model, legal_roberta_tokenizer, paragraph_text, device)

                # Ensemble decision logic
                predictions = [bert_pred, roberta_pred]
                confidences = [bert_conf, roberta_conf]
                final_pred_id = predictions[np.argmax(confidences)] if max(confidences) > 0 else -1
                final_pred = id2label.get(final_pred_id, "Unknown")

                # Log results
                write_to_file(output_file, f"Clause: '{paragraph_text}'\n", 'a')
                write_to_file(output_file, f"Legal-BERT Prediction: {id2label[bert_pred]} (Confidence: {bert_conf})\n", 'a')
                write_to_file(output_file, f"Legal-RoBERTa Prediction: {id2label[roberta_pred]} (Confidence: {roberta_conf})\n", 'a')
                write_to_file(output_file, f"Final Prediction: {final_pred}\n", 'a')
                write_to_file(output_file, "=" * 80 + "\n\n", 'a')
'''
# cc_for_real.py

import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertTokenizer,
    RobertaTokenizer
)
from datasets import Dataset, DatasetDict
# from concurrent.futures import ThreadPoolExecutor  # Not needed if you're not using parallel code

# OPTIONAL: If you still see spawn/bootstrapping errors on macOS, try:
# import multiprocessing
# multiprocessing.set_start_method("fork")

def main():
    # Directory paths for contract files and output results
    contract_directory_path = './word_txt_outputs'
    output_directory = './'
    debug_file_path = "./debugging_cc_for_real.txt"

    # Helper function to write to a file in append mode
    def write_to_file(filename, text, mode='a'):
        with open(filename, mode) as file:
            file.write(text)

    # Clear the debug file if it exists
    if os.path.exists(debug_file_path):
        os.remove(debug_file_path)
        print(f"Deleted: {debug_file_path}")

    # Use Apple Silicon GPU if available, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Paths for fine-tuned models and tokenizers
    bert_model_path = "./fine_tuned_legal_bert"
    bert_tokenizer_path = "./fine_tuned_legal_bert_tokenizer"
    roberta_model_path = "./fine_tuned_legal_roberta"
    roberta_tokenizer_path = "./fine_tuned_legal_roberta_tokenizer"

    # Load dataset
    df1 = pd.read_csv("Clauses 1.csv")
    df2 = pd.read_csv("Clauses 2.csv")
    combined_df = pd.concat([df1, df2], ignore_index=True)

    label_column = 'Clause Heading'
    unique_labels = combined_df[label_column].unique()
    num_labels = len(unique_labels)

    # Build label mappings
    label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    # Check if the fine-tuned models exist
    bert_model_exists = os.path.exists(bert_model_path) and os.path.exists(bert_tokenizer_path)
    roberta_model_exists = os.path.exists(roberta_model_path) and os.path.exists(roberta_tokenizer_path)

    if not (bert_model_exists and roberta_model_exists):
        # We haven't fine-tuned yet, so let's train from scratch
        print("No fine-tuned models found; training from base models...")

        combined_df['labels'] = combined_df[label_column].map(label2id)

        # Split each label group 80/20 for train/val
        train_df_list, validation_df_list = [], []
        for label, group in combined_df.groupby(label_column):
            train_size = int(0.8 * len(group))
            train_df_list.append(group[:train_size])
            validation_df_list.append(group[train_size:])

        train_df = pd.concat(train_df_list).reset_index(drop=True)
        validation_df = pd.concat(validation_df_list).reset_index(drop=True)

        train_dataset = Dataset.from_pandas(train_df[['Clause Content', 'labels']])
        validation_dataset = Dataset.from_pandas(validation_df[['Clause Content', 'labels']])
        dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})

        # Load base models + tokenizers
        # (Ensure you have no local folder named 'nlpaueb/legal-bert-base-uncased')
        print("Loading base 'legal-bert-base-uncased'...")
        legal_bert_model = BertForSequenceClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            num_labels=num_labels
        ).to(device)

        print("Loading base 'saibo/legal-roberta-base'...")
        legal_roberta_model = RobertaForSequenceClassification.from_pretrained(
            "saibo/legal-roberta-base",
            num_labels=num_labels
        ).to(device)

        legal_bert_tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        legal_roberta_tokenizer = RobertaTokenizer.from_pretrained("saibo/legal-roberta-base")

        # Tokenization function
        def preprocess_function(examples, tokenizer):
            encoding = tokenizer(
                examples['Clause Content'],
                truncation=True,
                padding='max_length',
                max_length=512
            )
            encoding["labels"] = examples["labels"]
            return encoding

        bert_dataset = dataset.map(lambda x: preprocess_function(x, legal_bert_tokenizer), batched=True)
        roberta_dataset = dataset.map(lambda x: preprocess_function(x, legal_roberta_tokenizer), batched=True)

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        def train_model(model, dataset, tokenizer=None):
            from transformers import Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                tokenizer=tokenizer
            )
            trainer.train()
            return trainer

        print("Fine-tuning BERT model...")
        bert_trainer = train_model(legal_bert_model, bert_dataset, legal_bert_tokenizer)

        print("Fine-tuning RoBERTa model...")
        roberta_trainer = train_model(legal_roberta_model, roberta_dataset, legal_roberta_tokenizer)

        # Save fine-tuned models
        legal_bert_model.save_pretrained(bert_model_path)
        legal_roberta_model.save_pretrained(roberta_model_path)
        legal_bert_tokenizer.save_pretrained(bert_tokenizer_path)
        legal_roberta_tokenizer.save_pretrained(roberta_tokenizer_path)

    else:
        # Load the fine-tuned models for inference
        print("Loading fine-tuned BERT and RoBERTa models...")
        legal_bert_model = BertForSequenceClassification.from_pretrained(bert_model_path).to(device)
        legal_roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_path).to(device)
        legal_bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
        legal_roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_tokenizer_path)

    # Inference helpers
    def get_predictions(model, tokenizer, text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=512
        ).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        softmax = torch.nn.functional.softmax(logits, dim=-1)
        confidence, predicted_class = torch.max(softmax, dim=1)
        return predicted_class.item(), confidence.item()

    def extract_clauses_from_file(filepath, delimiter):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        raw_clauses = text.split(delimiter)
        clauses = [clause.strip() for clause in raw_clauses if clause.strip()]
        return clauses

    contract_file = "broken_down_contract.txt"
    delimiter_str = "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"

    # If you actually have "broken_down_contract.txt":
    if not os.path.exists(contract_file):
        print(f"Could not find {contract_file}, skipping clause extraction.")
    else:
        print("Extracting clauses and running predictions...")
        clauses_list = extract_clauses_from_file(contract_file, delimiter_str)
        output_file = os.path.join(output_directory, "results_clauses_list")

        for clause in clauses_list:
            bert_pred, bert_conf = get_predictions(legal_bert_model, legal_bert_tokenizer, clause)
            roberta_pred, roberta_conf = get_predictions(legal_roberta_model, legal_roberta_tokenizer, clause)

            # Simple ensemble
            predictions = [bert_pred, roberta_pred]
            confidences = [bert_conf, roberta_conf]
            final_pred_id = predictions[np.argmax(confidences)] if max(confidences) > 0 else -1
            final_pred = id2label.get(final_pred_id, "Unknown")

            write_to_file(output_file, f"Clause: '{clause}'\n")
            write_to_file(output_file, f"Legal-BERT: {id2label[bert_pred]} (conf: {bert_conf})\n")
            write_to_file(output_file, f"Legal-RoBERTa: {id2label[roberta_pred]} (conf: {roberta_conf})\n")
            write_to_file(output_file, f"Final Prediction: {final_pred}\n")
            write_to_file(output_file, "=" * 80 + "\n\n")
            '''
            Hi
            '''

    print("Done with the main() function!")

if __name__ == "__main__":
    main()
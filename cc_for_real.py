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

# -------------------------------------------------------------------------
# Helper functions to freeze layers (from your best trial with freeze=1)
# -------------------------------------------------------------------------
def freeze_bert_layers(model, freeze_until=0):
    """
    Freezes the first freeze_until encoder layers in a BERT model.
    freeze_until=0 => no freezing
    freeze_until=6 => freeze first 6 layers, etc.
    """
    for i, layer in enumerate(model.bert.encoder.layer):
        if i < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False

def freeze_roberta_layers(model, freeze_until=0):
    """
    Freezes the first freeze_until encoder layers in a RoBERTa model.
    freeze_until=0 => no freezing
    freeze_until=6 => freeze first 6 layers, etc.
    """
    for i, layer in enumerate(model.roberta.encoder.layer):
        if i < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False

# -------------------------------------------------------------------------
def main():
    # Directory paths for output results
    output_directory = './'
    debug_file_path = "./debugging_cc_for_real.txt"

    # Helper function to write to a file in append mode
    def write_to_file(filename, text, mode='a'):
        with open(filename, mode, encoding="utf-8") as file:
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

    label_column = 'Clause Heading'  # Name of the label column in your CSV
    unique_labels = combined_df[label_column].unique()
    num_labels = len(unique_labels)

    # Build label mappings
    label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    # Check if the fine-tuned models exist
    bert_model_exists = os.path.exists(bert_model_path) and os.path.exists(bert_tokenizer_path)
    roberta_model_exists = os.path.exists(roberta_model_path) and os.path.exists(roberta_tokenizer_path)

    # ------------------
    # TRAIN IF NECESSARY
    # ------------------
    if not (bert_model_exists and roberta_model_exists):
        print("No fine-tuned models found; training from base models...")

        # Map labels in the combined_df
        combined_df['labels'] = combined_df[label_column].map(label2id)

        # Split each label group 80/20 for train/val
        train_df_list, validation_df_list = [], []
        for label, group in combined_df.groupby(label_column):
            train_size = int(0.8 * len(group))
            train_df_list.append(group[:train_size])
            validation_df_list.append(group[train_size:])

        train_df = pd.concat(train_df_list).reset_index(drop=True)
        validation_df = pd.concat(validation_df_list).reset_index(drop=True)

        # Build Hugging Face datasets
        train_dataset = Dataset.from_pandas(train_df[['Clause Content', 'labels']])
        val_dataset = Dataset.from_pandas(validation_df[['Clause Content', 'labels']])
        dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

        # Load base models + tokenizers
        print("Loading base 'legal-bert-base-uncased'...")
        bert_model = BertForSequenceClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            num_labels=num_labels
        ).to(device)

        print("Loading base 'saibo/legal-roberta-base'...")
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            "saibo/legal-roberta-base",
            num_labels=num_labels
        ).to(device)

        # ----------------------------
        # Freeze the first 1 layer
        # ----------------------------
        freeze_bert_layers(bert_model, freeze_until=1)
        freeze_roberta_layers(roberta_model, freeze_until=1)

        bert_tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        roberta_tokenizer = RobertaTokenizer.from_pretrained("saibo/legal-roberta-base")

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

        bert_dataset = dataset.map(
            lambda x: preprocess_function(x, bert_tokenizer),
            batched=True
        )
        roberta_dataset = dataset.map(
            lambda x: preprocess_function(x, roberta_tokenizer),
            batched=True
        )

        # ----------------------------
        # Training Arguments
        # (Using best hyperparams)
        # ----------------------------
        from transformers import TrainingArguments, Trainer
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            # Best trial had ~2.64e-5
            learning_rate=2.64e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            # Best trial used 4 epochs
            num_train_epochs=4,
            weight_decay=0.01,
        )

        def train_model(model, dataset, tokenizer):
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
        bert_trainer = train_model(bert_model, bert_dataset, bert_tokenizer)

        print("Fine-tuning RoBERTa model...")
        roberta_trainer = train_model(roberta_model, roberta_dataset, roberta_tokenizer)

        # Save fine-tuned models
        bert_model.save_pretrained(bert_model_path)
        roberta_model.save_pretrained(roberta_model_path)
        bert_tokenizer.save_pretrained(bert_tokenizer_path)
        roberta_tokenizer.save_pretrained(roberta_tokenizer_path)

        # We'll keep these for inference
        legal_bert_model = bert_model
        legal_roberta_model = roberta_model
        legal_bert_tokenizer = bert_tokenizer
        legal_roberta_tokenizer = roberta_tokenizer

    else:
        # Load the fine-tuned models for inference
        print("Loading fine-tuned BERT and RoBERTa models...")
        legal_bert_model = BertForSequenceClassification.from_pretrained(bert_model_path).to(device)
        legal_roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_path).to(device)
        legal_bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
        legal_roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_tokenizer_path)

        # Build a train/validation split again for inference
        combined_df['labels'] = combined_df[label_column].map(label2id)
        train_df_list, validation_df_list = [], []
        for label, group in combined_df.groupby(label_column):
            train_size = int(0.8 * len(group))
            train_df_list.append(group[:train_size])
            validation_df_list.append(group[train_size:])

        train_df = pd.concat(train_df_list).reset_index(drop=True)
        validation_df = pd.concat(validation_df_list).reset_index(drop=True)

    # ----------------------------------------------------------------
    # Inference helper (get predictions for a single clause string)
    # ----------------------------------------------------------------
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

    # -----------------------------------
    # Evaluate on the validation set
    # -----------------------------------
    print("Running predictions on the validation set...")
    output_file = os.path.join(output_directory, "results_clauses_list_validation.txt")

    # Overwrite any previous file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== VALIDATION SET PREDICTIONS ===\n\n")

    total_clauses = len(validation_df)

    # We'll track how many are correct for each model + ensemble
    bert_correct_count = 0
    roberta_correct_count = 0
    ensemble_correct_count = 0

    for i, row in validation_df.iterrows():
        clause = row["Clause Content"]
        true_label_id = row["labels"]
        true_label_str = id2label[true_label_id]

        # BERT
        bert_pred_id, bert_conf = get_predictions(legal_bert_model, legal_bert_tokenizer, clause)
        bert_pred_str = id2label[bert_pred_id]
        is_bert_correct = (bert_pred_str == true_label_str)
        if is_bert_correct:
            bert_correct_count += 1

        # RoBERTa
        roberta_pred_id, roberta_conf = get_predictions(legal_roberta_model, legal_roberta_tokenizer, clause)
        roberta_pred_str = id2label[roberta_pred_id]
        is_roberta_correct = (roberta_pred_str == true_label_str)
        if is_roberta_correct:
            roberta_correct_count += 1

        # Ensemble (confidence-based)
        predictions = [bert_pred_id, roberta_pred_id]
        confidences = [bert_conf, roberta_conf]
        final_pred_id = predictions[np.argmax(confidences)]
        final_pred_str = id2label[final_pred_id]
        is_ensemble_correct = (final_pred_str == true_label_str)
        if is_ensemble_correct:
            ensemble_correct_count += 1

        # Write results
        write_to_file(output_file, f"Clause #{i}:\n")
        write_to_file(output_file, f"Clause Text: {clause}\n")
        write_to_file(output_file, f"True Label: {true_label_str}\n")
        write_to_file(output_file, f"BERT Prediction: {bert_pred_str} (conf: {bert_conf:.4f}), Correct? {is_bert_correct}\n")
        write_to_file(output_file, f"RoBERTa Prediction: {roberta_pred_str} (conf: {roberta_conf:.4f}), Correct? {is_roberta_correct}\n")
        write_to_file(output_file, f"Ensemble Prediction: {final_pred_str}, Correct? {is_ensemble_correct}\n")
        write_to_file(output_file, "=" * 80 + "\n\n")

    # Accuracy calculations
    bert_accuracy = 100.0 * bert_correct_count / total_clauses if total_clauses else 0.0
    roberta_accuracy = 100.0 * roberta_correct_count / total_clauses if total_clauses else 0.0
    ensemble_accuracy = 100.0 * ensemble_correct_count / total_clauses if total_clauses else 0.0

    summary = (
        f"BERT Accuracy: {bert_correct_count}/{total_clauses} = {bert_accuracy:.2f}%\n"
        f"RoBERTa Accuracy: {roberta_correct_count}/{total_clauses} = {roberta_accuracy:.2f}%\n"
        f"Ensemble Accuracy: {ensemble_correct_count}/{total_clauses} = {ensemble_accuracy:.2f}%\n"
    )

    write_to_file(output_file, summary)
    print(summary)
    print("Done with the main() function!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

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
    RobertaTokenizer,
)
from datasets import Dataset, DatasetDict

# -------------------------------------------------------------------------
# Helper functions to freeze layers
# -------------------------------------------------------------------------
def freeze_bert_layers(model, freeze_until=0):
    """Freezes the first `freeze_until` encoder layers in a BERT model."""
    for i, layer in enumerate(model.bert.encoder.layer):
        if i < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False

def freeze_roberta_layers(model, freeze_until=0):
    """Freezes the first `freeze_until` encoder layers in a RoBERTa model."""
    for i, layer in enumerate(model.roberta.encoder.layer):
        if i < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False

# -------------------------------------------------------------------------
# Custom TrainerCallback to log more info each epoch
# -------------------------------------------------------------------------
from transformers import TrainerCallback, TrainerState, TrainerControl

class ExtendedLoggingCallback(TrainerCallback):
    """
    Logs epoch-by-epoch val loss, val accuracy, LR, etc.
    Also can log gradient norms on_log if needed.
    """
    def __init__(self, logfile):
        super().__init__()
        self.logfile = logfile  # file path for logging

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called after each logging step."""
        if logs is None:
            return
        epoch = state.epoch or 0
        step = state.global_step
        with open(self.logfile, "a", encoding="utf-8") as f:
            f.write(f"[Step={step}, Epoch={epoch:.2f}] {logs}\n")

# -------------------------------------------------------------------------
def main():
    """
    1) Read the two CSV files (Clauses 1.csv, Clauses 2.csv).
    2) Train from scratch using base Legal-BERT and Legal-RoBERTa with the EXACT hyperparams:
       - BERT => lr=2.64e-5, epochs=10, freeze=1
       - RoBERTa => lr=9e-5, epochs=4, freeze=1
    3) Evaluate on an 80/20 label-wise split, log predictions to validation_predictions.txt.
    4) (Optional) Evaluate on real contracts in ./processed_contracts.
    """
    # Basic file/log setup
    output_directory = './'
    debug_file_path = "./debugging_cc_for_real.txt"
    training_log_file = "./training_log.txt"

    # Remove old logs if desired
    if os.path.exists(training_log_file):
        os.remove(training_log_file)
    if os.path.exists(debug_file_path):
        os.remove(debug_file_path)
        print(f"Deleted: {debug_file_path}")

    # Device choice
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # CSV paths
    csv1_path = "Clauses 1.csv"
    csv2_path = "Clauses 2.csv"
    if not os.path.exists(csv1_path):
        raise FileNotFoundError(f"File not found: {csv1_path}")
    if not os.path.exists(csv2_path):
        raise FileNotFoundError(f"File not found: {csv2_path}")

    # Load CSVs and combine
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    combined_df = pd.concat([df1, df2], ignore_index=True)

    label_column = "Clause Heading"
    text_column = "Clause Content"
    if label_column not in combined_df.columns or text_column not in combined_df.columns:
        raise ValueError(
            f"CSV must have columns '{label_column}' and '{text_column}'. "
            f"Found columns: {list(combined_df.columns)}"
        )

    # Build label set
    unique_labels = combined_df[label_column].unique()
    num_labels = len(unique_labels)
    print(f"Found {num_labels} distinct labels in the CSV data.")

    label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    # Convert the label column to IDs
    combined_df['labels'] = combined_df[label_column].map(label2id)

    # --------------------------------------------------------------------
    # Build an 80/20 split for each label group (so each label is represented)
    # --------------------------------------------------------------------
    train_df_list, validation_df_list = [], []
    for label, group in combined_df.groupby(label_column):
        split_size = int(0.8 * len(group))
        train_df_list.append(group[:split_size])
        validation_df_list.append(group[split_size:])

    train_df = pd.concat(train_df_list).reset_index(drop=True)
    validation_df = pd.concat(validation_df_list).reset_index(drop=True)

    from datasets import DatasetDict, Dataset
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df[[text_column, 'labels']]),
        "validation": Dataset.from_pandas(validation_df[[text_column, 'labels']])
    })

    print(f"Train set size: {len(train_df)}, Validation set size: {len(validation_df)}")

    # --------------------------------------------------------------------
    # Load base 'legal-bert' and 'legal-roberta'
    # --------------------------------------------------------------------
    print("Loading base 'nlpaueb/legal-bert-base-uncased'...")
    bert_model = BertForSequenceClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    print("Loading base 'saibo/legal-roberta-base'...")
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        "saibo/legal-roberta-base",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Freeze=1 layer each
    freeze_bert_layers(bert_model, freeze_until=1)
    freeze_roberta_layers(roberta_model, freeze_until=1)

    # Tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    roberta_tokenizer = RobertaTokenizer.from_pretrained("saibo/legal-roberta-base")

    # Preprocessing
    def preprocess_fn(batch, tokenizer):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding='max_length',
            max_length=512
        )

    bert_dataset = dataset.map(lambda x: preprocess_fn(x, bert_tokenizer), batched=True)
    roberta_dataset = dataset.map(lambda x: preprocess_fn(x, roberta_tokenizer), batched=True)

    # --------------------------------------------------------------------
    # Training arguments (same as you originally had)
    # --------------------------------------------------------------------
    from transformers import TrainingArguments, Trainer

    # BERT => lr=2.64e-5, epochs=10
    bert_training_args = TrainingArguments(
        output_dir="./bert_results",
        evaluation_strategy="epoch",
        learning_rate=2.64e-5,     # as requested
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,       # as requested
        weight_decay=0.01,
        logging_dir="./logs_bert",
        logging_steps=50,
        report_to=[],  # no HF logging
    )

    # RoBERTa => lr=9e-5, epochs=4
    roberta_training_args = TrainingArguments(
        output_dir="./roberta_results",
        evaluation_strategy="epoch",
        learning_rate=9e-5,        # as requested
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,        # as requested
        weight_decay=0.01,
        logging_dir="./logs_roberta",
        logging_steps=50,
        report_to=[],  # no HF logging
    )

    ext_logger = ExtendedLoggingCallback(training_log_file)

    # --------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------
    def train_bert(model, dset):
        trainer = Trainer(
            model=model,
            args=bert_training_args,
            train_dataset=dset["train"],
            eval_dataset=dset["validation"],
            tokenizer=bert_tokenizer,
            callbacks=[ext_logger],
        )
        trainer.train()
        return trainer

    def train_roberta(model, dset):
        trainer = Trainer(
            model=model,
            args=roberta_training_args,
            train_dataset=dset["train"],
            eval_dataset=dset["validation"],
            tokenizer=roberta_tokenizer,
            callbacks=[ext_logger],
        )
        trainer.train()
        return trainer

    print("\n=== Fine-tuning BERT (lr=2.64e-5, epochs=10, freeze=1) ===")
    bert_trainer = train_bert(bert_model, bert_dataset)

    print("\n=== Fine-tuning RoBERTa (lr=9e-5, epochs=4, freeze=1) ===")
    roberta_trainer = train_roberta(roberta_model, roberta_dataset)

    # --------------------------------------------------------------------
    # Save fine-tuned models & tokenizers
    # --------------------------------------------------------------------
    bert_model_path = "./fine_tuned_legal_bert"
    bert_tokenizer_path = "./fine_tuned_legal_bert_tokenizer"
    roberta_model_path = "./fine_tuned_legal_roberta"
    roberta_tokenizer_path = "./fine_tuned_legal_roberta_tokenizer"

    print(f"\nSaving BERT to {bert_model_path}...")
    bert_model.save_pretrained(bert_model_path)
    bert_tokenizer.save_pretrained(bert_tokenizer_path)

    print(f"Saving RoBERTa to {roberta_model_path}...")
    roberta_model.save_pretrained(roberta_model_path)
    roberta_tokenizer.save_pretrained(roberta_tokenizer_path)

    # --------------------------------------------------------------------
    # Evaluate on Validation
    # --------------------------------------------------------------------
    def get_predictions(model, tokenizer, text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            softmax_vals = torch.nn.functional.softmax(logits, dim=-1)
            conf, pred = torch.max(softmax_vals, dim=1)
        return pred.item(), conf.item()

    val_log_file = os.path.join(output_directory, "validation_predictions.txt")
    with open(val_log_file, "w", encoding="utf-8") as f:
        f.write("=== VALIDATION SET DETAILED PREDICTIONS ===\n\n")

    total_val = len(validation_df)
    bert_correct = 0
    roberta_correct = 0
    ensemble_correct = 0

    def write_to_file(filename, text, mode='a'):
        with open(filename, mode, encoding="utf-8") as file:
            file.write(text)

    for idx, row in validation_df.iterrows():
        clause_text = row[text_column]
        true_id = row["labels"]
        true_label_str = id2label[true_id]

        # BERT
        bert_pred_id, bert_conf = get_predictions(bert_model, bert_tokenizer, clause_text)
        bert_pred_str = id2label[bert_pred_id]
        is_bert_correct = (bert_pred_str == true_label_str)
        if is_bert_correct:
            bert_correct += 1

        # RoBERTa
        roberta_pred_id, roberta_conf = get_predictions(roberta_model, roberta_tokenizer, clause_text)
        roberta_pred_str = id2label[roberta_pred_id]
        is_roberta_correct = (roberta_pred_str == true_label_str)
        if is_roberta_correct:
            roberta_correct += 1

        # Confidence-based ensemble
        predictions = [bert_pred_id, roberta_pred_id]
        confidences = [bert_conf, roberta_conf]
        final_pred_id = predictions[np.argmax(confidences)]
        final_pred_str = id2label[final_pred_id]
        is_ensemble_correct = (final_pred_str == true_label_str)
        if is_ensemble_correct:
            ensemble_correct += 1

        # Log
        write_to_file(val_log_file, f"Clause #{idx}\n")
        write_to_file(val_log_file, f"Clause Text:\n{clause_text}\n")
        write_to_file(val_log_file, f"** True Label: {true_label_str} **\n")
        write_to_file(val_log_file, f"BERT => {bert_pred_str} (conf={bert_conf:.4f}), Correct? {is_bert_correct}\n")
        write_to_file(val_log_file, f"RoBERTa => {roberta_pred_str} (conf={roberta_conf:.4f}), Correct? {is_roberta_correct}\n")
        write_to_file(val_log_file, f"Ensemble => {final_pred_str}, Correct? {is_ensemble_correct}\n")
        write_to_file(val_log_file, "=" * 80 + "\n\n")

    # Validation accuracy
    bert_acc = 100.0 * bert_correct / total_val if total_val > 0 else 0.0
    roberta_acc = 100.0 * roberta_correct / total_val if total_val > 0 else 0.0
    ensemble_acc = 100.0 * ensemble_correct / total_val if total_val > 0 else 0.0

    summary = (
        f"BERT Accuracy: {bert_correct}/{total_val} = {bert_acc:.2f}%\n"
        f"RoBERTa Accuracy: {roberta_correct}/{total_val} = {roberta_acc:.2f}%\n"
        f"Ensemble Accuracy: {ensemble_correct}/{total_val} = {ensemble_acc:.2f}%\n"
    )
    print(summary)
    write_to_file(val_log_file, "\n=== VALIDATION ACCURACY ===\n")
    write_to_file(val_log_file, summary)

    # --------------------------------------------------------------------
    # (Optional) Evaluate on real contracts in ./processed_contracts
    # --------------------------------------------------------------------
    processed_contracts_dir = "./processed_contracts"
    delimiter = "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"

    if os.path.exists(processed_contracts_dir):
        real_log_file = os.path.join(output_directory, "real_contracts_predictions.txt")
        with open(real_log_file, "w", encoding="utf-8") as f:
            f.write("=== REAL CONTRACTS PREDICTIONS ===\n\n")

        def write_real_log(msg):
            with open(real_log_file, "a", encoding="utf-8") as f:
                f.write(msg)

        for contract_dir in os.listdir(processed_contracts_dir):
            contract_path = os.path.join(processed_contracts_dir, contract_dir)
            if os.path.isdir(contract_path):
                contract_name = contract_dir
                for fname in os.listdir(contract_path):
                    if fname.endswith("_broken_down.txt"):
                        broken_down_file = os.path.join(contract_path, fname)

                        with open(broken_down_file, "r", encoding="utf-8") as fd:
                            content = fd.read()

                        clauses = content.split(delimiter)
                        for i, clause_text in enumerate(clauses):
                            clause_text = clause_text.strip()
                            if not clause_text:
                                continue

                            # BERT
                            bert_pred_id, bert_conf = get_predictions(bert_model, bert_tokenizer, clause_text)
                            bert_pred_str = id2label.get(bert_pred_id, "UNKNOWN")

                            # RoBERTa
                            roberta_pred_id, roberta_conf = get_predictions(roberta_model, roberta_tokenizer, clause_text)
                            roberta_pred_str = id2label.get(roberta_pred_id, "UNKNOWN")

                            # Confidence-based
                            if bert_conf <= 0.50 and roberta_conf <= 0.50:
                                final_pred_str = "UNKNOWN"
                            elif bert_conf > 0.50 and roberta_conf > 0.50:
                                final_pred_str = bert_pred_str if bert_conf > roberta_conf else roberta_pred_str
                            elif bert_conf > 0.50:
                                final_pred_str = bert_pred_str
                            elif roberta_conf > 0.50:
                                final_pred_str = roberta_pred_str
                            else:
                                final_pred_str = "UNKNOWN"

                            write_real_log(f"Contract Name: {contract_name}\n")
                            write_real_log(f"Clause #{i}\n")
                            write_real_log(f"Clause Text:\n{clause_text}\n")
                            write_real_log(f"BERT Prediction: {bert_pred_str} (conf: {bert_conf:.4f})\n")
                            write_real_log(f"RoBERTa Prediction: {roberta_pred_str} (conf: {roberta_conf:.4f})\n")
                            write_real_log(f"Ensemble Prediction: {final_pred_str}\n")
                            write_real_log("=" * 80 + "\n\n")
    else:
        print(f"Directory not found: {processed_contracts_dir}")

    print("\nAll done! Validation logs are in 'validation_predictions.txt', "
          "and real-contract predictions (if any) are in 'real_contracts_predictions.txt'.\n")

if __name__ == "__main__":
    main()
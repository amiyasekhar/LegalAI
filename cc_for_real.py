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
        """Called after each logging step. We can log epoch, loss, val_acc, LR, etc."""
        if logs is None:
            return
        epoch = state.epoch or 0
        step = state.global_step
        with open(self.logfile, "a", encoding="utf-8") as f:
            f.write(f"[Step={step}, Epoch={epoch:.2f}] {logs}\n")

# -------------------------------------------------------------------------
def main():
    """
    We'll train (if not already trained) and then evaluate our fine-tuned 
    BERT + RoBERTa on the validation set, logging each clause's predictions.
    """
    output_directory = './'
    debug_file_path = "./debugging_cc_for_real.txt"

    # If you want to capture detailed training logs:
    training_log_file = "./training_log.txt"
    if os.path.exists(training_log_file):
        os.remove(training_log_file)

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

    # Directories for saving fine-tuned models
    bert_model_path = "./fine_tuned_legal_bert"
    bert_tokenizer_path = "./fine_tuned_legal_bert_tokenizer"
    roberta_model_path = "./fine_tuned_legal_roberta"
    roberta_tokenizer_path = "./fine_tuned_legal_roberta_tokenizer"

    # Load CSV data
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

    # ===========================
    # TRAIN IF NECESSARY
    # ===========================
    if not (bert_model_exists and roberta_model_exists):
        print("No fine-tuned models found; training from base models...")

        # Convert the label column to IDs
        combined_df['labels'] = combined_df[label_column].map(label2id)

        # Split each label group 80/20
        train_df_list, validation_df_list = [], []
        for label, group in combined_df.groupby(label_column):
            split_size = int(0.8 * len(group))
            train_df_list.append(group[:split_size])
            validation_df_list.append(group[split_size:])

        train_df = pd.concat(train_df_list).reset_index(drop=True)
        validation_df = pd.concat(validation_df_list).reset_index(drop=True)

        # Build a DatasetDict
        from datasets import DatasetDict, Dataset
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df[['Clause Content', 'labels']]),
            "validation": Dataset.from_pandas(validation_df[['Clause Content', 'labels']])
        })

        # Load base models
        print("Loading base 'legal-bert-base-uncased'...")
        from transformers import BertForSequenceClassification
        bert_model = BertForSequenceClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            num_labels=num_labels
        ).to(device)

        print("Loading base 'saibo/legal-roberta-base'...")
        from transformers import RobertaForSequenceClassification
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            "saibo/legal-roberta-base",
            num_labels=num_labels
        ).to(device)

        # Freeze=1 for each model
        freeze_bert_layers(bert_model, freeze_until=1)
        freeze_roberta_layers(roberta_model, freeze_until=1)

        # Tokenizers
        from transformers import BertTokenizer, RobertaTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        roberta_tokenizer = RobertaTokenizer.from_pretrained("saibo/legal-roberta-base")

        def preprocess_fn(examples, tokenizer):
            return tokenizer(
                examples['Clause Content'],
                truncation=True,
                padding='max_length',
                max_length=512
            )

        # Map the dataset
        bert_dataset = dataset.map(lambda x: preprocess_fn(x, bert_tokenizer), batched=True)
        roberta_dataset = dataset.map(lambda x: preprocess_fn(x, roberta_tokenizer), batched=True)

        # ----------------------------
        # Train BERT with your best HP
        # BERT => lr=2.64e-5, epochs=10, freeze=1
        # ----------------------------
        from transformers import TrainingArguments, Trainer

        bert_training_args = TrainingArguments(
            output_dir="./bert_results",
            evaluation_strategy="epoch",
            learning_rate=2.64e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,  # BERT => 10 epochs
            weight_decay=0.01,
            logging_dir="./logs_bert",
            logging_steps=50,
            report_to=[],  # no HF logging
        )

        # Extended logging callback for BERT
        bert_ext_logger = ExtendedLoggingCallback(training_log_file)

        def train_bert(model, dset):
            trainer = Trainer(
                model=model,
                args=bert_training_args,
                train_dataset=dset["train"],
                eval_dataset=dset["validation"],
                tokenizer=bert_tokenizer,
                callbacks=[bert_ext_logger],
            )
            trainer.train()
            return trainer

        print("Fine-tuning BERT with epochs=10, lr=2.64e-5, freeze=1 ...")
        bert_trainer = train_bert(bert_model, bert_dataset)

        # Save the fine-tuned BERT
        bert_model.save_pretrained(bert_model_path)
        bert_tokenizer.save_pretrained(bert_tokenizer_path)

        # ----------------------------
        # Train RoBERTa with your best HP
        # RoBERTa => lr=9e-5, epochs=4, freeze=1
        # ----------------------------
        roberta_training_args = TrainingArguments(
            output_dir="./roberta_results",
            evaluation_strategy="epoch",
            learning_rate=9e-5,  # higher LR
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=4,  # fewer epochs
            weight_decay=0.01,
            logging_dir="./logs_roberta",
            logging_steps=50,
            report_to=[],  # no HF logging
        )

        roberta_ext_logger = ExtendedLoggingCallback(training_log_file)

        def train_roberta(model, dset):
            trainer = Trainer(
                model=model,
                args=roberta_training_args,
                train_dataset=dset["train"],
                eval_dataset=dset["validation"],
                tokenizer=roberta_tokenizer,
                callbacks=[roberta_ext_logger],
            )
            trainer.train()
            return trainer

        print("Fine-tuning RoBERTa with epochs=4, lr=9e-5, freeze=1 ...")
        roberta_trainer = train_roberta(roberta_model, roberta_dataset)

        # Save the fine-tuned RoBERTa
        roberta_model.save_pretrained(roberta_model_path)
        roberta_tokenizer.save_pretrained(roberta_tokenizer_path)

        # We'll keep these loaded for inference
        legal_bert_model = bert_model
        legal_roberta_model = roberta_model
        legal_bert_tokenizer = bert_tokenizer
        legal_roberta_tokenizer = roberta_tokenizer

    else:
        # Models found, load for inference
        print("Loading existing fine-tuned BERT & RoBERTa...")
        from transformers import BertForSequenceClassification, RobertaForSequenceClassification
        from transformers import BertTokenizer, RobertaTokenizer

        legal_bert_model = BertForSequenceClassification.from_pretrained(bert_model_path).to(device)
        legal_bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)

        legal_roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_path).to(device)
        legal_roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_tokenizer_path)

        # (Re)-Build a train/validation split for inference
        combined_df['labels'] = combined_df[label_column].map(label2id)
        train_df_list, validation_df_list = [], []
        for label, group in combined_df.groupby(label_column):
            size = int(0.8 * len(group))
            train_df_list.append(group[:size])
            validation_df_list.append(group[size:])
        train_df = pd.concat(train_df_list).reset_index(drop=True)
        validation_df = pd.concat(validation_df_list).reset_index(drop=True)

    # -----------------------------------
    # Evaluate on the validation set
    # -----------------------------------
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
            softmax = torch.nn.functional.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(softmax, dim=1)
        return predicted_class.item(), confidence.item()

    print("\n--- Now evaluating on the validation set using your fine-tuned models ---\n")
    val_log_file = os.path.join(output_directory, "validation_predictions.txt")

    # Overwrite any previous file
    with open(val_log_file, "w", encoding="utf-8") as f:
        f.write("=== VALIDATION SET DETAILED PREDICTIONS ===\n\n")

    total_val_clauses = len(validation_df)

    # We'll track how many are correct for each model + ensemble
    bert_correct_count = 0
    roberta_correct_count = 0
    ensemble_correct_count = 0

    for idx, row in validation_df.iterrows():
        clause_text = row["Clause Content"]
        true_label_id = row["labels"]
        true_label_str = id2label[true_label_id]

        # BERT prediction
        bert_pred_id, bert_conf = get_predictions(legal_bert_model, legal_bert_tokenizer, clause_text)
        bert_pred_str = id2label[bert_pred_id]
        is_bert_correct = (bert_pred_str == true_label_str)
        if is_bert_correct:
            bert_correct_count += 1

        # RoBERTa prediction
        roberta_pred_id, roberta_conf = get_predictions(legal_roberta_model, legal_roberta_tokenizer, clause_text)
        roberta_pred_str = id2label[roberta_pred_id]
        is_roberta_correct = (roberta_pred_str == true_label_str)
        if is_roberta_correct:
            roberta_correct_count += 1

        # Confidence-based ensemble
        predictions = [bert_pred_id, roberta_pred_id]
        confidences = [bert_conf, roberta_conf]
        final_pred_id = predictions[np.argmax(confidences)]
        final_pred_str = id2label[final_pred_id]
        is_ensemble_correct = (final_pred_str == true_label_str)
        if is_ensemble_correct:
            ensemble_correct_count += 1

        # Log the predictions
        write_to_file(val_log_file, f"Clause #{idx}\n")
        write_to_file(val_log_file, f"Clause Text:\n{clause_text}\n")
        write_to_file(val_log_file, f"** True Label: {true_label_str} **\n")
        write_to_file(val_log_file, f"BERT => {bert_pred_str} (conf={bert_conf:.4f}), Correct? {is_bert_correct}\n")
        write_to_file(val_log_file, f"RoBERTa => {roberta_pred_str} (conf={roberta_conf:.4f}), Correct? {is_roberta_correct}\n")
        write_to_file(val_log_file, f"Ensemble => {final_pred_str}, Correct? {is_ensemble_correct}\n")
        write_to_file(val_log_file, "=" * 80 + "\n\n")

    # Final accuracy
    bert_val_acc = 100.0 * bert_correct_count / total_val_clauses if total_val_clauses else 0.0
    roberta_val_acc = 100.0 * roberta_correct_count / total_val_clauses if total_val_clauses else 0.0
    ensemble_val_acc = 100.0 * ensemble_correct_count / total_val_clauses if total_val_clauses else 0.0

    print(f"\n=== Validation Accuracy ===\n")
    write_to_file(val_log_file, f"\n=== Validation Accuracy ===\n")
    print(f"BERT: {bert_correct_count}/{total_val_clauses} = {bert_val_acc:.2f}%")
    write_to_file(val_log_file, f"BERT: {bert_correct_count}/{total_val_clauses} = {bert_val_acc:.2f}%\n")
    print(f"RoBERTa: {roberta_correct_count}/{total_val_clauses} = {roberta_val_acc:.2f}%")
    write_to_file(val_log_file, f"RoBERTa: {roberta_correct_count}/{total_val_clauses} = {roberta_val_acc:.2f}%\n")
    print(f"Ensemble: {ensemble_correct_count}/{total_val_clauses} = {ensemble_val_acc:.2f}%")
    write_to_file(val_log_file, f"Ensemble: {ensemble_correct_count}/{total_val_clauses} = {ensemble_val_acc:.2f}%\n")
    write_to_file(val_log_file, "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n\n")
    print(f"Detailed predictions (including true label and each model's guess) are logged to '{val_log_file}'")
    write_to_file(val_log_file, "********TESTING ON REAL CONTRACTS NOW********\n\n")

    # Root directory for processed contracts
    processed_contracts_dir = "./processed_contracts"
    # Delimiter used in the *_broken_down.txt files
    delimiter = "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"

    # Traverse each contract directory in ./processed_contracts
    if os.path.exists(processed_contracts_dir):
        for contract_dir in os.listdir(processed_contracts_dir):
            contract_path = os.path.join(processed_contracts_dir, contract_dir)
            if os.path.isdir(contract_path):
                # (Optional) If you want just the folder name without extension:
                # contract_name = contract_dir.replace(".Contract", "")
                contract_name = contract_dir  # or parse as needed

                # Find the _broken_down.txt file
                for fname in os.listdir(contract_path):
                    if fname.endswith("_broken_down.txt"):
                        broken_down_file = os.path.join(contract_path, fname)

                        with open(broken_down_file, "r", encoding="utf-8") as fd:
                            content = fd.read()

                        # Split by the delimiter to get individual clauses
                        clauses = content.split(delimiter)

                        # For each clause, get predictions
                        for i, clause_text in enumerate(clauses):
                            clause_text = clause_text.strip()
                            if not clause_text:
                                continue

                            # Run BERT predictions
                            bert_pred_id, bert_conf = get_predictions(
                                legal_bert_model, legal_bert_tokenizer, clause_text
                            )
                            bert_pred_str = id2label.get(bert_pred_id, "UNKNOWN")

                            # Run RoBERTa predictions
                            roberta_pred_id, roberta_conf = get_predictions(
                                legal_roberta_model, legal_roberta_tokenizer, clause_text
                            )
                            roberta_pred_str = id2label.get(roberta_pred_id, "UNKNOWN")

                            # Confidence-based ensemble
                            '''
                            predictions = [bert_pred_id, roberta_pred_id]
                            confidences = [bert_conf, roberta_conf]
                            final_pred_id = predictions[np.argmax(confidences)]
                            final_pred_str = id2label.get(final_pred_id, "UNKNOWN")
                            '''
                            
                            # Custom logic
                            if bert_conf <= 0.50 and roberta_conf <= 0.50:
                                # Both models are at or below 50% confidence => UNKNOWN
                                final_pred_str = "UNKNOWN"
                            elif bert_conf > 0.50 and roberta_conf > 0.50:
                                # Both models have > 50% confidence
                                # => pick the model with the higher confidence
                                if bert_conf > roberta_conf:
                                    final_pred_str = id2label.get(bert_pred_id, "UNKNOWN")
                                else:
                                    final_pred_str = id2label.get(roberta_pred_id, "UNKNOWN")
                            elif bert_conf > 0.50:
                                # Only BERT has > 50% confidence
                                final_pred_str = id2label.get(bert_pred_id, "UNKNOWN")
                            elif roberta_conf > 0.50:
                                # Only RoBERTa has > 50% confidence
                                final_pred_str = id2label.get(roberta_pred_id, "UNKNOWN")
                            else:
                                # Fallback
                                final_pred_str = "UNKNOWN"

                            # Log results to real_test.txt, including contract name
                            write_to_file(val_log_file, f"Contract Name: {contract_name}\n")
                            write_to_file(val_log_file, f"Clause #{i}\n")
                            write_to_file(val_log_file, f"Clause Text:\n{clause_text}\n")
                            write_to_file(val_log_file, f"BERT Prediction: {bert_pred_str} (conf: {bert_conf:.4f})\n")
                            write_to_file(val_log_file, f"RoBERTa Prediction: {roberta_pred_str} (conf: {roberta_conf:.4f})\n")
                            write_to_file(val_log_file, f"Ensemble Prediction: {final_pred_str}\n")
                            write_to_file(val_log_file, "=" * 80 + "\n\n")
    else:
        print(f"Directory not found: {processed_contracts_dir}")
if __name__ == "__main__":
    main()
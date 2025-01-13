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
# Helper functions to expand classifier heads
# -------------------------------------------------------------------------
import torch.nn as nn

def expand_bert_classifier(model, new_labels):
    """
    Expands the final classification layer in a BertForSequenceClassification
    to accommodate new labels, preserving old weights and biases.
    """
    old_label2id = model.config.label2id
    old_id2label = model.config.id2label

    # Count how many labels we had
    old_num_labels = len(old_label2id)
    hidden_size = model.classifier.in_features  # BERT's final layer is model.classifier (nn.Linear)

    # Expand label2id/id2label
    start_index = old_num_labels
    for lbl in new_labels:
        old_label2id[lbl] = start_index
        start_index += 1

    # Build updated id2label
    new_id2label = {idx: lbl for lbl, idx in old_label2id.items()}
    model.config.label2id = old_label2id
    model.config.id2label = new_id2label

    # Create a new classifier layer
    new_num_labels = old_num_labels + len(new_labels)
    old_classifier = model.classifier  # This is nn.Linear
    W_old = old_classifier.weight.data
    b_old = old_classifier.bias.data

    new_classifier = nn.Linear(hidden_size, new_num_labels)
    # Copy old weights/bias into new
    with torch.no_grad():
        new_classifier.weight[:old_num_labels, :] = W_old
        new_classifier.bias[:old_num_labels] = b_old

    model.classifier = new_classifier


def expand_roberta_classifier(model, new_labels):
    """
    Expands the final classification layer in a RobertaForSequenceClassification
    to accommodate new labels, preserving old weights and biases.
    By default in HF, the final Linear is model.classifier.out_proj.
    """
    old_label2id = model.config.label2id
    old_id2label = model.config.id2label

    # Count how many labels we had
    old_num_labels = len(old_label2id)
    # For RoBERTa, the final linear layer is model.classifier.out_proj
    old_classifier = model.classifier.out_proj
    hidden_size = old_classifier.in_features

    # Expand label2id/id2label
    start_index = old_num_labels
    for lbl in new_labels:
        old_label2id[lbl] = start_index
        start_index += 1

    # Build updated id2label
    new_id2label = {idx: lbl for lbl, idx in old_label2id.items()}
    model.config.label2id = old_label2id
    model.config.id2label = new_id2label

    # Create a new classifier layer
    new_num_labels = old_num_labels + len(new_labels)
    W_old = old_classifier.weight.data
    b_old = old_classifier.bias.data

    new_out_proj = nn.Linear(hidden_size, new_num_labels)
    with torch.no_grad():
        new_out_proj.weight[:old_num_labels, :] = W_old
        new_out_proj.bias[:old_num_labels] = b_old

    model.classifier.out_proj = new_out_proj

# -------------------------------------------------------------------------
def main():
    """
    1) Loads existing fine-tuned BERT & RoBERTa.
    2) Expands their classifier heads with 4 new labels.
    3) Continues training them using the new Excel clauses.
    4) Versions the updated models to *_v2 folders.
    5) Builds an 80-20 split from the entire data for 'validation' & 'test'.
    6) Evaluates updated models on validation set, test set, and real contracts.
    7) Logs predictions & accuracies similarly to your existing script.
    """
    ############################################################################
    # Basic setup
    ############################################################################
    output_directory = './'
    debug_file_path = "./debugging_cc_for_real.txt"
    training_log_file = "./training_log.txt"

    # Clean old logs if desired
    if os.path.exists(training_log_file):
        os.remove(training_log_file)
    if os.path.exists(debug_file_path):
        os.remove(debug_file_path)
        print(f"Deleted: {debug_file_path}")

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Paths to your already fine-tuned models
    old_bert_model_path = "./fine_tuned_legal_bert"
    old_bert_tokenizer_path = "./fine_tuned_legal_bert_tokenizer"
    old_roberta_model_path = "./fine_tuned_legal_roberta"
    old_roberta_tokenizer_path = "./fine_tuned_legal_roberta_tokenizer"

    # Where we'll save the *updated* models (versioning)
    new_bert_model_path = "./fine_tuned_legal_bert_v2"
    new_bert_tokenizer_path = "./fine_tuned_legal_bert_tokenizer_v2"
    new_roberta_model_path = "./fine_tuned_legal_roberta_v2"
    new_roberta_tokenizer_path = "./fine_tuned_legal_roberta_tokenizer_v2"

    ############################################################################
    # 1) Load existing (already fine-tuned) BERT & RoBERTa
    ############################################################################
    if (not os.path.exists(old_bert_model_path)) or (not os.path.exists(old_roberta_model_path)):
        raise FileNotFoundError(
            "Could not find existing fine-tuned model folders. "
            "Make sure you have fine_tuned_legal_bert/ and fine_tuned_legal_roberta/ first."
        )

    print("Loading existing fine-tuned BERT & RoBERTa...")
    legal_bert_model = BertForSequenceClassification.from_pretrained(old_bert_model_path).to(device)
    legal_bert_tokenizer = BertTokenizer.from_pretrained(old_bert_tokenizer_path)

    legal_roberta_model = RobertaForSequenceClassification.from_pretrained(old_roberta_model_path).to(device)
    legal_roberta_tokenizer = RobertaTokenizer.from_pretrained(old_roberta_tokenizer_path)

    ############################################################################
    # 2) Expand classifier heads for new labels
    ############################################################################
    # Suppose your original model had N labels; now we add 4 new ones:
    new_labels = ["dispute resolution", "fee", "invoice", "price and payment"]

    # Expand BERT's final layer
    print("Expanding BERT classifier for new labels...")
    expand_bert_classifier(legal_bert_model, new_labels)

    # Expand RoBERTa's final layer
    print("Expanding RoBERTa classifier for new labels...")
    expand_roberta_classifier(legal_roberta_model, new_labels)

    # (Optional) Freeze layers if desired
    freeze_bert_layers(legal_bert_model, freeze_until=1)
    freeze_roberta_layers(legal_roberta_model, freeze_until=1)

    ############################################################################
    # 3) Read all data: Clauses 1.csv, Clauses 2.csv, and the new .xlsx
    ############################################################################
    df1 = pd.read_csv("Clauses 1.csv")
    df2 = pd.read_csv("Clauses 2.csv")
    df_xls = pd.read_excel("clause_content_variety_latest_clauses.xlsx")

    # Merge them all
    combined_df = pd.concat([df1, df2, df_xls], ignore_index=True)

    # Label column
    label_column = 'Clause Heading'
    text_column = 'Clause Content'

    # Collect all unique labels
    all_unique_labels = combined_df[label_column].unique()

    # Updated label2id from the model config
    # (Now includes the newly added 4 classes)
    label2id = legal_bert_model.config.label2id
    id2label = legal_bert_model.config.id2label

    # Check if any truly unknown labels exist in the data:
    # (Because we *did* just add 4 new ones, but if there are others not in label2id, that’s a problem)
    new_label_set = set(all_unique_labels) - set(label2id.keys())
    if new_label_set:
        raise ValueError(
            f"Found new label(s) that the expanded model doesn't support: {new_label_set}.\n"
            "You must handle them (add them to new_labels or re-check data)."
        )

    # Convert to IDs
    combined_df['labels'] = combined_df[label_column].map(label2id)

    ############################################################################
    # 4) Continue training on *just the XLSX* data (or combined, your choice)
    ############################################################################
    # Here we only fine-tune on the new Excel data again:
    df_train = df_xls.copy()
    df_train['labels'] = df_train[label_column].map(label2id)

    from datasets import Dataset

    def preprocess_fn_bert(examples):
        return legal_bert_tokenizer(
            examples[text_column],
            truncation=True,
            padding='max_length',
            max_length=512
        )

    def preprocess_fn_roberta(examples):
        return legal_roberta_tokenizer(
            examples[text_column],
            truncation=True,
            padding='max_length',
            max_length=512
        )

    ds_bert_train = Dataset.from_pandas(df_train[[text_column, 'labels']])
    ds_bert_train = ds_bert_train.map(preprocess_fn_bert, batched=True)

    ds_roberta_train = Dataset.from_pandas(df_train[[text_column, 'labels']])
    ds_roberta_train = ds_roberta_train.map(preprocess_fn_roberta, batched=True)

    # Setup training args
    bert_training_args = TrainingArguments(
        output_dir="./bert_results_v2",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs_bert_v2",
        logging_steps=50,
        report_to=[],
    )
    roberta_training_args = TrainingArguments(
        output_dir="./roberta_results_v2",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs_roberta_v2",
        logging_steps=50,
        report_to=[],
    )

    # Optional extended logger
    ext_logger = ExtendedLoggingCallback(training_log_file)

    # Train BERT
    trainer_bert = Trainer(
        model=legal_bert_model,
        args=bert_training_args,
        train_dataset=ds_bert_train,
        tokenizer=legal_bert_tokenizer,
        callbacks=[ext_logger],
    )
    print("\nContinuing training BERT on XLSX data...")
    trainer_bert.train()

    # Train RoBERTa
    trainer_roberta = Trainer(
        model=legal_roberta_model,
        args=roberta_training_args,
        train_dataset=ds_roberta_train,
        tokenizer=legal_roberta_tokenizer,
        callbacks=[ext_logger],
    )
    print("\nContinuing training RoBERTa on XLSX data...")
    trainer_roberta.train()

    # Save updated models to v2
    print("\nSaving updated BERT to fine_tuned_legal_bert_v2...")
    legal_bert_model.save_pretrained(new_bert_model_path)
    legal_bert_tokenizer.save_pretrained(new_bert_tokenizer_path)

    print("Saving updated RoBERTa to fine_tuned_legal_roberta_v2...")
    legal_roberta_model.save_pretrained(new_roberta_model_path)
    legal_roberta_tokenizer.save_pretrained(new_roberta_tokenizer_path)

    ############################################################################
    # 5) Build an 80-20 split (Val/Test) from *all* data, evaluate & log
    ############################################################################
    from sklearn.model_selection import train_test_split

    # Shuffle combined_df, then split
    combined_shuffled = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_size = int(0.8 * len(combined_shuffled))
    validation_df = combined_shuffled[:val_size].copy()
    test_df = combined_shuffled[val_size:].copy()

    print(f"Validation set: {len(validation_df)} clauses")
    print(f"Test set: {len(test_df)} clauses")

    # Reload your new v2 models for inference
    bert_v2 = BertForSequenceClassification.from_pretrained(new_bert_model_path).to(device)
    bert_tok_v2 = BertTokenizer.from_pretrained(new_bert_tokenizer_path)

    roberta_v2 = RobertaForSequenceClassification.from_pretrained(new_roberta_model_path).to(device)
    roberta_tok_v2 = RobertaTokenizer.from_pretrained(new_roberta_tokenizer_path)

    # For logging predictions
    val_log_file = os.path.join(output_directory, "validation_predictions.txt")
    test_log_file = os.path.join(output_directory, "test_predictions.txt")

    # Overwrite logs
    with open(val_log_file, "w", encoding="utf-8") as f:
        f.write("=== VALIDATION SET (80%) PREDICTIONS ===\n\n")
    with open(test_log_file, "w", encoding="utf-8") as f:
        f.write("=== TEST SET (20%) PREDICTIONS ===\n\n")

    def write_to_file(filename, text, mode='a'):
        with open(filename, mode, encoding="utf-8") as file:
            file.write(text)

    # Prediction helper
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

    # Evaluate function
    def evaluate_and_log(df, out_file):
        total = len(df)
        bert_correct = 0
        roberta_correct = 0
        ensemble_correct = 0

        for idx, row in df.iterrows():
            clause_text = row[text_column]
            true_label_id = row['labels']
            true_label_str = id2label[true_label_id]

            # BERT v2
            bert_pred_id, bert_conf = get_predictions(bert_v2, bert_tok_v2, clause_text)
            bert_pred_str = id2label[bert_pred_id]
            is_bert_correct = (bert_pred_str == true_label_str)
            if is_bert_correct:
                bert_correct += 1

            # RoBERTa v2
            roberta_pred_id, roberta_conf = get_predictions(roberta_v2, roberta_tok_v2, clause_text)
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
            write_to_file(out_file, f"Clause #{idx}\n")
            write_to_file(out_file, f"Clause Text:\n{clause_text}\n")
            write_to_file(out_file, f"** True Label: {true_label_str} **\n")
            write_to_file(out_file, f"BERT => {bert_pred_str} (conf={bert_conf:.4f}), Correct? {is_bert_correct}\n")
            write_to_file(out_file, f"RoBERTa => {roberta_pred_str} (conf={roberta_conf:.4f}), Correct? {is_roberta_correct}\n")
            write_to_file(out_file, f"Ensemble => {final_pred_str}, Correct? {is_ensemble_correct}\n")
            write_to_file(out_file, "=" * 80 + "\n\n")

        # Accuracy
        if total > 0:
            b_acc = 100.0 * bert_correct / total
            r_acc = 100.0 * roberta_correct / total
            e_acc = 100.0 * ensemble_correct / total
        else:
            b_acc = r_acc = e_acc = 0.0

        summary = (f"BERT Accuracy: {bert_correct}/{total} = {b_acc:.2f}%\n"
                   f"RoBERTa Accuracy: {roberta_correct}/{total} = {r_acc:.2f}%\n"
                   f"Ensemble Accuracy: {ensemble_correct}/{total} = {e_acc:.2f}%\n")
        write_to_file(out_file, summary)
        print(summary)

    # Evaluate on Validation
    print("\n=== Evaluating on Validation (80%) ===\n")
    evaluate_and_log(validation_df, val_log_file)

    # Evaluate on Test
    print("\n=== Evaluating on Test (20%) ===\n")
    evaluate_and_log(test_df, test_log_file)

    ############################################################################
    # 6) Evaluate on real contracts (same code as your original approach)
    ############################################################################
    real_log_file = os.path.join(output_directory, "real_contracts_predictions.txt")
    with open(real_log_file, "w", encoding="utf-8") as f:
        f.write("=== REAL CONTRACTS PREDICTIONS ===\n\n")

    def write_real_log(msg):
        with open(real_log_file, "a", encoding="utf-8") as f:
            f.write(msg)

    processed_contracts_dir = "./processed_contracts"
    delimiter = "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"

    if os.path.exists(processed_contracts_dir):
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
                            bert_pred_id, bert_conf = get_predictions(bert_v2, bert_tok_v2, clause_text)
                            bert_pred_str = id2label.get(bert_pred_id, "UNKNOWN")

                            # RoBERTa
                            roberta_pred_id, roberta_conf = get_predictions(roberta_v2, roberta_tok_v2, clause_text)
                            roberta_pred_str = id2label.get(roberta_pred_id, "UNKNOWN")

                            # Confidence-based
                            if bert_conf <= 0.50 and roberta_conf <= 0.50:
                                final_pred_str = "UNKNOWN"
                            elif bert_conf > 0.50 and roberta_conf > 0.50:
                                # If both are confident, pick the higher confidence
                                if bert_conf > roberta_conf:
                                    final_pred_str = bert_pred_str
                                else:
                                    final_pred_str = roberta_pred_str
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

    print("\nFinished! Validation results in 'validation_predictions.txt', "
          "test results in 'test_predictions.txt', "
          "and real-contract predictions in 'real_contracts_predictions.txt'.\n")

if __name__ == "__main__":
    main()
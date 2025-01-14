#!/usr/bin/env python3

import os
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertTokenizer,      # <--- Imported globally
    RobertaTokenizer,   # <--- Imported globally
    TrainerCallback,
    TrainerState,
    TrainerControl,
    default_data_collator
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------
# 1) Freeze Layers
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
# 2) Extended Logging Callback
# -------------------------------------------------------------------------
class ExtendedLoggingCallback(TrainerCallback):
    """
    Logs epoch-by-epoch training loss, LR, etc. to a file.
    """
    def __init__(self, logfile):
        super().__init__()
        self.logfile = logfile

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        epoch = state.epoch or 0
        step = state.global_step
        with open(self.logfile, "a", encoding="utf-8") as f:
            f.write(f"[Step={step}, Epoch={epoch:.2f}] {logs}\n")

# -------------------------------------------------------------------------
# 3) Expand BERT & RoBERTa Classifiers
# -------------------------------------------------------------------------
def expand_bert_classifier(model, new_labels):
    old_label2id = model.config.label2id
    old_num_labels = len(old_label2id)

    hidden_size = model.classifier.in_features
    old_classifier = model.classifier
    W_old = old_classifier.weight.data.clone()
    b_old = old_classifier.bias.data.clone()

    # Add new labels if missing
    start_index = old_num_labels
    for lbl in new_labels:
        if lbl not in old_label2id:
            old_label2id[lbl] = start_index
            start_index += 1

    new_num_labels = len(old_label2id)
    model.config.label2id = old_label2id
    model.config.id2label = {v: k for k, v in old_label2id.items()}
    model.config.num_labels = new_num_labels
    model.num_labels = new_num_labels

    new_classifier = nn.Linear(hidden_size, new_num_labels)
    with torch.no_grad():
        new_classifier.weight[:old_num_labels, :] = W_old
        new_classifier.bias[:old_num_labels] = b_old

    model.classifier = new_classifier

def expand_roberta_classifier(model, new_labels):
    old_label2id = model.config.label2id
    old_num_labels = len(old_label2id)

    out_proj = model.classifier.out_proj
    hidden_size = out_proj.in_features
    W_old = out_proj.weight.data.clone()
    b_old = out_proj.bias.data.clone()

    # Add new labels
    start_index = old_num_labels
    for lbl in new_labels:
        if lbl not in old_label2id:
            old_label2id[lbl] = start_index
            start_index += 1

    new_num_labels = len(old_label2id)
    model.config.label2id = old_label2id
    model.config.id2label = {v: k for k, v in old_label2id.items()}
    model.config.num_labels = new_num_labels
    model.num_labels = new_num_labels

    new_out_proj = nn.Linear(hidden_size, new_num_labels)
    with torch.no_grad():
        new_out_proj.weight[:old_num_labels, :] = W_old
        new_out_proj.bias[:old_num_labels] = b_old

    model.classifier.out_proj = new_out_proj

# -------------------------------------------------------------------------
# 4) Compute Fisher for EWC
# -------------------------------------------------------------------------
def compute_fisher_diagonal(model, dataloader, device):
    """
    Approx. diag(Fisher) for EWC. We'll do forward-backward for each batch,
    accumulate grad^2, then average over all batches.
    """
    model.eval()
    fisher = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            fisher[n] = torch.zeros_like(p.data)

    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to(device)

        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data**2

    # average
    for n in fisher:
        fisher[n] /= len(dataloader)

    model.train()
    return fisher

# -------------------------------------------------------------------------
# 5) Distill+EWC Trainer
# -------------------------------------------------------------------------
class DistillEWCTrainer(Trainer):
    """
    A custom Trainer that does partial-slice EWC + Distillation
    (only for old-labeled batches).
    """
    def __init__(
        self,
        old_model=None,
        fisher=None,
        ewc_lambda=500.0,
        distil_alpha=0.3,
        old_params=None,
        old_label_ids=None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.old_model = old_model
        self.fisher = fisher
        self.ewc_lambda = ewc_lambda
        self.distil_alpha = distil_alpha
        self.old_params = old_params if old_params else {}
        self.old_label_ids = set(old_label_ids) if old_label_ids else set()

        if old_model is not None and not self.old_params:
            for n, p in old_model.named_parameters():
                self.old_params[n] = p.data.clone()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch.nn.functional as F
        labels = inputs["labels"].to(model.device)
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        ce_loss = nn.CrossEntropyLoss()(logits, labels)

        # Distill only if entire batch is old-labeled
        unique_labels = labels.unique().cpu().tolist()
        do_distill = all(lbl_id in self.old_label_ids for lbl_id in unique_labels)

        distill_loss = 0.0
        if do_distill and self.old_model is not None:
            with torch.no_grad():
                old_out = self.old_model(**{k: v for k, v in inputs.items() if k != "labels"})
                old_logits = old_out.logits
            old_dim = old_logits.shape[1]
            student_logits_for_old = logits[:, :old_dim]

            T = 1.0
            new_log_probs = F.log_softmax(student_logits_for_old / T, dim=-1)
            old_probs = F.softmax(old_logits / T, dim=-1)
            kd = F.kl_div(new_log_probs, old_probs, reduction="batchmean") * (T**2)
            distill_loss = kd * self.distil_alpha

        # EWC penalty
        ewc_loss = 0.0
        if self.fisher and self.old_params:
            for n, p in model.named_parameters():
                if n not in self.fisher or not p.requires_grad:
                    continue
                old_p = self.old_params[n]
                fisher_p = self.fisher[n]
                if p.shape == old_p.shape:
                    ewc_loss += torch.sum(fisher_p * (p - old_p)**2)
                else:
                    # partial slice final layer
                    min_dims = [min(o, n_) for o, n_ in zip(old_p.shape, p.shape)]
                    slice_args = tuple(slice(0, md) for md in min_dims)
                    overlap_fisher = fisher_p[slice_args]
                    overlap_old = old_p[slice_args]
                    overlap_new = p[slice_args]
                    ewc_loss += torch.sum(overlap_fisher * (overlap_new - overlap_old)**2)
            ewc_loss *= self.ewc_lambda

        total_loss = ce_loss + distill_loss + ewc_loss
        if return_outputs:
            return (total_loss, outputs)
        else:
            return total_loss

# -------------------------------------------------------------------------
def evaluate_and_log(model_bert, model_roberta, ds_bert, ds_roberta, output_txt, set_name="VAL"):
    """
    Evaluate on a dataset => logs BERT, RoBERTa, and ensemble predictions
    *with the actual clause text* to a text file.
    """
    device = next(model_bert.parameters()).device
    total = len(ds_bert)
    if len(ds_bert) != len(ds_roberta):
        print(f"Warning: {set_name} ds_bert vs ds_roberta mismatch in length.")
    max_len = min(len(ds_bert), len(ds_roberta))

    bert_correct = 0
    roberta_correct = 0
    ensemble_correct = 0

    with open(output_txt, "a", encoding="utf-8") as f:
        f.write(f"\n=== {set_name} SET PREDICTIONS ===\n\n")

        for i in range(max_len):
            ex_b = ds_bert[i]
            ex_r = ds_roberta[i]

            # The dataset includes "Clause Content" text
            clause_text = ex_b["Clause Content"]  # from the original column
            true_label_id = ex_b["labels"].item()

            # Build batch
            input_b = {}
            input_r = {}
            for k, v in ex_b.items():
                if k not in ["labels", "Clause Content"]:
                    input_b[k] = v.unsqueeze(0).to(device)
            for k, v in ex_r.items():
                if k not in ["labels", "Clause Content"]:
                    input_r[k] = v.unsqueeze(0).to(device)

            # BERT forward
            with torch.no_grad():
                out_b = model_bert(**input_b)
                logits_b = out_b.logits
                probs_b = torch.softmax(logits_b, dim=-1)
                conf_b, pred_b = torch.max(probs_b, dim=-1)
            b_pred_id = pred_b.item()
            b_conf = conf_b.item()

            # RoBERTa forward
            with torch.no_grad():
                out_r = model_roberta(**input_r)
                logits_r = out_r.logits
                probs_r = torch.softmax(logits_r, dim=-1)
                conf_r, pred_r = torch.max(probs_r, dim=-1)
            r_pred_id = pred_r.item()
            r_conf = conf_r.item()

            # ensemble
            predictions = [b_pred_id, r_pred_id]
            confidences = [b_conf, r_conf]
            final_pred_id = predictions[torch.argmax(torch.tensor(confidences))]

            # Check correctness
            is_b_correct = (b_pred_id == true_label_id)
            is_r_correct = (r_pred_id == true_label_id)
            is_ensemble_correct = (final_pred_id == true_label_id)
            if is_b_correct:
                bert_correct += 1
            if is_r_correct:
                roberta_correct += 1
            if is_ensemble_correct:
                ensemble_correct += 1

            # Log
            f.write(f"Clause #{i}\n")
            f.write(f"Clause Text:\n{clause_text}\n")
            f.write(f"True Label ID: {true_label_id}\n")
            f.write(f"BERT => pred_id={b_pred_id}, conf={b_conf:.4f}, correct? {is_b_correct}\n")
            f.write(f"RoBERTa => pred_id={r_pred_id}, conf={r_conf:.4f}, correct? {is_r_correct}\n")
            f.write(f"Ensemble => pred_id={final_pred_id}, correct? {is_ensemble_correct}\n")
            f.write("=" * 80 + "\n\n")

    # Summaries
    b_acc = 100.0 * bert_correct / max_len if max_len else 0.0
    r_acc = 100.0 * roberta_correct / max_len if max_len else 0.0
    e_acc = 100.0 * ensemble_correct / max_len if max_len else 0.0

    print(f"[{set_name}] BERT Accuracy: {b_acc:.2f}%  ({bert_correct}/{max_len})")
    print(f"[{set_name}] RoBERTa Accuracy: {r_acc:.2f}%  ({roberta_correct}/{max_len})")
    print(f"[{set_name}] Ensemble Accuracy: {e_acc:.2f}%  ({ensemble_correct}/{max_len})")


def main():
    """
    1) 80/20 => train & val
    2) Distill+EWC partial-slice training
    3) Evaluate on validation => log predictions
    4) (For demo) reuse val as "test", log again
    5) Evaluate real contracts
    """

    debug_file = "./debugging_80_20.txt"
    training_log_file = "./training_log_80_20.txt"
    if os.path.exists(training_log_file):
        os.remove(training_log_file)
    if os.path.exists(debug_file):
        os.remove(debug_file)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    old_bert_path = "./fine_tuned_legal_bert"
    old_roberta_path = "./fine_tuned_legal_roberta"

    if not os.path.exists(old_bert_path) or not os.path.exists(old_roberta_path):
        raise FileNotFoundError("Could not find old teacher models: fine_tuned_legal_bert or fine_tuned_legal_roberta")

    print("Loading old teacher BERT & RoBERTa...")
    old_bert_teacher = BertForSequenceClassification.from_pretrained(old_bert_path).to(device)
    old_roberta_teacher = RobertaForSequenceClassification.from_pretrained(old_roberta_path).to(device)

    # Use the globally imported BERT & RoBERTa tokenizers
    old_bert_tokenizer = BertTokenizer.from_pretrained("./fine_tuned_legal_bert_tokenizer")
    old_roberta_tokenizer = RobertaTokenizer.from_pretrained("./fine_tuned_legal_roberta_tokenizer")

    # Student
    new_labels = ["dispute resolution", "fee", "invoice", "price and payment"]
    print("Cloning student & expanding for new labels...")
    bert_student = BertForSequenceClassification.from_pretrained(old_bert_path).to(device)
    expand_bert_classifier(bert_student, new_labels)

    roberta_student = RobertaForSequenceClassification.from_pretrained(old_roberta_path).to(device)
    expand_roberta_classifier(roberta_student, new_labels)

    freeze_bert_layers(bert_student, freeze_until=1)
    freeze_roberta_layers(roberta_student, freeze_until=1)

    # Load data
    df1 = pd.read_csv("Clauses 1.csv")
    df2 = pd.read_csv("Clauses 2.csv")
    df_xls = pd.read_excel("clause_content_variety_latest_clauses.xlsx")
    combined_df = pd.concat([df1, df2, df_xls], ignore_index=True)

    label_col = "Clause Heading"
    text_col = "Clause Content"

    all_labels = set(combined_df[label_col].unique())
    label2id = bert_student.config.label2id
    missing = all_labels - set(label2id.keys())
    if missing:
        raise ValueError(f"Found new label(s) not in expanded model: {missing}")
    combined_df["labels"] = combined_df[label_col].map(label2id)

    # 80/20
    train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    print(f"Train set: {len(train_df)}, Validation set: {len(val_df)}")

    from datasets import Dataset
    ds_train = Dataset.from_pandas(train_df)
    ds_val = Dataset.from_pandas(val_df)

    # Preprocessing
    def preprocess_bert(examples):
        tokenized = old_bert_tokenizer(
            examples[text_col],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        return {**examples, **tokenized}

    def preprocess_roberta(examples):
        tokenized = old_roberta_tokenizer(
            examples[text_col],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        return {**examples, **tokenized}

    ds_bert_train = ds_train.map(preprocess_bert, batched=True)
    ds_bert_val = ds_val.map(preprocess_bert, batched=True)
    ds_roberta_train = ds_train.map(preprocess_roberta, batched=True)
    ds_roberta_val = ds_val.map(preprocess_roberta, batched=True)

    # remove columns we don't need except 'labels' + 'Clause Content' + token keys
    remove_cols_bert = set(ds_bert_train.column_names) - set(
        ["labels", text_col, "input_ids", "attention_mask", "token_type_ids"]
    )
    remove_cols_roberta = set(ds_roberta_train.column_names) - set(
        ["labels", text_col, "input_ids", "attention_mask"]
    )

    ds_bert_train = ds_bert_train.remove_columns(list(remove_cols_bert))
    ds_bert_val = ds_bert_val.remove_columns(list(remove_cols_bert))
    ds_roberta_train = ds_roberta_train.remove_columns(list(remove_cols_roberta))
    ds_roberta_val = ds_roberta_val.remove_columns(list(remove_cols_roberta))

    ds_bert_train.set_format("torch")
    ds_bert_val.set_format("torch")
    ds_roberta_train.set_format("torch")
    ds_roberta_val.set_format("torch")

    # EWC => old-labeled data
    old_label2id = old_bert_teacher.config.label2id
    old_label_ids = set(old_label2id.values())

    def is_old_label(x):
        return x["labels"].item() in old_label_ids

    ds_bert_old = ds_bert_train.filter(is_old_label)
    ds_roberta_old = ds_roberta_train.filter(is_old_label)

    from torch.utils.data import DataLoader
    old_loader_bert = DataLoader(ds_bert_old, batch_size=8, shuffle=False, collate_fn=default_data_collator)
    old_loader_roberta = DataLoader(ds_roberta_old, batch_size=8, shuffle=False, collate_fn=default_data_collator)

    print("Computing EWC fisher for old BERT teacher on old-labeled data...")
    fisher_bert = compute_fisher_diagonal(old_bert_teacher, old_loader_bert, device)
    print("Computing EWC fisher for old RoBERTa teacher on old-labeled data...")
    fisher_roberta = compute_fisher_diagonal(old_roberta_teacher, old_loader_roberta, device)

    # Distill+EWC train
    from transformers import TrainingArguments

    bert_args = TrainingArguments(
        output_dir="./bert_distill_ewc_results_80_20",
        evaluation_strategy="no",
        learning_rate=2.64e-5,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir="./logs_bert_distill_ewc",
        logging_steps=50,
        report_to=[],
    )
    old_params_bert = {n: p.data.clone() for n, p in old_bert_teacher.named_parameters()}

    from transformers import Trainer
    bert_trainer = DistillEWCTrainer(
        model=bert_student,
        old_model=old_bert_teacher,
        fisher=fisher_bert,
        ewc_lambda=500.0,
        distil_alpha=0.3,
        old_params=old_params_bert,
        old_label_ids=old_label_ids,
        args=bert_args,
        train_dataset=ds_bert_train,
        tokenizer=old_bert_tokenizer,
        callbacks=[ExtendedLoggingCallback("./training_log_80_20.txt")]
    )
    print("\n--- Fine-tuning BERT (Distill+EWC) on 80% training set ---\n")
    bert_trainer.train()

    # Save
    new_bert_path = "./fine_tuned_legal_bert_v2"
    new_bert_tok_path = "./fine_tuned_legal_bert_tokenizer_v2"
    bert_student.save_pretrained(new_bert_path)
    old_bert_tokenizer.save_pretrained(new_bert_tok_path)

    roberta_args = TrainingArguments(
        output_dir="./roberta_distill_ewc_results_80_20",
        evaluation_strategy="no",
        learning_rate=9e-5,
        per_device_train_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_dir="./logs_roberta_distill_ewc",
        logging_steps=50,
        report_to=[],
    )
    old_params_roberta = {n: p.data.clone() for n, p in old_roberta_teacher.named_parameters()}

    roberta_trainer = DistillEWCTrainer(
        model=roberta_student,
        old_model=old_roberta_teacher,
        fisher=fisher_roberta,
        ewc_lambda=500.0,
        distil_alpha=0.3,
        old_params=old_params_roberta,
        old_label_ids=old_label_ids,
        args=roberta_args,
        train_dataset=ds_roberta_train,
        tokenizer=old_roberta_tokenizer,
        callbacks=[ExtendedLoggingCallback("./training_log_80_20.txt")]
    )
    print("\n--- Fine-tuning RoBERTa (Distill+EWC) on 80% training set ---\n")
    roberta_trainer.train()

    new_roberta_path = "./fine_tuned_legal_roberta_v2"
    new_roberta_tok_path = "./fine_tuned_legal_roberta_tokenizer_v2"
    roberta_student.save_pretrained(new_roberta_path)
    old_roberta_tokenizer.save_pretrained(new_roberta_tok_path)

    # Reload for inference
    bert_v2 = BertForSequenceClassification.from_pretrained(new_bert_path).to(device)
    roberta_v2 = RobertaForSequenceClassification.from_pretrained(new_roberta_path).to(device)

    val_and_test_logfile = "./val_and_test_predictions_80_20.txt"
    if os.path.exists(val_and_test_logfile):
        os.remove(val_and_test_logfile)

    print("\n=== Evaluating on Validation Set (20%) ===")
    evaluate_and_log(
        bert_v2, 
        roberta_v2, 
        ds_bert_val, 
        ds_roberta_val, 
        val_and_test_logfile, 
        set_name="VALIDATION"
    )

    print("\n=== Evaluating on 'Test Set' (the same 20%) ===")
    evaluate_and_log(
        bert_v2, 
        roberta_v2, 
        ds_bert_val,  # same data for demonstration
        ds_roberta_val, 
        val_and_test_logfile, 
        set_name="TEST"
    )

    # Evaluate on real contracts
    processed_contracts_dir = "./processed_contracts"
    delimiter = "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"
    real_log_file = "./real_contracts_predictions_updated_80_20.txt"

    if os.path.exists(real_log_file):
        os.remove(real_log_file)
    with open(real_log_file, "w", encoding="utf-8") as f:
        f.write("=== REAL CONTRACTS PREDICTIONS ===\n\n")

    def write_real_log(msg):
        with open(real_log_file, "a", encoding="utf-8") as ff:
            ff.write(msg)

    print("\n=== Real Contracts Evaluation ===\n")
    if os.path.exists(processed_contracts_dir):
        # NO local re-import of BertTokenizer or RobertaTokenizer
        bert_tok_v2 = BertTokenizer.from_pretrained(new_bert_tok_path)
        roberta_tok_v2 = RobertaTokenizer.from_pretrained(new_roberta_tok_path)

        for contract_dir in os.listdir(processed_contracts_dir):
            cpath = os.path.join(processed_contracts_dir, contract_dir)
            if os.path.isdir(cpath):
                contract_name = contract_dir
                for fname in os.listdir(cpath):
                    if fname.endswith("_broken_down.txt"):
                        broken_down_file = os.path.join(cpath, fname)
                        with open(broken_down_file, "r", encoding="utf-8") as fd:
                            content = fd.read()

                        clauses = content.split(delimiter)
                        for i, c_text in enumerate(clauses):
                            c_text = c_text.strip()
                            if not c_text:
                                continue

                            # BERT
                            inputs_b = bert_tok_v2(
                                c_text, 
                                return_tensors="pt",
                                truncation=True,
                                padding="max_length",
                                max_length=512
                            ).to(device)
                            with torch.no_grad():
                                out_b = bert_v2(**inputs_b)
                                prob_b = torch.softmax(out_b.logits, dim=-1)
                                cb, pb = torch.max(prob_b, dim=-1)
                            b_pred_str = bert_v2.config.id2label.get(pb.item(), "UNKNOWN")
                            conf_b = cb.item()

                            # RoBERTa
                            inputs_r = roberta_tok_v2(
                                c_text,
                                return_tensors="pt",
                                truncation=True,
                                padding="max_length",
                                max_length=512
                            ).to(device)
                            with torch.no_grad():
                                out_r = roberta_v2(**inputs_r)
                                prob_r = torch.softmax(out_r.logits, dim=-1)
                                cr, pr = torch.max(prob_r, dim=-1)
                            r_pred_str = roberta_v2.config.id2label.get(pr.item(), "UNKNOWN")
                            conf_r = cr.item()

                            # Ensemble
                            if conf_b <= 0.50 and conf_r <= 0.50:
                                final_pred_str = "UNKNOWN"
                            elif conf_b > 0.50 and conf_r > 0.50:
                                final_pred_str = b_pred_str if conf_b > conf_r else r_pred_str
                            elif conf_b > 0.50:
                                final_pred_str = b_pred_str
                            elif conf_r > 0.50:
                                final_pred_str = r_pred_str
                            else:
                                final_pred_str = "UNKNOWN"

                            write_real_log(f"Contract: {contract_name}\n")
                            write_real_log(f"Clause #{i}\n")
                            write_real_log(f"Clause Text:\n{c_text}\n")
                            write_real_log(f"BERT => {b_pred_str} (conf={conf_b:.4f})\n")
                            write_real_log(f"RoBERTa => {r_pred_str} (conf={conf_r:.4f})\n")
                            write_real_log(f"Ensemble => {final_pred_str}\n")
                            write_real_log("=" * 80 + "\n\n")
    else:
        print(f"No processed_contracts_dir found at: {processed_contracts_dir}")

    print("\nALL DONE!")
    print(f"Train logs => {training_log_file}")
    print(f"Val+Test predictions => {val_and_test_logfile}")
    print(f"Real-contract logs => {real_log_file}")


if __name__ == "__main__":
    main()
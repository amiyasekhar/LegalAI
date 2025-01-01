import os
import torch
import numpy as np
import pandas as pd
import optuna
from optuna import Trial
from optuna.trial import TrialState

from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertTokenizer,
    RobertaTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState
)
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric

def write_to_file(filename, text, mode='a'):
    """Helper to write text to a file."""
    with open(filename, mode) as file:
        file.write(text)

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
    """Same for RoBERTa."""
    for i, layer in enumerate(model.roberta.encoder.layer):
        if i < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False

def compute_metrics(eval_pred):
    """
    Simple accuracy metric using Hugging Face 'evaluate' library.
    The trainer will automatically compute:
      'eval_loss'  -> Cross-entropy on validation set
      'eval_accuracy' -> The 'accuracy' you compute here
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy_metric = load_metric("accuracy")
    return accuracy_metric.compute(predictions=preds, references=labels)

class LogTrainAccuracyCallback(TrainerCallback):
    """
    After each epoch, evaluate on the Trainer's *training* dataset
    so we can log 'train_accuracy' and 'train_loss' to see progress or overfitting.
    """
    def __init__(self):
        super().__init__()
        self.trainer = None  # We'll set this later via set_trainer()

    def set_trainer(self, trainer):
        """Store a reference to the trainer so we can call trainer.evaluate()."""
        self.trainer = trainer

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        # If we don't have a stored Trainer reference, do nothing
        if not self.trainer:
            return

        # Evaluate on the *training* dataset
        train_metrics = self.trainer.evaluate(
            eval_dataset=self.trainer.train_dataset,
            metric_key_prefix="train"  # shows up as train_loss, train_accuracy, etc.
        )
        # Log them so they appear in the Trainer's logs
        self.trainer.log(train_metrics)

        # Print them for clarity
        epoch = int(state.epoch)
        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_metrics.get('train_loss', float('nan')):.4f}, "
            f"train_accuracy={100*train_metrics.get('train_accuracy', 0):.2f}%"
        )

class LogValidationAccuracyCallback(TrainerCallback):
    """
    After each evaluation pass (end of each epoch if evaluation_strategy="epoch"),
    log the validation (eval) loss and accuracy.
    """
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs
    ):
        # metrics will contain "eval_loss" and "eval_accuracy" if computed by compute_metrics
        epoch = int(state.epoch)
        val_loss = metrics.get("eval_loss", float('nan'))
        val_acc = metrics.get("eval_accuracy", 0) * 100

        print(
            f"[Epoch {epoch}] "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_acc:.2f}%"
        )

def objective(trial: Trial,
              train_dataset_bert, val_dataset_bert,
              train_dataset_roberta, val_dataset_roberta,
              id2label, device, log_file):

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True)
    freeze_to = trial.suggest_int("freeze_layers", 0, 6)  # freeze up to 6 layers
    epochs = trial.suggest_int("num_epochs", 2, 4)        # 2..4 epochs

    # Build BERT model
    bert_model = BertForSequenceClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased",
        num_labels=len(id2label)
    ).to(device)
    freeze_bert_layers(bert_model, freeze_until=freeze_to)

    # Build RoBERTa model
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        "saibo/legal-roberta-base",
        num_labels=len(id2label)
    ).to(device)
    freeze_roberta_layers(roberta_model, freeze_until=freeze_to)

    # Shared training args
    training_args = TrainingArguments(
        output_dir="./results_optuna",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs_optuna",
        logging_steps=50,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )

    # Early stopping if val accuracy doesn't improve for 1 epoch
    callbacks = [EarlyStoppingCallback(early_stopping_patience=1)]

    # ------------------
    # BERT Trainer
    # ------------------
    trainer_bert = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=train_dataset_bert,
        eval_dataset=val_dataset_bert,  # needed for eval each epoch
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    # Add our custom logging callbacks
    log_cb_bert = LogTrainAccuracyCallback()
    log_cb_bert.set_trainer(trainer_bert)
    trainer_bert.add_callback(log_cb_bert)

    log_val_cb_bert = LogValidationAccuracyCallback()
    trainer_bert.add_callback(log_val_cb_bert)

    # Train BERT
    trainer_bert.train()

    # Final BERT validation check
    result_bert = trainer_bert.predict(val_dataset_bert)
    logits_bert = result_bert.predictions
    labels_bert = result_bert.label_ids
    preds_bert = np.argmax(logits_bert, axis=-1)
    acc_bert = 100.0 * (preds_bert == labels_bert).sum() / len(labels_bert)

    # ------------------
    # RoBERTa Trainer
    # ------------------
    trainer_roberta = Trainer(
        model=roberta_model,
        args=training_args,
        train_dataset=train_dataset_roberta,
        eval_dataset=val_dataset_roberta,  # needed for eval each epoch
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    log_cb_roberta = LogTrainAccuracyCallback()
    log_cb_roberta.set_trainer(trainer_roberta)
    trainer_roberta.add_callback(log_cb_roberta)

    log_val_cb_roberta = LogValidationAccuracyCallback()
    trainer_roberta.add_callback(log_val_cb_roberta)

    # Train RoBERTa
    trainer_roberta.train()

    # Final RoBERTa validation check
    result_roberta = trainer_roberta.predict(val_dataset_roberta)
    logits_roberta = result_roberta.predictions
    labels_roberta = result_roberta.label_ids
    preds_roberta = np.argmax(logits_roberta, axis=-1)
    acc_roberta = 100.0 * (preds_roberta == labels_roberta).sum() / len(labels_roberta)

    # We'll optimize for the average accuracy of BERT + RoBERTa
    avg_acc = (acc_bert + acc_roberta) / 2.0

    line = (
        f"\nTrial {trial.number} -> lr={learning_rate:.2e}, freeze={freeze_to}, epochs={epochs}\n"
        f"BERT_acc={acc_bert:.2f}, RoBERTa_acc={acc_roberta:.2f}, Avg_acc={avg_acc:.2f}\n"
    )
    print(line)
    write_to_file(log_file, line)

    return avg_acc

def main():
    log_file = "./train_validation_log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    # Use Apple Silicon GPU if available, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load data
    df1 = pd.read_csv("Clauses 1.csv")
    df2 = pd.read_csv("Clauses 2.csv")
    combined_df = pd.concat([df1, df2], ignore_index=True)

    label_column = 'Clause Heading'
    unique_labels = combined_df[label_column].unique()
    num_labels = len(unique_labels)
    label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    # Convert label col to IDs
    combined_df['labels'] = combined_df[label_column].map(label2id)

    # Split 80/20 per label group
    train_df_list, val_df_list = [], []
    for label, group in combined_df.groupby(label_column):
        size = int(0.8 * len(group))
        train_df_list.append(group[:size])
        val_df_list.append(group[size:])

    train_df = pd.concat(train_df_list).reset_index(drop=True)
    val_df = pd.concat(val_df_list).reset_index(drop=True)

    # Build a DatasetDict for Hugging Face
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df[['Clause Content', 'labels']]),
        "validation": Dataset.from_pandas(val_df[['Clause Content', 'labels']])
    })

    # Tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    roberta_tokenizer = RobertaTokenizer.from_pretrained("saibo/legal-roberta-base")

    def preprocess_function_bert(examples):
        return bert_tokenizer(
            examples['Clause Content'],
            truncation=True,
            padding='max_length',
            max_length=512
        )

    def preprocess_function_roberta(examples):
        return roberta_tokenizer(
            examples['Clause Content'],
            truncation=True,
            padding='max_length',
            max_length=512
        )

    bert_dataset = dataset.map(preprocess_function_bert, batched=True)
    roberta_dataset = dataset.map(preprocess_function_roberta, batched=True)

    # The function Optuna will call each trial
    def optuna_objective(trial):
        return objective(
            trial,
            train_dataset_bert=bert_dataset["train"],
            val_dataset_bert=bert_dataset["validation"],
            train_dataset_roberta=roberta_dataset["train"],
            val_dataset_roberta=roberta_dataset["validation"],
            id2label=id2label,
            device=device,
            log_file=log_file
        )

    # Create an Optuna study
    import optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=5)

    # Summarize the best trial
    best_trial = study.best_trial
    line = (
        f"\nBest trial #: {best_trial.number}\n"
        f"  Value (avg_acc): {best_trial.value:.2f}\n"
        f"  Params: {best_trial.params}\n"
    )
    print(line)
    write_to_file(log_file, line)

    print("Done with main!")

if __name__ == "__main__":
    main()

'''
we will stick with this code



we notice the following trends


Params: lr ~2.64e-05, freeze=1, epochs=4, we get •	BERT_acc = 89.08%
	•	RoBERTa_acc = 86.78%
	•	Avg_acc = 87.93%

Params: lr ~2.12e-05, freeze=6, epochs=4
	•	Results:
	•	BERT_acc = 81.03%
	•	RoBERTa_acc = 85.06%
	•	Avg_acc = 83.05%

	•	Params: lr ~2.62e-05, freeze=6, epochs=2
	•	Results:
	•	BERT_acc = 67.24%
	•	RoBERTa_acc = 81.03%
	•	Avg_acc = 74.14%

given the trends we notice, can we fine tune further?
'''
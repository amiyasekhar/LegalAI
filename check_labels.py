import os
import pandas as pd
from transformers import BertForSequenceClassification, RobertaForSequenceClassification

def main():
    ###################################################################
    # 1) Paths to your fine-tuned BERT & RoBERTa model directories
    ###################################################################
    old_bert_model_path = "./fine_tuned_legal_bert"
    old_roberta_model_path = "./fine_tuned_legal_roberta"
    
    if not os.path.exists(old_bert_model_path):
        raise FileNotFoundError(f"BERT model folder not found: {old_bert_model_path}")
    if not os.path.exists(old_roberta_model_path):
        raise FileNotFoundError(f"RoBERTa model folder not found: {old_roberta_model_path}")

    ###################################################################
    # 2) Load the old models
    ###################################################################
    bert_model = BertForSequenceClassification.from_pretrained(old_bert_model_path)
    roberta_model = RobertaForSequenceClassification.from_pretrained(old_roberta_model_path)

    # Retrieve label2id from each model
    bert_label2id = bert_model.config.label2id
    roberta_label2id = roberta_model.config.label2id

    print("=== BERT Model Labels ===")
    for k, v in bert_label2id.items():
        print(f"  - '{k}' -> id: {v}")
    print()

    print("=== RoBERTa Model Labels ===")
    for k, v in roberta_label2id.items():
        print(f"  - '{k}' -> id: {v}")
    print()

    ###################################################################
    # 3) Read data from CSV 1, CSV 2, and XLSX
    ###################################################################
    csv1_path = "Clauses 1.csv"
    csv2_path = "Clauses 2.csv"
    xlsx_path = "clause_content_variety_latest_clauses.xlsx"

    if not os.path.exists(csv1_path):
        raise FileNotFoundError(f"CSV 1 not found: {csv1_path}")
    if not os.path.exists(csv2_path):
        raise FileNotFoundError(f"CSV 2 not found: {csv2_path}")
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"XLSX not found: {xlsx_path}")

    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    df_xls = pd.read_excel(xlsx_path)

    # Combine them
    combined_df = pd.concat([df1, df2, df_xls], ignore_index=True)

    # Column name for clause headings:
    label_column = "Clause Heading"

    # All unique labels from the data
    all_labels_in_data = set(combined_df[label_column].unique())

    print(f"Total unique labels in data: {len(all_labels_in_data)}")

    ###################################################################
    # 4) Compare each model’s supported labels vs. data labels
    ###################################################################
    bert_supported = set(bert_label2id.keys())
    roberta_supported = set(roberta_label2id.keys())

    # BERT
    bert_unrecognized = all_labels_in_data - bert_supported
    bert_recognized = all_labels_in_data & bert_supported

    # RoBERTa
    roberta_unrecognized = all_labels_in_data - roberta_supported
    roberta_recognized = all_labels_in_data & roberta_supported

    print("\n=== BERT Coverage ===")
    print(f"- BERT recognizes {len(bert_recognized)} labels out of {len(all_labels_in_data)} total in the data.")
    print("  * BERT-supported labels in data:", sorted(bert_recognized))
    print()
    print(f"- BERT does NOT recognize {len(bert_unrecognized)} labels:")
    print("  * Missing:", sorted(bert_unrecognized))

    print("\n=== RoBERTa Coverage ===")
    print(f"- RoBERTa recognizes {len(roberta_recognized)} labels out of {len(all_labels_in_data)} total in the data.")
    print("  * RoBERTa-supported labels in data:", sorted(roberta_recognized))
    print()
    print(f"- RoBERTa does NOT recognize {len(roberta_unrecognized)} labels:")
    print("  * Missing:", sorted(roberta_unrecognized))

    print("\nDone. If you see any labels in the 'Missing' lists that you want the models to handle, you'll need to expand them accordingly.")

if __name__ == "__main__":
    main()
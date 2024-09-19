import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, recall_score, precision_score
from transformers import AutoConfig, AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaForTokenClassification
from datasets import load_from_disk, Dataset
from torch.nn.functional import cross_entropy
from transformers import DataCollatorForTokenClassification

def tokenize_and_align_labels(dataset, tokenizer):
    # Concatenate tokens into strings for tokenization
    sentences = ["".join(tokens) for tokens in dataset["tokens"]]
    # Tokenize the sentences using the tokenizer
    tokenized_outputs = tokenizer(sentences, truncation=True, padding=True, is_split_into_words=False, return_offsets_mapping=True)

    # Initialize an empty list to store the new labels
    aligned_labels = []

    # Iterate over each sentence and its corresponding NER tags
    for sentence_idx, original_labels in enumerate(dataset["ner_tags"]):
        # Initialize the label list for the current tokenized sentence
        current_labels = [0] * len(tokenized_outputs["input_ids"][sentence_idx])

        # Retrieve offset mappings
        offsets = tokenized_outputs["offset_mapping"][sentence_idx]

        # Map original labels to the new tokenized sentence
        for char_idx, (start, end) in enumerate(offsets):
            # Find the token index corresponding to the character index
            token_idx = tokenized_outputs.char_to_token(sentence_idx, char_idx)
            if token_idx is not None:
                current_labels[token_idx] = original_labels[char_idx]
        
        aligned_labels.append(current_labels)

    # Add the aligned labels to the tokenized outputs
    tokenized_outputs["labels"] = aligned_labels
    return tokenized_outputs
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths
    model_path = "./trained_ner_classifier_jp"
    tokenizer_name = "xlm-roberta-base"
    test_dataset_path = "./tokenized_test_dataset"
    results_path = "test_results"
    os.makedirs(results_path, exist_ok=True)

    # Load model and tokenizer
    index_to_tag = {0: 'O', 1: 'PER', 2: 'ORG', 3: 'ORG-P', 4: 'ORG-O', 5: 'LOC', 6: 'INS', 7: 'PRD', 8: 'EVT'}
    tag_to_index = {tag: idx for idx, tag in index_to_tag.items()}
    
    xlmr_config = AutoConfig.from_pretrained(
        tokenizer_name,
        num_labels=9,
        id2label=index_to_tag,
        label2id=tag_to_index
    )
    
    model = RobertaForTokenClassification.from_pretrained(model_path, config=xlmr_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="pt")

    # Load test dataset
    test_dataset = load_from_disk(test_dataset_path)
    #test_dataset = test_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    def process_test(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        new_batch = data_collator(features)
        
        with torch.no_grad():
            input_ids = new_batch["input_ids"].to(device)
            attention_mask = new_batch["attention_mask"].to(device)
            output = model(input_ids, attention_mask)
            predicted_label_id = torch.argmax(output.logits, axis=-1).cpu().numpy()

        true_label_id = new_batch["labels"].to(device)
        loss = cross_entropy(
            output.logits.view(-1, 9),
            true_label_id.view(-1),
            reduction="none"
        ).view(len(input_ids), -1).cpu().numpy()
        
        return {"loss": loss, "predicted_labels": predicted_label_id}

    # Apply the processing function to the test dataset
    test_output = test_dataset.map(process_test, batched=True, batch_size=32)
    test_output_df = test_output.to_pandas()

    test_output_df["label_tags"] = test_output_df["labels"].apply(
        lambda row: [index_to_tag[i] for i in row])
    test_output_df["predicted_label_tags"] = test_output_df[["predicted_labels", "attention_mask"]].apply(
        lambda row: [index_to_tag[i] for i in row.predicted_labels][:len(row.attention_mask)], axis=1)

    # Compute confusion matrix
    y_true = list(chain.from_iterable(test_output_df["label_tags"]))
    y_pred = list(chain.from_iterable(test_output_df["predicted_label_tags"]))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(index_to_tag.values()), normalize="true")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 8))
    conf_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(index_to_tag.values()))
    conf_display.plot(cmap="Blues", values_format=".3f", ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_path, "confusion_matrix.png"))
    plt.close()
    print("plot is saved in the following path...",results_path)
    # Compute metrics
    accuracy = np.trace(conf_matrix) / conf_matrix.sum()
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Save metrics to text file
    with open(os.path.join(results_path, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Precision: {', '.join(f'{tag}: {p:.4f}' for tag, p in zip(index_to_tag.values(), precision))}\n")
        f.write(f"Recall: {', '.join(f'{tag}: {r:.4f}' for tag, r in zip(index_to_tag.values(), recall))}\n")
    
    print("result text file is saved in the following path...",results_path+"metrics.txt")

if __name__ == "__main__":
    main()

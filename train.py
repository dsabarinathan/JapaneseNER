import json
import os
import argparse
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from transformers.models.roberta import RobertaForTokenClassification
from transformers import DataCollatorForTokenClassification
import torch
from utils import metrics_func

def load_and_process_data(json_path):
    # Load dataset
    with open(json_path, "r",encoding="utf-8") as file:
        annotated_data = json.load(file)

    # Initialize an empty list to store the types
    types = []

    # Loop through each entry in the JSON data
    for entry in annotated_data:
        # Check if 'entities' field exists in the entry
        if 'entities' in entry:
            # Loop through each entity in the 'entities' list
            for entity in entry['entities']:
                # Check if 'type' field exists in the entity
                if 'type' in entity:
                    # Append the 'type' value to the types list
                    types.append(entity['type'])

    # Optional: remove duplicates
    types = list(set(types))

    # Create character-based annotated dataset
    char_tokens_list = []
    char_ner_tags_list = []
    for entry in annotated_data:
        char_tokens = list(entry["text"])
        char_ner_tags = ["O"] * len(char_tokens)
        for entity in entry["entities"]:
            for i in range(entity["span"][0], entity["span"][1]):
                if entity["type"] == "人名":  # person
                    char_ner_tags[i] = "PER"
                elif entity["type"] == "法人名":  # organization (corporation general)
                    char_ner_tags[i] = "ORG"
                elif entity["type"] == "政治的組織名":  # organization (political)
                    char_ner_tags[i] = "ORG-P"
                elif entity["type"] == "その他の組織名":  # organization (others)
                    char_ner_tags[i] = "ORG-O"
                elif entity["type"] == "地名":  # location
                    char_ner_tags[i] = "LOC"
                elif entity["type"] == "施設名":  # institution (facility)
                    char_ner_tags[i] = "INS"
                elif entity["type"] == "製品名":  # product
                    char_ner_tags[i] = "PRD"
                elif entity["type"] == "イベント名":  # event
                    char_ner_tags[i] = "EVT"
        char_tokens_list.append(char_tokens)
        char_ner_tags_list.append(char_ner_tags)

    # Define dataset features
    dataset_features = Features({
        "tokens": Sequence(feature=Value(dtype='string'), length=-1),
        "ner_tags": Sequence(feature=ClassLabel(names=["O", "PER", "ORG", "ORG-P", "ORG-O", "LOC", "INS", "PRD", "EVT"]), length=-1)
    })

    # Create dataset from dictionary
    ner_dataset = Dataset.from_dict(
        {"tokens": char_tokens_list, "ner_tags": char_ner_tags_list},
        features=dataset_features
    )

    # Generate converter for index(int)-to-tag(string) and tag(string)-to-index(int)
    tag_feature = ner_dataset.features["ner_tags"].feature
    index_to_tag = {index: tag for index, tag in enumerate(tag_feature.names)}
    tag_to_index = {tag: index for index, tag in enumerate(tag_feature.names)}

    # Split dataset into training and validation datasets
    train_val_split = ner_dataset.train_test_split(test_size=0.15, shuffle=True)

    # Save the train and validation datasets separately
    train_val_split['train'].save_to_disk('./train_dataset')
    train_val_split['test'].save_to_disk('./test_dataset')

    return train_val_split, index_to_tag, tag_to_index

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

def main(json_path):
    train_val_split, index_to_tag, tag_to_index = load_and_process_data(json_path)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # Apply the tokenization and label alignment to the dataset
    tokenized_dataset = train_val_split.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        remove_columns=["tokens", "ner_tags"],
        batched=True,
        batch_size=128
    )
    
    tokenized_dataset['train'].save_to_disk('./tokenized_train_dataset')
    tokenized_dataset['test'].save_to_disk('./tokenized_test_dataset')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_collator = DataCollatorForTokenClassification(
        tokenizer,
        return_tensors="pt"
    )

    xlmr_config = AutoConfig.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(tag_to_index),
        id2label=index_to_tag,
        label2id=tag_to_index
    )
    model = RobertaForTokenClassification.from_pretrained("xlm-roberta-base", config=xlmr_config).to(device)

    training_args = TrainingArguments(
        output_dir="./xlm-roberta-ner-ja",
        log_level="error",
        num_train_epochs=10,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        evaluation_strategy="epoch",
        fp16=True,
        logging_steps=len(tokenized_dataset["train"]),
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=metrics_func,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer
    )

   # trainer.train()

    # Save fine-tuned model locally
   # model_save_path = "./trained_ner_classifier_jp"
   # os.makedirs(model_save_path, exist_ok=True)
   # model.save_pretrained(model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NER model using XLM-RoBERTa.")
    parser.add_argument("--json_path", type=str, help="Path to the JSON file containing the dataset.")
    args = parser.parse_args()

    main(args.json_path)

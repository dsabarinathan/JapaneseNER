# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:03:17 2024

@author: SABARI
"""
import os
import torch
from transformers import AutoConfig
from transformers.models.roberta.modeling_roberta import RobertaForTokenClassification
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import spacy
from spacy.tokens import Doc, Span
from spacy import displacy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JapaneseNER():
    def __init__(self, model_path, model_name="xlm-roberta-base"):
        self._index_to_tag = {0: 'O',
                              1: 'PER',
                              2: 'ORG',
                              3: 'ORG-P',
                              4: 'ORG-O',
                              5: 'LOC',
                              6: 'INS',
                              7: 'PRD',
                              8: 'EVT'}
        
        self._tag_to_index = {v: k for k, v in self._index_to_tag.items()}
        self._tag_feature_num_classes = len(self._index_to_tag)
        self._model_name = model_name
        self._model_path = model_path
        
        xlmr_config = AutoConfig.from_pretrained(
            self._model_name,
            num_labels=self._tag_feature_num_classes,
            id2label=self._index_to_tag,
            label2id=self._tag_to_index
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.model = (RobertaForTokenClassification
                      .from_pretrained(self._model_path, config=xlmr_config)
                      .to(device))
    
    def prepare(self):
        # Create dataset for prediction
        sample_encoding = self.tokenizer([ 
          "鈴木は4月の陽気の良い日に、鈴をつけて熊本県の阿蘇山に登った",
          "中国では、中国共産党による一党統治が続く",
        ], truncation=True, padding=True,  # Ensure all sequences are of the same length
                                    max_length=512, return_tensors="pt")
        
        sample_encoding = {k: v.to(device) for k, v in sample_encoding.items()}

        # Perform prediction
        with torch.no_grad():
            output = self.model(**sample_encoding)
        
        predicted_label_id = torch.argmax(output.logits, axis=-1).cpu().numpy()[0]
        print("Predicted labels:", predicted_label_id)

    def predict(self, text):
        encoding = self.tokenizer([text], truncation=True, padding=True, max_length=512, return_tensors="pt")
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        # Perform prediction
        with torch.no_grad():
            output = self.model(**encoding)
        
        # Get the predicted label ids
        predicted_label_id = torch.argmax(output.logits, axis=-1).cpu().numpy()[0]
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        
        # Map the predicted labels to their corresponding tag
        predictions = [self._index_to_tag[label_id] for label_id in predicted_label_id]
        
        return tokens, predictions

# Instantiate the NER model
model_path = "./trained_ner_classifier_jp/"
ner_model = JapaneseNER(model_path)
ner_model.prepare()

# Function to integrate with spaCy displacy for visualization
def ner_inference(text):
    # Get tokens and predictions
    tokens, predictions = ner_model.predict(text)
    
    # Create a spaCy document to visualize with displacy
    nlp = spacy.blank("ja")  # Initialize a blank Japanese model in spaCy
    doc = Doc(nlp.vocab, words=tokens)  # Create a spaCy Doc object with tokens
    
    # Create entity spans from predictions and add them to the Doc object
    ents = []
    start_idx = 0
    for i, label in enumerate(predictions):
        if label != 'O':  # Skip non-entity tokens
            span = Span(doc, start_idx, start_idx + 1, label=label)  # Create Span for the token
            ents.append(span)
        start_idx += 1
    doc.ents = ents  # Set the entities in the Doc
    
    # Render using spaCy displacy
    html = displacy.render(doc, style="ent", jupyter=False)  # Generate HTML for entities
    return html

# Sample text for demonstration
sample_text = "鈴木一朗は2020年に引退した。女優の石原さとみは多くの映画で主演している。"

# Create Gradio interface
import gradio as gr

iface = gr.Interface(
    fn=ner_inference,  # The function to call for prediction
    inputs=gr.Textbox(lines=5, placeholder="Enter Japanese text for NER...", value=sample_text),  # Input widget with sample text
    outputs="html",  # Output will be in HTML format using displacy
    title="Japanese Named Entity Recognition (NER)",
    description="Enter Japanese text and see the named entities highlighted in the output."
)

# Launch the interface
iface.launch()

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:03:17 2024

@author: SABARI
"""
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaForTokenClassification
from spacy.tokens import Doc, Span
from spacy import displacy
import spacy
from flask import Flask, request, jsonify

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JapaneseNER:
    def __init__(self, model_path, model_name="xlm-roberta-base"):
        self._index_to_tag = {0: 'O', 1: 'PER', 2: 'ORG', 3: 'ORG-P', 4: 'ORG-O', 5: 'LOC', 6: 'INS', 7: 'PRD', 8: 'EVT'}
        self._tag_to_index = {v: k for k, v in self._index_to_tag.items()}
        self._tag_feature_num_classes = len(self._index_to_tag)
        self._model_name = model_name
        self._model_path = model_path

        xlmr_config = AutoConfig.from_pretrained(self._model_name, num_labels=self._tag_feature_num_classes, id2label=self._index_to_tag, label2id=self._tag_to_index)

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.model = RobertaForTokenClassification.from_pretrained(self._model_path, config=xlmr_config).to(device)

    def predict(self, text):
        encoding = self.tokenizer([text], truncation=True, max_length=512, return_tensors="pt")
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

# Create Flask API
app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    # Get the text input from the POST request
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    
    # Get tokens and predictions
    tokens, predictions = ner_model.predict(text)
    
    # Create a spaCy document to visualize with displacy
    nlp = spacy.blank("en")  # Initialize a blank English model in spaCy
    doc = Doc(nlp.vocab, words=tokens)  # Create a spaCy Doc object with tokens
    
    # Create entity spans from predictions and add them to the Doc object
    ents = []
    start = 0
    for i, label in enumerate(predictions):
        if label != 'O':  # Skip non-entity tokens
            span = Span(doc, start, start + 1, label=label)  # Create Span for the token
            ents.append(span)
        start += 1
    doc.ents = ents  # Set the entities in the Doc
    
    # Render using spacy displacy
    html = displacy.render(doc, style="ent", jupyter=False)  # Generate HTML for entities
    
    response = jsonify({"tokens": tokens, "predictions": predictions, "html": html})
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    
    # Return JSON response
    return response

if __name__ == '__main__':
    # Instantiate the NER model
    model_path = "./trained_ner_classifier_jp/"
    ner_model = JapaneseNER(model_path)
    print("NER model intialized....")
    app.run(debug=True, use_reloader=False)

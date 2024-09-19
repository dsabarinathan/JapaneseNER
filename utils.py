import numpy as np
from seqeval.metrics import f1_score
import torch
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

def metrics_func(eval_arg):    
  index_to_tag = {0: 'O',1: 'PER',
    2: 'ORG',
    3: 'ORG-P',
    4: 'ORG-O',
    5: 'LOC',
    6: 'INS',
    7: 'PRD',
    8: 'EVT'}
  preds = np.argmax(eval_arg.predictions, axis=2)
  batch_size, seq_len = preds.shape
  y_true, y_pred = [], []
  for b in range(batch_size):
    true_label, pred_label = [], []
    for s in range(seq_len):
      if eval_arg.label_ids[b, s] != -100:  # -100 must be ignored
        true_label.append(index_to_tag[eval_arg.label_ids[b][s]])
        pred_label.append(index_to_tag[preds[b][s]])
    y_true.append(true_label)
    y_pred.append(pred_label)
  return {"f1": f1_score(y_true, y_pred)}


class CustomRobertaForTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]  # Ignore the pooler as it's not used
    _keys_to_ignore_on_load_missing = [r"position_ids"]  # Ignore missing position IDs

    def __init__(self, config):
        super().__init__(config)

        #
        # The names of layers ("roberta", etc.) are critical!
        # Changing these names can result in weights and biases being ignored when saving checkpoints.
        #

        self.num_labels = config.num_labels
        # Load the pre-trained RoBERTa model without the pooling layer
        self.roberta_model = RobertaModel(config, add_pooling_layer=False)
        # Dropout layer for regularization
        self.dropout_layer = torch.nn.Dropout(config.hidden_dropout_prob)
        # Linear layer for token classification
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        # Initialize weights
        ### self.init_weights()
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Forward pass through the RoBERTa model
        roberta_outputs = self.roberta_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,  # This will always be None
            **kwargs
        )
        # Apply dropout and pass through the linear layer
        sequence_output = self.dropout_layer(roberta_outputs[0])
        logits = self.classifier(sequence_output)
        # Compute the loss if labels are provided
        loss = None
        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        # Return the output as a TokenClassifierOutput
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=roberta_outputs.hidden_states,
            attentions=roberta_outputs.attentions
        )

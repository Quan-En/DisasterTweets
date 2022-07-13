
import torch
import torch.nn as nn
from transformers import BertModel

PRETRAINED_MODEL_NAME = "bert-base-cased"

# Bert-Classifier
class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.embedding = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.fc = nn.Linear(768, 2)

    def forward(self, tokens, masks=None):

        # BERT
        embedded_dict = self.embedding(tokens, attention_mask=masks)
        cls_vector = embedded_dict[0][:, 0, :]

        # Fully Connected
        outputs = self.fc(cls_vector)

        return outputs
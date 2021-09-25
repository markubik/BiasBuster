from transformers import LongformerTokenizer, LongformerForSequenceClassification

import torch
import torch.nn.functional as F

MODEL_PATH = './model'

class HyperpartisanModel:
    labels = ['NORMAL', 'HYPERPARTISAN']

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.loaded = False

    def load(self):
        self.tokenizer = LongformerTokenizer.from_pretrained(MODEL_PATH)
        self.model = LongformerForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.eval()

        self.loaded = True

    def predict(self, text):
        if not self.loaded:
            raise Excpetion('You must load model first!')

        inputs = self.tokenizer(text, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)
        outputs = self.model(**inputs, labels=labels) #TODO investigate performance

        return self.labels[torch.argmax(F.softmax(outputs.logits, dim=-1))]


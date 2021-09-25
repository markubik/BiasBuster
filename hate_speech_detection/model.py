from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_PATH = './model'

class HateSpeechModel:
    labels = ['HATE_SPEECH', 'OFFENSIVE',  'NORMAL']

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.loaded = False

    def load(self, from_local_dir=True):
        if from_local_dir:
            self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
            self.model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        else: 
            self.tokenizer = BertTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
            self.model = BertForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
        self.loaded = True

    def predict(self, text):
        if not self.loaded:
            raise Excpetion('You must load model first!')

        n_chunks = min(16, -(len(text)//(-512)))
        chunks = [text[0+i:512+i] for i in range(0, len(text), 512)]
        inputs = self.tokenizer(chunks, return_tensors="pt", padding=True)
        labels = torch.tensor([1] * n_chunks).unsqueeze(0)
        outputs = self.model(**inputs, labels=labels)

        return self.labels[torch.argmax(F.softmax(torch.mean(outputs.logits, dim=-2), dim=-1))]

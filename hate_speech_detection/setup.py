from transformers import BertTokenizer, BertForSequenceClassification
import os

MODEL_PATH = './model'

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
    model = BertForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    tokenizer.save_pretrained(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)

    print('Tokenizer and model save to', MODEL_PATH)

    
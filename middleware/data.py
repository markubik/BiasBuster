from typing import TypedDict

def new_article(header, text):
    return {
        "header": header,
        "text": text,
    }

def new_detector(name, url):
    return {
        'name': name,
        'url': url,
    }

def new_detector_resp(name, prediction, error=None):
    return {
        'name': name,
        'prediction': prediction,
        'error': error,
    }

def new_preds(hatespeech, hyperpartisan, stance):
    return {
        'hatespeech': hatespeech,
        'hyperpartisan': hyperpartisan,
        'stance': stance,
    }

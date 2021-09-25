# Middleware

Server that interacts with chrome extension. Orchiestrates dispathcing requests to detectors.

## Setup with Anaconda
Create environment:
```console
conda create --name middleware python=3.8
```
Activate environment:
```console
conda activate middleware
```
Install required python packages:
```console
pip install -q -r requirements.txt
```

## Server
Start server with
```console
python server.py
```
It will run on http://0.0.0.0:5000/ by default.

Test it with
```console
curl -H "Content-Type: application/json" -d '{"url":"https://edition.cnn.com/2021/09/24/uk/sabina-nessa-murder-london-cctv-gbr-intl/index.html"}' 0.0.0.0:5000/predict
```

Sample response:
```json
{
    "bias": "UNBIASED", // UNBIASED | BIASED | STRONGLY_BIASED
    "predictions": {
        "hatespeech": {
            "prediction":"HATESPEECH" // NORMAL | OFFENSIVE | HATESPEECH
        },
        "hyperpartisan": {
            "prediction": "NORMAL", // NORMAL | HYPERPARTISAN
        },
        "stance": {
            "prediction": "DISCUSS", // UNRELATED | AGREE | DISAGREE | DISCUSS
        },
    }
}
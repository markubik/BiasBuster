# Stance detection

## Setup with Anaconda
Create environment:
```console
conda create --name stance python=3.5
```
Activate environment:
```console
conda activate stance
```
Install required python packages:
```console
pip install -q -r requirements.txt
```

## Test model
Run:
```console
python model_test.py 
```

## Server
Start server with
```console
python server.py
```
It will run on http://0.0.0.0:5300/ by default.

Test it with
```console
curl -H "Content-Type: application/json" -d '{"header":"Biden has a cat", "text":"I love rainbows"}' 0.0.0.0:5300/predict
```

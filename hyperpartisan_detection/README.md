# Hyperpartisan detection

Predicts if the article is hyperpartisan or not.

## Setup with Anaconda
Create environment:
```console
conda create --name hyperpartisan python=3.8
```
Activate environment:
```console
conda activate hyperpartisan
```
Install required python packages:
```console
pip install -q -r requirements.txt
```
Install tokenizer and model files:
```console
python setup.py
```
**Manually** get a copy of model weights and put them into ```model``` directory.

## Server
Start server with
```console
python server.py
```
It will run on http://0.0.0.0:5200/ by default.

Test it with
```console
curl -H "Content-Type: application/json" -d '{"text":"New York (CNN Business) President Joe Biden gave a gift to every major company in America by forcing them to mandate vaccines or stringently test their employees for Covid. Their reaction to the new rule: glee. Corporate America had been trying to navigate two competing pandemic realities: Companies are desperately trying to get back to business as usual, and mandating vaccines is among the best ways to accomplish that. But a labor shortage had tied their hands, as businesses have been worried that forcing people to get the shot would send some desperately needed employees and potential new hires packing. Some state and local governments had imposed various vaccine mandates, others had outright banned them — and all the while vaccines have also become politically charged."}' 127.0.0.1:5200/predict
```
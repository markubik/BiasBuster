''' Train script '''

# !pip install -q torch==1.7.1 torchtext==0.8.1 bleach==3.2.1 beautifulsoup4==4.6.3 datasets==1.2.1 transformers==4.2.1 pytorch-lightning==1.1.4

import datasets
from datasets import list_datasets, load_dataset, list_metrics, load_metric
datasets_list = list_datasets()
import torch
import re
import bleach
from bs4 import BeautifulSoup
from datetime import datetime

def log(msg):
    with open('log.txt', 'a+') as f:
        now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        f.write('[' + now + ']   ' + msg + '\n')

log('Start')

dataset  = datasets.load_dataset('hyperpartisan_news_detection', 'bypublisher')

log('Dataset loaded')

position = 1

def clean_text(text, label):
    """ Clean the input text and remove special characters """
#     soup = BeautifulSoup(text)
    text = bleach.clean(text,strip=True)
    text = text.replace('<p>', '')
    text = text.replace('</p>', '')
    text = text.replace('\n', '')
    text = text.replace('&amp;#160;', '')
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    if str(label) == 'True':
        new_label = 1
    else:
        new_label = 0
    return text, new_label


def convert_to_features(example):
    # Tokenize contexts and questions (as pairs of inputs)
    text_, target_ = clean_text(example['text'], example['hyperpartisan'])
    encodings = tokenizer.encode_plus(text_, pad_to_max_length=True, max_length=2048,
                                           add_special_tokens=True,
                                            return_token_type_ids=False,
                                            return_attention_mask=True,
                                            padding='max_length', truncation=True,
                                           )
    targets = torch.tensor(target_,dtype=torch.long)
    

    encodings.update({'targets': targets,
                      'attention_mask': encodings['attention_mask']})
    return encodings

from transformers import LongformerTokenizer
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

## Take subset of data
train_size = 20000
val_size = 5000
import numpy as np
train_indices = np.random.randint(0, len(dataset['train']), train_size)
val_indices = np.random.randint(0, len(dataset['validation']), val_size)
train_dataset = dataset['train'].select(train_indices)
val_dataset = dataset['validation'].select(val_indices)

train_dataset =  train_dataset.map(convert_to_features, load_from_cache_file=False)
val_dataset =  val_dataset.map(convert_to_features, load_from_cache_file=False)

columns = ['input_ids', 'attention_mask', 'targets']
train_dataset.set_format(type='torch', columns=columns)
val_dataset.set_format(type='torch', columns=columns)

torch.save(train_dataset, 'train_data.pt')
torch.save(val_dataset, 'valid_data.pt')

log('Data proccessed')

import torch
from transformers import LongformerTokenizer
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
train_dataset = torch.load('train_data.pt')
val_dataset = torch.load('valid_data.pt')


import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from datasets import load_metric
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from transformers import (
    AdamW,
    LongformerTokenizer,
    LongformerForSequenceClassification,
    get_linear_schedule_with_warmup
)
import logging
import argparse
import time

from pytorch_lightning.loggers.wandb import WandbLogger
import os
from pathlib import Path
from string import punctuation

class LonformerFineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(LonformerFineTuner, self).__init__()
        self.hparams = hparams        
        self.model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", return_dict=True, num_labels=2)
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.output_dir = Path(self.hparams.output_dir)
        self.step_count = 0
        
        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())
            
            
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "validation": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        self.em_score_list = []
        
    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False
            
    def freeze_embeds(self):
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def lmap(self, f, x):
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.global_rank <= 0
    
    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
        
    def forward(self, input_ids, attention_mask, labels):
        return self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

    def _step(self, batch):
        labels = batch["targets"]

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )

        loss = outputs[0]

        return loss
    
    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
    def _generative_step(self, batch) :
        
        labels = batch["targets"]

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        
        loss = outputs[0]
        logits = F.softmax(outputs[1], dim=1)
        preds = torch.argmax(logits, 1)
        preds = preds.cpu().data.numpy()
        targets = labels.cpu().data.numpy()
#         print("Target and Preds: ", targets, preds)
        
        t0 = time.time()
            
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]  
    
        loss = self._step(batch)
        base_metrics = {'val_loss': loss}
        base_metrics.update(gen_time=gen_time)
        
    
        em_score  = accuracy_score(targets, preds)
        
        self.em_score_list.append(em_score)
        
        em_score = torch.tensor(em_score,dtype=torch.float32)
    
        base_metrics.update(accuracy=em_score)
        
        return base_metrics

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
  
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch)
    
    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        
        if len(self.em_score_list) <= 2:
            average_em_score = sum(self.em_score_list) / len(self.em_score_list) 
            
        else:
            latest_em_score = self.em_score_list[:-2]
            average_em_score = sum(latest_em_score) / len(latest_em_score) 
        
        average_em_score = torch.tensor(average_em_score,dtype=torch.float32)
        tensorboard_logs.update(accuracy=average_em_score)
        
        self.target_gen= []
        self.prediction_gen=[]
        return {"avg_val_loss": avg_loss, 
                "accuracy" : average_em_score,
                "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, 
                          eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]
  
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step(closure=closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()
  
    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict
    
    def train_dataloader(self):   
        n_samples = self.n_obs['train']
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler

        return dataloader

    def val_dataloader(self):
        n_samples = self.n_obs['validation']
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)
    
    def test_dataloader(self):
        n_samples = self.n_obs['test']
        test_dataset = get_dataset(tokenizer=self.tokenizer, type_path="test", num_samples=n_samples, args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)
    
    def on_save_checkpoint(self, checkpoint):
        save_path = self.output_dir.joinpath(model_prefix)
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

model_name = "longformer"
token_len = 4096
model_prefix = f"{model_name}-{token_len}"

args_dict = dict(
    output_dir="", # path to save the checkpoints
    max_input_length=token_len,
    freeze_encoder=False,
    freeze_embeds=False,
    learning_rate=2e-5,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=1,
    eval_batch_size=1,
    num_train_epochs=25,
    gradient_accumulation_steps=1,
    n_gpu=1,
    resume_from_checkpoint=None, 
    val_check_interval = 0.5, 
    n_val=-1,
    n_train=-1,
    n_test=-1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)


args_dict.update({'output_dir': "./" + model_prefix + "_final", 'num_train_epochs':2,
             'train_batch_size': 1, 'eval_batch_size': 1})
args = argparse.Namespace(**args_dict)


## Define Checkpoint function
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath="./" + model_prefix + "_checkpoint", prefix=model_prefix, monitor="accuracy", mode="max", save_top_k=1
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    resume_from_checkpoint=args.resume_from_checkpoint,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    val_check_interval=args.val_check_interval,
    callbacks=[LoggingCallback()],
    # progress_bar_refresh_rate=0
)

model = LonformerFineTuner(args)
trainer = pl.Trainer(**train_params)

log('Training started...')

trainer.fit(model)

log('Training done')

import textwrap
from tqdm.auto import tqdm
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from datasets import load_metric
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from transformers import (
    AdamW,
    LongformerTokenizer,
    LongformerForSequenceClassification,
    get_linear_schedule_with_warmup
)
import logging
import argparse
import time

model_path = 'longformer-4096_final/longformer-4096'
tokenizer = LongformerTokenizer.from_pretrained(model_path)
model = LongformerForSequenceClassification.from_pretrained(model_path, return_dict=True, num_labels=2)

import torch
train_dataset = torch.load('train_data.pt')
val_dataset = torch.load('valid_data.pt')
loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
it = iter(loader)

batch = next(it)
batch["input_ids"].shape

batch["input_ids"] = batch["input_ids"].to('cuda')
batch["attention_mask"] = batch["attention_mask"].to('cuda')
batch["targets"] = batch["targets"].to('cuda')
model.to('cuda')
outputs = model.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

labels = batch['targets']
logits = F.softmax(outputs['logits'], dim=1)
print(logits)
preds = torch.argmax(logits, 1)
preds = preds.cpu().data.numpy()
targets = labels.cpu().data.numpy()    

texts = [tokenizer.decode(ids) for ids in batch['input_ids']]

for i in range(1):
    lines = textwrap.wrap("Input Text:\n%s\n" % texts[i], width=100)
    print("\n".join(lines))
    print("\nActual Class: %s" % targets[i])
    print("\nPredicted Class from Longformer: %s" % preds[i])
    print("=====================================================================\n")

predictions = []
actual_labels = []

for batch in loader:
    batch["input_ids"] = batch["input_ids"].to('cuda')
    batch["attention_mask"] = batch["attention_mask"].to('cuda')
    batch["targets"] = batch["targets"]
    model.to('cuda')
    outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

    labels = batch['targets']
    logits = F.softmax(outputs['logits'], dim=1)
    preds = torch.argmax(logits, 1)
    preds = preds.cpu().data.numpy()
    targets = labels.cpu().data.numpy()
    for i in range(len(batch)):
        predictions.append(preds)
        actual_labels.append(targets)


    do_print = False
    if do_print:  

        texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
        
        lines = textwrap.wrap("Input Text:\n%s\n" % texts[i], width=100)
        print("\n".join(lines))
        print("\nActual Class: %s" % targets[i])
        print("\nPredicted Class from Longformer: %s" % preds[i])
        print("=====================================================================\n")

from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(actual_labels, predictions))

print(classification_report(actual_labels, predictions))


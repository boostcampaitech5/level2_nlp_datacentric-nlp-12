# 1. Import libraries
import os
import pytz
import sys
from datetime import datetime

import gluonnlp as nlp  # 데이터 가공 library
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datasets import load_dataset
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup   # 안정적인 학습을 위한 lr scheduler

from args import parse_arguments
from preprocess import preprocess_df
from util import start_wandb, finish_wandb, calc_accuracy


class BERTDataset(Dataset):
    """Define custom BERT dataset."""
    def __init__(self, dataset, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        texts = dataset["input_text"].tolist()
        targets = dataset["target"].tolist()

        self.sentences = [transform([i]) for i in texts]
        self.labels = [np.int32(i) for i in targets]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    """Define custom BERT classifier."""
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def get_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.get_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)


def main():
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "./config.yaml"
    config = parse_arguments(config_path)

    now = datetime.now(pytz.timezone("Asia/Seoul"))
    run_name = f"{config.run_name}_{now.strftime('%d-%H-%M')}"

    # 2. Set Hyperparameters
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("DEVICE:", DEVICE)

    BASE_DIR = os.getcwd()
    SEED = 42

    ## Setting parameters
    max_len = 64
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 5
    max_grad_norm = 1
    log_interval = 200
    learning_rate =  5e-5

    # 3. Load Tokenizer and Model
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    
    # 4. Define Dataset
    column_names = ["ID", "input_text", "label_text", "target", "predefined_news_category", "annotations", "url", "date"]
    revision = config.dataset["revision"]

    train = load_dataset("Smoked-Salmon-s/TC_Competition",
                        split="train",
                        column_names=column_names,
                        revision=revision)
    dataset_train = train.to_pandas().iloc[1:].reset_index(drop=True).astype({"target": "int64"})

    valid = load_dataset("Smoked-Salmon-s/TC_Competition",
                         split="validation",
                         column_names=column_names,
                         revision=revision)
    dataset_eval = valid.to_pandas().iloc[1:].reset_index(drop=True).astype({"target": "int64"})
    
    # 5. Preprocess data
    dataset_train = preprocess_df(dataset_train)
    dataset_eval = preprocess_df(dataset_eval)
    
    data_train = BERTDataset(dataset_train, tok, max_len, True, False)
    data_eval = BERTDataset(dataset_eval, tok, max_len, True, False)

    train_dataloader = DataLoader(data_train, batch_size=batch_size)
    eval_dataloader = DataLoader(data_eval, batch_size=batch_size)
    
    # 6. Define Model
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(DEVICE)
    
    # 7. Define Optimizer and Scheduler
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    # 8. Train and Evaluate Model
    start_wandb(config, run_name)  # Notify the beginning of experiment to WandB
    
    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0

        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            optimizer.zero_grad()
            
            token_ids = token_ids.long().to(DEVICE)
            segment_ids = segment_ids.long().to(DEVICE)
            valid_length= valid_length
            label = label.long().to(DEVICE)
            
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print(f"epoch {e+1} batch id {batch_id+1} loss {loss.data.cpu().numpy()} train acc {train_acc / (batch_id+1)}")
            wandb.log({"Train Loss": loss.data.cpu().numpy(), "Train Accuracy": train_acc / (batch_id+1)})
        print(f"epoch {e+1} train acc {train_acc / (batch_id+1)}")

        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            token_ids = token_ids.long().to(DEVICE)
            segment_ids = segment_ids.long().to(DEVICE)
            valid_length= valid_length
            label = label.long().to(DEVICE)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
            wandb.log({"Eval Accuracy": test_acc / (batch_id+1)})
        print(f"epoch {e+1} test acc {test_acc / (batch_id+1)}")

    finish_wandb(config, run_name)  # Notify the end of experiment to WandB

    # 9. Test Model
    test = load_dataset("Smoked-Salmon-s/TC_Competition",
                        split="test",
                        column_names=column_names,
                        revision=revision)
    dataset_test = test.to_pandas().iloc[1:].reset_index(drop=True)
    dataset_test["target"] = [0] * len(dataset_test)

    data_test = BERTDataset(dataset_test, tok, max_len, True, False)
    test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    preds = []
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, _) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        token_ids = token_ids.long().to(DEVICE)
        segment_ids = segment_ids.long().to(DEVICE)
        valid_length= valid_length
        out = model(token_ids, valid_length, segment_ids)
        max_vals, max_indices = torch.max(out, 1)
        preds.extend(list(max_indices))

    print(f"epoch {e+1} test acc {test_acc / (batch_id+1)}")
    preds = [int(p) for p in preds]

    dataset_test["target"] = preds
    output_path = os.path.join(BASE_DIR, config.data_path["output_csv_path"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset_test.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

# Import libraries
import argparse
import os
import pytz
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import load_config
from dataset import BERTDataset
from preprocess import preprocess_df
from tokenization_kobert import KoBertTokenizer
from util import compute_metrics


def main():
    # Load config
    parser = argparse.ArgumentParser()
    
    parser.add_argument('config_file', type=str, default='config.yaml',
                        nargs='?', help='The path to the config file')

    args = parser.parse_args()

    config = load_config(args.config_file)

    # Set Hyperparameters
    SEED = 456
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(DEVICE)

    # BASE_DIR = os.getcwd()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '../data'))
    OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../output'))
    PREDICTION_DIR = os.path.abspath(os.path.join(BASE_DIR, '../prediction'))

    column_names = ['ID', 'text', 'target', 'url', 'date']  # For huggingface load_dataset function setting

    # Load tokenizer and model
    model_name = 'monologg/kobert'
    tokenizer = KoBertTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

    # Load data
    if config.use_local_dataset:
        data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        dataset_train, dataset_valid = train_test_split(data, test_size=0.3, random_state=SEED)
    else:
        train = load_dataset('Smoked-Salmon-s/TC_Competition',
                             split='train',
                             column_names=column_names,
                             revision=config.huggingface_datasets.revision)
        dataset_train = train.to_pandas().iloc[1:].reset_index(drop=True).astype({'target': 'int64'})

        valid = load_dataset('Smoked-Salmon-s/TC_Competition',
                             split='validation',
                             column_names=column_names,
                             revision=config.huggingface_datasets.revision)
        dataset_valid = valid.to_pandas().iloc[1:].reset_index(drop=True).astype({'target': 'int64'})

    # Preprocess data
    dataset_train = preprocess_df(dataset_train, config.dataset.hanja, config.dataset.special, config.dataset.jonghab)
    dataset_valid = preprocess_df(dataset_valid, config.dataset.hanja, config.dataset.special, config.dataset.jonghab)

    # Define dataset
    data_train = BERTDataset(dataset_train, tokenizer, config.dataset.max_seq_len)
    data_valid = BERTDataset(dataset_valid, tokenizer, config.dataset.max_seq_len)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set wandb initial settings
    report_to = 'wandb' if config.use_wandb else 'none'

    if config.use_wandb:
        now = datetime.now(pytz.timezone('Asia/Seoul'))
        exp_name = f'{config.wandb.name}_{now.strftime("%d-%H-%M")}'

        wandb.init(entity=config.wandb.entity,
                   name=exp_name,
                   project=config.wandb.project,
                   config=config)

    # Train model
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy='steps',
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        learning_rate=2e-05,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        seed=SEED,
        fp16=config.trainer.fp16,
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if config.use_wandb:
        wandb.finish()

    # Evaluate model
    if config.use_local_dataset:
        dataset_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    else:
        test = load_dataset('Smoked-Salmon-s/TC_Competition',
                            split='test',
                            column_names=column_names,
                            revision=config.huggingface_datasets.revision)
        dataset_test = test.to_pandas().iloc[1:].reset_index(drop=True)

        dataset_test = dataset_test.iloc[:, :-1]
        column_names.remove('target')
        dataset_test.columns = column_names

    SUBMISSION_FILEPATH = os.path.join(DATA_DIR, 'sample_submission.csv')
    submission = pd.read_csv(SUBMISSION_FILEPATH)
    submission = submission.rename(columns={'label': 'target'})

    model.eval()
    preds = []
    for _, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test)):
        inputs = tokenizer(sample['text'], return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)

    # Prediction
    PREDICTION_FILEPATH = os.path.join(PREDICTION_DIR, config.output_filename.prediction)
    os.makedirs(os.path.dirname(PREDICTION_FILEPATH), exist_ok=True)
    submission['target'] = preds
    submission.to_csv(PREDICTION_FILEPATH, index=False)

    # Validation set post-analysis
    dev_preds = []
    for _, sample in tqdm(dataset_valid.iterrows(), total=len(dataset_valid)):
        inputs = tokenizer(sample['text'], return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            dev_preds.extend(pred)

    VALID_PREDICTION_FILEPATH = os.path.join(PREDICTION_DIR, config.output_filename.validation_prediction)
    dataset_valid['pred'] = dev_preds
    dataset_valid = dataset_valid[['ID', 'text', 'target', 'pred', 'url', 'date']]
    dataset_valid.to_csv(VALID_PREDICTION_FILEPATH, index=False)


if __name__ == '__main__':
    main()
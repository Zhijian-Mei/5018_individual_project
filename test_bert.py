import argparse
import random

import torch
from torch import nn, cuda
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, BertModel, T5Config, BertTokenizer, \
    BertForSequenceClassification
import jsonlines
from data_util import *
from sklearn.metrics import accuracy_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-mode', type=str, default='c')
    parser.add_argument('-prompt', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    gpu = args.gpu
    device = torch.device(f'cuda:{gpu}' if cuda.is_available() else 'cpu')

    model_name = 'bert-base-uncased'
    print(f'Backbone model name: {model_name}')

    labels = []
    premises = []
    hypothesises = []

    print('loading test data')
    with jsonlines.open('data/dev_matched_sampled-1.jsonl') as f:
        for line in f.iter():
            label = line['gold_label']
            premise = line['sentence1']
            hypothesis = line['sentence2']
            labels.append(label)
            premises.append(premise)
            hypothesises.append(hypothesis)

    matched_dataset = testDataset(premises, hypothesises, labels, mode=args.mode, prompt=True if args.prompt else False)
    matched_dataLoader = DataLoader(matched_dataset, batch_size=args.batch_size)

    labels = []
    premises = []
    hypothesises = []

    with jsonlines.open('data/dev_mismatched_sampled-1.jsonl') as f:
        for line in f.iter():
            label = line['gold_label']
            premise = line['sentence1']
            hypothesis = line['sentence2']
            labels.append(label)
            premises.append(premise)
            hypothesises.append(hypothesis)

    mismatched_dataset = testDataset(premises, hypothesises, labels, mode=args.mode,
                                     prompt=True if args.prompt else False)
    mismatched_dataLoader = DataLoader(mismatched_dataset, batch_size=args.batch_size)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False,
    ).to(device)

    ## load fine-tuned checkpoint

    checkpoint = torch.load('checkpoint/best_t5-small_epoch0_0.33_g_1.pt')123123
    model.load_state_dict(checkpoint['model'])

    predicts = []
    labels = []


    def f(x):
        if 'entailment' in x:
            return 0
        elif 'neutral' in x:
            return 1
        elif 'contradiction' in x:
            return 2
        else:
            return random.choice([0, 1, 2])


    for i in tqdm(matched_dataLoader):
        text, output = i[0], i[1]


        input_ = tokenizer.batch_encode_plus(
            text,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        outputs = model(input_ids=input_.input_ids, attention_mask=input_.attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu()

        labels.extend(output.tolist())
        predicts.extend(preds.tolist())

    labels = list(map(f, labels))

    accuracy = round(accuracy_score(labels, predicts), 2)
    print(f': accuracy {accuracy} for match_dataset')

    predicts = []
    labels = []
    
    for i in tqdm(mismatched_dataLoader):
        text, output = i[0], i[1]

        input_ = tokenizer.batch_encode_plus(
            text,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        outputs = model(input_ids=input_.input_ids, attention_mask=input_.attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu()

        labels.extend(output.tolist())
        predicts.extend(preds.tolist())

    labels = list(map(f, labels))

    accuracy = round(accuracy_score(labels, predicts), 2)
    print(f': accuracy {accuracy} for mismatch_dataset')

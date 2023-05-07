import argparse
import random

import torch
from torch import nn, cuda
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, BertModel,T5Config
import jsonlines
from data_util import *
from sklearn.metrics import accuracy_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-mode', type=str, default='g')
    parser.add_argument('-prompt', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    gpu = args.gpu
    device = torch.device(f'cuda:{gpu}' if cuda.is_available() else 'cpu')

    model_name = 't5-small'
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    ## load fine-tuned checkpoint

    checkpoint = torch.load('checkpoint/best_t5-small_epoch0_0.33_g_12.pt')
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

        labels.extend(list(output))
        input_ = tokenizer.batch_encode_plus(
            text,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(input_ids=input_.input_ids, attention_mask=input_.attention_mask,do_sample=False)

        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        predicts.extend(output_texts)
        print(output_texts)
        print(output)
        print(list(map(f, output_texts)))
        print(list(map(f, output)))


    labels = list(map(f, labels))
    predicts = list(map(f, predicts))

    accuracy = round(accuracy_score(labels, predicts), 2)
    print(f': accuracy {accuracy} for match_dataset')

    for i in tqdm(mismatched_dataLoader):
        text, output = i[0], i[1]

        labels.extend(list(output))
        input_ = tokenizer.batch_encode_plus(
            text,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(input_ids=input_.input_ids, attention_mask=input_.attention_mask,do_sample=False)

        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        predicts.extend(output_texts)

    labels = list(map(f, labels))
    predicts = list(map(f, predicts))

    accuracy = round(accuracy_score(labels, predicts), 2)
    print(f': accuracy {accuracy} for mismatch_dataset')

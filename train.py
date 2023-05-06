import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_util import *
from torch import nn, cuda
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForTokenClassification
from model import *
from evaluation import *

from transformers import AutoTokenizer, T5ForConditionalGeneration

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-gpu', type=str, default = '0')
    parser.add_argument('-mode', type=str, default='generation')
    parser.add_argument('-prompt', type=int, default=0)
    args = parser.parse_args()
    return args


def get_token_labal(input_encoding, label, max_length):
    attention_mask = input_encoding['attention_mask']
    golden_labels = []
    for j in range(input_encoding['input_ids'].shape[0]):
        label_for_token = [-100 for _ in range(max_length)]
        for k in range(max_length):
            if attention_mask[j][k] == 1:
                label_for_token[k] = 0
            else:
                break
            if input_encoding.token_to_chars(j, k) is None:
                label_for_token[k] = -100
                continue
            start, end = input_encoding.token_to_chars(j, k)
            for position in label[j]:
                if position == -1:
                    break
                if start <= position < end:
                    label_for_token[k] = 1
                    break
        golden_labels.append(label_for_token)
    return golden_labels


def get_char_label(input_encoding, label, text_length, max_length=1024):
    golden_labels = []
    for j in range(input_encoding['input_ids'].shape[0]):
        label_for_char = [-100 for _ in range(max_length)]
        for k in range(max_length):
            if k < text_length[j]:
                label_for_char[k] = 0
        for position in label[j]:
            label_for_char[int(position)] = 1
        golden_labels.append(label_for_char)
    return golden_labels


if __name__ == '__main__':
    args = get_args()
    print(args)
    gpu = args.gpu
    device = torch.device(f'cuda:{gpu}' if cuda.is_available() else 'cpu')

    model_name = 't5-small'
    print(f'Backbone model name: {model_name}')

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # print(inputs['input_ids'].shape)
    # outputs = model(**inputs)
    #
    # last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states.shape)

    print('loading data')

    dataset = MyDataset(tokenizer,mode =args.mode,prompt=True if args.prompt else False)

    train_batch_size = args.batch_size
    eval_batch_size = args.batch_size
    train_set,eval_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=eval_batch_size)

    epoch = 20
    global_step = 0

    for e in range(epoch):
        model.train()
        for i in tqdm(train_loader,
                      # mininterval=200
                      ):
            text, output = i[0], i[1]
            print(text)
            print(output)
            quit()
            input_encoding = tokenizer.batch_encode_plus(
                text,
                max_length=256,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            golden_labels = get_token_labal(input_encoding, label, max_length)
            # golden_labels = get_char_label(input_encoding,label,text_length)
            # for j in range(len(golden_labels)):
            #     print(golden_labels[j])
            #     print(label[j])
            # quit()
            golden_labels = torch.LongTensor(golden_labels).to(device)
            logits, loss = model(input_encoding, golden_labels)

            # print(logits.argmax(-1).cpu().tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 100 == 0:
                print('loss: ', loss.item())

        f1score = 0
        count = 0
        model.eval()
        for i in tqdm(eval_loader,mininterval=200):
            text, label, _ = i[0], i[1], i[2]
            input_encoding = tokenizer.batch_encode_plus(
                text,
                max_length=max_length,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            logits = model(input_encoding)
            predicted_token_class_ids = logits.argmax(-1)
            label = label.tolist()
            predicted_labels = []
            for j in range(len(label)):
                label[j] = [int(it) for it in label[j] if it != -1]
            for j in range(input_encoding['input_ids'].shape[0]):
                label_for_char = []
                for k in range(1, max_length):
                    if predicted_token_class_ids[j][k] == 1 and input_encoding['attention_mask'][j][k] == 1:
                        start, end = input_encoding.token_to_chars(j, k)
                        for position in range(start, end):
                            label_for_char.append(position)
                predicted_labels.append(label_for_char)
            # if len(predicted_labels) != 0:
            #     print(predicted_labels)
            for j in range(len(predicted_labels)):
                f1score += f1(predicted_labels[j], label[j])
                count += 1

        f1score = f1score / count
        print(f'f1_score: {f1score} at epoch {e}')
        torch.save({'model': model.state_dict()}, f"checkpoint/{model_name}_epoch{e}_{'freeze' if args.freeze == 1 else 'unfreeze'}.pt")
        if f1score > best_f1:
            best_f1 = f1score
            torch.save({'model': model.state_dict()},
                       f"checkpoint/best_{model_name}_epoch{e}_f1:{round(best_f1, 3)}_{'freeze' if args.freeze == 1 else 'unfreeze'}.pt")
            print('saving better checkpoint')

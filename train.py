import argparse
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_util import *
from torch import nn, cuda
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
from model import *
from evaluation import *
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, T5ForConditionalGeneration


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-mode', type=str, default='g')
    parser.add_argument('-prompt', type=int, default=0)
    parser.add_argument('-lr', type=float, default=2e-5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    gpu = args.gpu
    device = torch.device(f'cuda:{gpu}' if cuda.is_available() else 'cpu')

    model_name = 't5-small'
    print(f'Backbone model name: {model_name}')

    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('loading data')

    dataset = MyDataset(tokenizer, mode=args.mode, prompt=True if args.prompt else False)

    train_batch_size = args.batch_size
    eval_batch_size = args.batch_size
    train_set, eval_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    print('number of training sample: ',len(train_set))
    print('number fo eval sample: ',len(eval_set))
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=eval_batch_size,shuffle=False)

    epoch = 5
    global_step = 0


    def f(x):
        if 'entailment' in x:
            return 0
        elif 'neutral' in x:
            return 1
        elif 'contradiction' in x:
            return 2
        else:
            return random.choice([0, 1, 2])


    best_accuracy = -np.inf
    for e in range(epoch):
        model.train()
        epoch_loss = 0
        for i in tqdm(train_loader,
                      mininterval=200
                      ):
            text, output = i[0], i[1]

            input_ = tokenizer.batch_encode_plus(
                text,
                max_length=128,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            output_ = tokenizer.batch_encode_plus(
                output,
                max_length=128,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            labels = output_.input_ids
            labels[labels == 0] = -100

            loss = model(**input_, labels=labels).loss
            epoch_loss += loss.item()
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # outputs = model.generate(**input_)
            # output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print(text)
            # print(list(output))
            # print(output_texts)
            # print(round(accuracy_score(list(map(f, list(output))), list(map(f, output_texts))), 2))

            global_step += 1

        #     if global_step % 1 == 0:
        #         break
        # continue
        print(f'loss at epoch {e}: {epoch_loss}')
        model.eval()
        predicts = []
        labels = []
        accuracy = None
        for i in tqdm(eval_loader,
                      mininterval=200
                      ):
            text, output = i[0], i[1]

            input_ = tokenizer.batch_encode_plus(
                text,
                max_length=128,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            outputs = model.generate(**input_)

            output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels.extend(list(output))
            predicts.extend(output_texts)
            # print(text)
            # print(list(output))
            # print(output_texts)
            # print(round(accuracy_score(list(map(f, list(output))), list(map(f, output_texts))), 2))

            # for output_text in output_texts:
            #     if 'entailment' in output_text:
            #         results.append(0)
            #     elif 'neutral' in output_text:
            #         results.append(1)
            #     elif 'contradiction' in output_text:
            #         results.append(2)
            #     else:
            #         results.append(random.choice([0,1,2]))
            # if len(labels) > 500:
            #     break
        labels = list(map(f, labels))
        predicts = list(map(f, predicts))

        accuracy = round(accuracy_score(labels, predicts), 2)

        print(f': accuracy {accuracy} at epoch {e}')
        torch.save({'model': model.state_dict()},
                   f"checkpoint/{model_name}_{accuracy}_epoch{e}_{args.mode}_{args.prompt}.pt")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({'model': model.state_dict()},
                       f"checkpoint/best_{model_name}_epoch{e}_{accuracy}_{args.mode}_{args.prompt}.pt")
            print('saving better checkpoint')

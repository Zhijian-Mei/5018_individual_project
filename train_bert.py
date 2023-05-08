import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer, BertTokenizer, BertConfig, AutoConfig,BertForSequenceClassification
from torch import cuda, nn
from data_util import *
from sklearn.metrics import accuracy_score
from model import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-mode', type=str, default='c')
    parser.add_argument('-prompt', type=int, default=0)
    parser.add_argument('-lr', type=float, default=0.005)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    gpu = args.gpu
    device = torch.device(f'cuda:{gpu}' if cuda.is_available() else 'cpu')

    model_name = 'bert-base-uncased'


    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name,num_labels = 3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('loading data')

    dataset = MyDataset(tokenizer, mode=args.mode, prompt=True if args.prompt else False)

    train_batch_size = args.batch_size
    eval_batch_size = args.batch_size
    train_set, eval_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=eval_batch_size)

    epoch = 100
    global_step = 0
    loss_fn = nn.CrossEntropyLoss()
    for e in range(epoch):
        model.train()
        for i in tqdm(train_loader,
                      mininterval=200
                      ):
            text, output = i[0], i[1].to(device)

            input_ = tokenizer.batch_encode_plus(
                text,
                max_length=512,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            outputs = model(**input_,labels=output)
            logits = outputs.logits
            loss = outputs.loss
            preds = torch.argmax(logits, dim=1).float()
            print(preds.tolist())
            print(output.cpu().tolist())
            print('loss: ', loss.item())
            print()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 1 == 0:
                break

        continue
        model.eval()
        predicts = []
        labels = []
        best_accuracy = -np.inf
        for i in tqdm(eval_loader,
                      mininterval=200
                      ):
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

            print(output.tolist())
            print(preds.tolist())

            labels.extend(output.tolist())
            predicts.extend(preds.tolist())

        accuracy = round(accuracy_score(labels, predicts), 2)

        print(f': accuracy {accuracy} at epoch {e}')

        torch.save({'model': model.state_dict()},
                   f"checkpoint/{model_name}_{accuracy}_epoch{e}_{args.mode}_{args.prompt}.pt")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({'model': model.state_dict()},
                       f"checkpoint/best_{model_name}_epoch{e}_{accuracy}_{args.mode}_{args.prompt}.pt")
            print('saving better checkpoint')

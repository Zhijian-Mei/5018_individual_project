import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer, BertTokenizer, BertConfig, AutoConfig
from torch import cuda, nn
from data_util import *
from sklearn.metrics import accuracy_score
from model import *
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-mode', type=str, default='c')
    parser.add_argument('-prompt', type=int, default=0)
    parser.add_argument('-lr', type=float, default=2e-5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    gpu = args.gpu
    device = torch.device(f'cuda:{gpu}' if cuda.is_available() else 'cpu')

    model_name = 'bert-base-uncased'

    config = BertConfig.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # model = BertForSequenceClassification.from_pretrained(model_name,num_labels = 3).to(device)
    bert = BertModel.from_pretrained(model_name).to(device)

    model = MyModel(bert,config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('loading data')

    dataset = MyDataset(tokenizer, mode=args.mode, prompt=True if args.prompt else False)

    train_batch_size = args.batch_size
    eval_batch_size = args.batch_size
    train_set, eval_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=eval_batch_size)

    epoch = 3
    global_step = 0
    loss_fn = nn.CrossEntropyLoss()
    for e in range(epoch):
        model.train()
        epoch_loss = 0
        for i in tqdm(train_loader,
                      # mininterval=200
                      ):
            text, output = i[0], i[1].to(device)

            input_ = tokenizer.batch_encode_plus(
                text,
                max_length=256,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            logits, loss = model(input_ ,labels=output)

            # logits = model_output.logits
            # loss = loss_fn(logits,output)

            # preds = torch.argmax(logits, dim=1)

            epoch_loss += loss.item()
            print(loss.item())
            # print(preds)
            # print(output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # if global_step % 2 == 0:
            #     break

        print(f'loss at epoch {e}: {epoch_loss}')
        model.eval()
        predicts = []
        labels = []
        best_accuracy = -np.inf
        for i in tqdm(train_loader,
                      # mininterval=200
                      ):
            text, output = i[0], i[1].to(device)

            input_ = tokenizer.batch_encode_plus(
                text,
                max_length=256,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            logits = model(input_)

            preds = torch.argmax(logits, dim=1).float()

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

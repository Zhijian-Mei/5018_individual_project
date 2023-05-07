import argparse

import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, AutoTokenizer, BertTokenizer
from torch import cuda
from data_util import *


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

    model_name = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels = 3,
        output_attentions = False,
        output_hidden_states = False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print('loading data')

    dataset = MyDataset(tokenizer, mode=args.mode, prompt=True if args.prompt else False)

    train_batch_size = args.batch_size
    eval_batch_size = args.batch_size
    train_set, eval_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=eval_batch_size)

    epoch = 20
    global_step = 0

    for e in range(epoch):
        model.train()
        for i in tqdm(train_loader,
                      mininterval=200
                      ):
            text, output = list(i[0]), i[1]

            input_ = tokenizer.batch_encode_plus(
                text,
                max_length=256,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            outputs = model(input_ids=input_.input_ids, attention_mask=input_.attention_mask, labels=output_.input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # outputs = model.generate(input_ids=input_.input_ids, attention_mask=input_.attention_mask)
            #
            # output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # print(logits.argmax(-1).cpu().tolist())

            global_step += 1

            # if global_step % 100 == 0:
            #     break
            #     print('loss: ', loss.item())

import csv
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from data_util import *

# Define the query template and verbalizer functions
def query_template(premise, hypothesis):
    return f"NLI: premise: {premise} hypothesis: {hypothesis}"

def verbalizer(output):
    labels = ["entailment", "neutral", "contradiction"]
    label_id = torch.argmax(output).item()
    if label_id >= len(labels):
        return labels[-1] # return the default label if the index is out of range
    else:
        return labels[label_id]

# Load the pre-trained model and tokenizer
model_name = "t5-small"
device = torch.device(f'cuda:0')
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the training hyperparameters
batch_size = 4
learning_rate = 1e-4
num_epochs = 100

# Set up the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

dataset = MyDataset(tokenizer, mode='c', prompt=False)

train_batch_size = batch_size
eval_batch_size = batch_size
train_set, eval_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False)
eval_loader = DataLoader(eval_set, batch_size=eval_batch_size)

def f(x):
    if 'entailment' in x:
        return 0
    elif 'neutral' in x:
        return 1
    elif 'contradiction' in x:
        return 2
    else:
        return random.choice([0, 1, 2])
# Train the model

model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}:")
    total_loss = 0.0
    for i in tqdm(train_loader):
        batch_premises = train_premises[i:i+batch_size]
        batch_hypotheses = train_hypotheses[i:i+batch_size]
        output = train_labels[i:i+batch_size]

        # batch_labels = torch.tensor(train_labels[i:i+batch_size], dtype=torch.long)
        optimizer.zero_grad()
        input_text = [query_template(premise, hypothesis) for premise, hypothesis in zip(batch_premises, batch_hypotheses)]

        input_ =  tokenizer.batch_encode_plus(
            input_text,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            ).to(device)
        output_ = tokenizer.batch_encode_plus(
            output,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            ).to(device)
        labels = output_.input_ids
        labels[labels==0] = -100
        # input_ids = tokenizer(input_text, padding=True, return_tensors='pt')
        loss = model(input_.input_ids, attention_mask=input_.attention_mask, labels=labels).loss
        loss.backward()
        optimizer.step()

        outputs = model.generate(input_ids=input_.input_ids, attention_mask=input_.attention_mask)

        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(list(output))
        print(output_texts)
        print(round(accuracy_score(list(map(f, list(output))), list(map(f, output_texts))), 2))

        total_loss += loss.item()
    print(f"  Training loss: {total_loss:.3f}")

model.eval()

accuracy = 0.0
t = 0
# Train the model
for epoch in range(1):
    print(f"Epoch {epoch + 1}:")
    total_loss = 0.0
    for i in range(0, len(test_premises), batch_size):
        batch_premises = test_premises[i:i+batch_size]
        batch_hypotheses = test_hypotheses[i:i+batch_size]
        label = test_labels[i:i+batch_size]
        input_text = [query_template(premise, hypothesis) for premise, hypothesis in zip(batch_premises, batch_hypotheses)]
        # print(input_text)
        input_ =  tokenizer.batch_encode_plus(
            input_text,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            )
        output = model.generate(input_ids=input_.input_ids,attention_mask=input_.attention_mask)
        output_ = tokenizer.batch_decode(output, skip_special_tokens=True)
        k = 0
        count = len(output_)
        hit = 0
        for i in output_:
          if i == label[k]:
            hit += 1
          k += 1
        acc = hit/count
        print(output_)
        print(hit/count)
        accuracy += acc
        t += 1
print('final')
print(accuracy/t)


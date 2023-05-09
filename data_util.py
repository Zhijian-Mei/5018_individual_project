from datasets import load_dataset

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import trange


def get_data():
    dataset = load_dataset('multi_nli')['train']
    dataset.set_format(columns=["premise", "hypothesis", "label"])
    dataset = dataset[:2500]
    return dataset

def get_review():
    dataset = load_dataset('imdb')['train']
    dataset.set_format(columns=['text','label'])
    return dataset
class Spam_dataset(Dataset):
    def __init__(self):
        self.dataset = get_review()
        self.input = []
        self.output = []

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx],self.output[idx]
class MyDataset(Dataset):
    def __init__(self, tokenizer, mode='g',prompt = False):
        self.dataset = get_data()
        self.premises = self.dataset['premise']
        self.hypothesises = self.dataset['hypothesis']
        self.labels = self.dataset['label']
        self.mode = mode
        self.prompt = prompt
        self.input = []
        self.output = []
        for i in trange(len(self.premises)):
            premise = self.premises[i]
            hypothesis = self.hypothesises[i]
            label = self.labels[i]
            output_ = int(label)
            if self.prompt:
                input_ = f'mnli: premise: {premise} hypothesis: {hypothesis} the relation is [MASK]'
                if self.mode == 'g':
                    if int(label) == 0:
                        output_ = f'mnli: premise: {premise} hypothesis: {hypothesis} the relation is entailment.'
                    elif int(label) == 1:
                        output_ = f'mnli: premise: {premise} hypothesis: {hypothesis} the relation is neutral.'
                    elif int(label) == 2:
                        output_ = f'mnli: premise: {premise} hypothesis: {hypothesis} the relation is contradiction.'
            else:
                input_ = f'mnli: premise: {premise} hypothesis: {hypothesis}'
                if self.mode == 'g':
                    if int(label) == 0:
                        output_ = f'entailment'
                    elif int(label) == 1:
                        output_ = f'neutral'
                    elif int(label) == 2:
                        output_ = f'contradiction'

            self.input.append(input_)
            self.output.append(output_)
        assert len(self.input) == len(self.output)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx],self.output[idx]

class testDataset(Dataset):
    def __init__(self,premises,hypothesises,labels, mode='g',prompt = False):
        self.premises = premises
        self.hypothesises = hypothesises
        self.labels = labels
        self.mode = mode
        self.prompt = prompt
        self.input = []
        self.output = []
        for i in range(len(self.premises)):
            premise = self.premises[i]
            hypothesis = self.hypothesises[i]
            label = self.labels[i]
            output_ = label
            if self.prompt:
                input_ = f'mnli: premise: {premise} hypothesis: {hypothesis} the relation is [MASK]'
                if self.mode == 'g':
                    output_ = f'mnli: premise: {premise} hypothesis: {hypothesis} the relation is {label}.'
            else:
                input_ = f'mnli: premise: {premise} hypothesis: {hypothesis}'
            self.input.append(input_)
            self.output.append(output_)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx],self.output[idx]

if __name__ == '__main__':
    dataset = get_data()
    print(dataset)
    labels = dataset['label']
    label0 = list(filter(lambda x:x==0,labels))
    label1 = list(filter(lambda x: x == 1, labels))
    label2 = list(filter(lambda x: x == 2, labels))
    print(len(label0),len(label1),len(label2))

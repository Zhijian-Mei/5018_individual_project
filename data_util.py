from datasets import load_dataset

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import trange


def get_data():
    dataset = load_dataset('multi_nli')['train']
    dataset.set_format(columns=["premise", "hypothesis", "label"])
    return dataset


class MyDataset(Dataset):
    def __init__(self, tokenizer, mode='generation',prompt = False):
        self.dataset = get_data()
        self.premises = self.dataset['premise']
        self.hypothesises = self.dataset['hypothesis']
        self.labels = self.dataset['label']
        self.mode = mode
        self.prompt = prompt
        self.input = []
        self.output = []
        for i in range(len(self.premises)):
            premise = self.premises[i]
            hypothesis = self.premises[i]
            label = self.labels[i]
            print(premise)
            quit()
            if self.prompt:
                input_ = f'what is the relation from the first sentence to the second sentence: {premise};{hypothesis}?'
                if self.mode == 'generation':
                    output_ = f'the relation is {label}'
                else:
                    output_ = label
            else:
                input_ = f'{premise};{hypothesis}'
                if self.mode == 'generation':
                    output_ = f'{label}'
                else:
                    output_ = label
            self.input.append(input_)
            self.output.append(output_)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx],self.output[idx]



if __name__ == '__main__':
    train_loader = get_data()
    for i in train_loader:
        print(i)
        quit()

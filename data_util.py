from datasets import load_dataset

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

def get_data():
    dataset = load_dataset('multi_nli')['train']
    dataset.set_format(type="torch", columns=["premise", "hypothesis","label"])
    return dataset

class MyDataset(Dataset):
    def __init__(self,tokenizer):
        self.dataset = get_data()
        self.input = []
        self.output = self.dataset['label']
        print(self.output)
        quit()
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        pass

if __name__ == '__main__':
    train_loader = get_data()
    for i in train_loader:
        print(i)
        quit()
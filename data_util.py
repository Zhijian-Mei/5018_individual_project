from datasets import load_dataset

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

def get_data():
    dataset = load_dataset('multi_nli')['train']
    dataset.set_format(type="torch", columns=["premise", "hypothesis","label"])
    print(dataset)
    dataloader = DataLoader(dataset, batch_size=4)
    return dataloader

if __name__ == '__main__':
    train_loader = get_data()
    for i in train_loader:
        print(i)
        quit()
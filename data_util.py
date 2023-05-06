from datasets import load_dataset

import pandas as pd
from torch.utils.data import Dataset
from tqdm import trange

def get_data():
    dataset = load_dataset('multi_nli')
    print(dataset)

if __name__ == '__main__':
    get_data()
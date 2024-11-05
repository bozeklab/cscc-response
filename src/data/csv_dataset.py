
from torch.utils.data import Dataset
import pandas as pd

class CSVDFDataset(Dataset):
    """Dataset class for a .csv that represents a DataFrame, and returns each row as the samples
    
    Arguments:
    csv_file -- filename
    filter_func -- handle to function to filter rows beforehnd
    transform -- transform to apply
    """

    def __init__(self, csv_file, filter_func = None, transform = None): 
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        if filter_func is not None:
            self.df = self.df[self.df.apply(filter_func, axis=1)]
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = self.df.iloc[idx].to_dict()
        if self.transform != None:
            sample = self.transform(sample)
        return sample
    
    def df(self):
        return self.df
import pickle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate
import os

def process_row(row_dict, keys):
    with open(row_dict['filename'], 'rb') as fp:
        sample_dict = pickle.load(fp)
    out_tuple = ()
    for key in keys:
        val = sample_dict.get(key, row_dict.get(key, None))
        if val is not None:
            out_tuple += (val,)
    return out_tuple

def collate_and_pad(samples): #x =list of tuple of tensor,tensor (which is the sample and the label)
    sequences = [s[0] for s in samples]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=-1.)
    other_info = default_collate([s[1:] for s in samples])
    out_tuple = [padded_sequences] + other_info 
    return out_tuple

def filter_fns(row):
    return os.path.isfile(row['filename'])
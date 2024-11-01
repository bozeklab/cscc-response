import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate
import os

def read_file(fn, mode):
    with open(fn, mode) as fp:
        return fp.read()

def unpickle(fn, mode):
    with open(fn, mode) as fp:
        return pickle.load(fp)

def unpickle_untuple(fn, mode):
    with open(fn, mode) as fp:
        return pickle.load(fp)[0]

def unpickle_dict_and_get_value(fn, mode='rb', key='features'):
    with open(fn, mode) as fp:
        return pickle.load(fp)[key]

def unpickle_filename_and_keep_some_values(row_dict, cols_to_keep):
    sample = unpickle_dict_and_get_value(row_dict['filename'])
    out_tuple = (sample,)
    for col in cols_to_keep:
        out_tuple += (row_dict[col],)
    return out_tuple

def cast_tensor(x, dtype):
    return x.to(dtype)

def to_float(x):
    return x.float()

def collate_and_pad(samples): #x =list of tuple of tensor,tensor (which is the sample and the label)
    sequences = [s[0] for s in samples]
    # labels = torch.tensor([s[1] for s in samples])
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=-1.) #.cuda() #transfer to device!!! this doesnt work
    other_info = default_collate([s[1:] for s in samples])
    out_tuple = [padded_sequences] + other_info 
    return out_tuple


# this function only does the list comprehension of the function above. The idea is that the following steps are done in gpu.
# it throws a warning during training as pytorch-lightning tries to estimate the batch size of lists recursively untils it finds a tensor,
# however, the tensors it finds are not the batch, but the samples.
# easiest way to solve is by passing the batch size to pl_module.log. I added an argument to the LitModel constructor to do so
def collate(samples): #x =list of tuple of tensor,tensor (which is the sample and the label)
    sequences = [s[0] for s in samples]
    labels = torch.tensor([s[1] for s in samples])
    return (sequences, labels)

def filter_fns(row):
    return os.path.isfile(row['filename'])
import torch
from lightning.pytorch.callbacks import Callback
import pickle
from utils import move_tensors_to_cpu


class SaveExtraIO(Callback):
    def __init__(self, on_test=True, on_predict=False):
        super(SaveExtraIO, self).__init__()
        self.on_test = on_test
        self.on_predict = on_predict

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.on_test:
            return
        labels_and_logits = pl_module.all_steps_labels_and_logits
        additional_inputs = pl_module.all_steps_additional_inputs
        additional_outputs = pl_module.all_steps_additional_outputs

        to_pickle = {
            'labels_and_logits': move_tensors_to_cpu(labels_and_logits),
            'additional_inputs': move_tensors_to_cpu(additional_inputs),
            'additional_outputs': move_tensors_to_cpu(additional_outputs)
        }
        with open('test_data.pkl', 'wb') as h:
            pickle.dump(to_pickle, h)

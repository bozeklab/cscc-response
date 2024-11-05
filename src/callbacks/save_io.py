import torch
from lightning.pytorch.callbacks import Callback
import pickle
from ..utils import move_tensors_to_cpu

class SaveExtraIO(Callback):
    def __init__(self, on_val=False, on_test=True, on_predict=False):
        super(SaveExtraIO, self).__init__()
        self.on_validation_epoch_end = self.get_save_io_impl('val_extra_io.pickle', on_val)
        self.on_test_epoch_end = self.get_save_io_impl('test_extra_io.pickle', on_test)
        self.on_predict_epoch_end = self.get_save_io_impl('predict_extra_io.pickle', on_predict)

    def get_save_io_impl(self, filename, flag):
        def save_io_impl(trainer, pl_module):
            if not flag:
                return
            labels_and_logits = pl_module.all_steps_labels_and_logits
            additional_inputs = pl_module.all_steps_additional_inputs
            additional_outputs = pl_module.all_steps_additional_outputs

            to_pickle = {
                'labels_and_logits': move_tensors_to_cpu(labels_and_logits),
                'additional_inputs': move_tensors_to_cpu(additional_inputs),
                'additional_outputs': move_tensors_to_cpu(additional_outputs)
            }
            with open(filename, 'wb') as h:
                pickle.dump(to_pickle, h)
        return save_io_impl

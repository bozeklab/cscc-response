import torch
from captum.attr import IntegratedGradients
from collections import OrderedDict

# the integrated gradients object needs a model whose forward returns just a tensor
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        o = self.model(x)
        if isinstance(o, (tuple, list, OrderedDict)):
            o = o[0]
        return o

class IntegratedGradientsExplainer:
    def __init__(self, model):
        self.model = ModelWrapper(model)
        self.ig = IntegratedGradients(self.model)

    def make_explanation(self, features_batch, c=0, internal_batch_size=None):
        self.model.zero_grad()
        attributions, approximation_error = self.ig.attribute(features_batch, target=c, internal_batch_size=internal_batch_size, return_convergence_delta=True)
        attributions = attributions.cpu().detach().numpy()
        return attributions
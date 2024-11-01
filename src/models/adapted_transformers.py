import torch
from torch import nn

class SASequenceShortener(nn.Module):
    def __init__(self, target_len, **ma_kwargs):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(**ma_kwargs)
        self.query = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1, target_len, ma_kwargs['embed_dim']), 0., 0.2))

    def forward(self, x):
        key_padding_mask = None
        if isinstance(x,tuple):
            x, key_padding_mask = x
        x = self.multihead_attn(self.query.repeat(x.size(0),1,1), x, x,average_attn_weights=False, key_padding_mask=key_padding_mask) #this repeat method works for nested tensors and batched sequences
        x = (x[0] + self.query, x[1:])
        return x

#averages latent representations instead of using cls token
class AdaptedModelAvg(nn.Module):
    def __init__(self, seq_shortener, model, embed_dim=768, num_classes=2):
        super().__init__()
        self.seq_shortener = seq_shortener
        self.model = model
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x, output_attentions=False, output_last_hidden=False, output_last_hidden_avg=False):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x, seq_shortener_attentions = self.seq_shortener(x) # second tuple element would be the attention heads
        x = self.model(inputs_embeds=x, output_attentions=output_attentions)
        last_hidden = x[0]
        last_hidden_avg = last_hidden.mean(-2)
        o = (self.fc(last_hidden_avg),)
        if output_attentions:
            o += ((seq_shortener_attentions, * x['attentions']),)
        if output_last_hidden:
            o += (last_hidden,)
        if output_last_hidden_avg:
            o += (last_hidden_avg,)
        return o

def freeze_model(model):
    for name, param in model.named_parameters():
        if any([x in name for x in ['encoder','decoder','transformer']]):
            param.requires_grad = False
        if any([x in name for x in ['LayerNorm','layer_norm', '.ln_']]):
            param.requires_grad = True
    return model

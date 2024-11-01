import torch
import torch.nn as nn

def get_transformer_encoder(
    # encoder layer args
    d_model, 
    nhead, 
    # encoder model args
    num_layers,
    # encoder layer kwargs
    dim_feedforward=2048, 
    dropout=0.1, 
    activation='relu', 
    layer_norm_eps=1e-05, 
    batch_first=False, 
    norm_first=False, 
    device=None, 
    dtype=None,
    # encoder model kwargs
    norm=None, 
    enable_nested_tensor=True, 
    mask_check=True
    ):
    encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout,activation,layer_norm_eps,batch_first,norm_first,device,dtype)
    transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers,norm,enable_nested_tensor,mask_check)
    return transformer_encoder

def get_last_hipt_transformer():
    return get_transformer_encoder(192, 3, 2, dim_feedforward=768, activation='gelu', batch_first=True)

# wagner et al cancer cell 2023
class CancerCellTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(768,512)
        self.model = get_transformer_encoder(512,8,2,512, activation='gelu', batch_first=True)
        self.cls = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1, 1, 512), 0., 0.2)) # bs=1, seq_len=1, embed_dim
        self.mlp_head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, num_classes))
    
    def forward(self, x):
        x = self.proj(x)
        x = torch.cat([self.cls.repeat(x.shape[0],1,1), x], dim=1)
        x = self.model(x)
        x = self.mlp_head(x[:,0])
        return x

class TransformerEncoderClassifierAvgPooling(nn.Module):
    def __init__(
            self,
            # input proj_layer
            d_in,
            # encoder layer args
            d_model, 
            nhead, 
            # encoder model args
            num_layers,
            # encoder layer kwargs
            dim_feedforward=2048, 
            dropout=0.1, 
            activation='gelu', 
            layer_norm_eps=1e-05, 
            batch_first=False, 
            norm_first=False, 
            device=None, 
            dtype=None,
            # encoder model kwargs
            norm=None, 
            enable_nested_tensor=True, 
            mask_check=True,
            num_classes=2):
        super().__init__()
        self.in_layer = nn.Identity() if d_in==d_model else nn.Linear(d_in, d_model)
        self.transformer_encoder = get_transformer_encoder(
            # encoder layer args
            d_model, 
            nhead, 
            # encoder model args
            num_layers,
            # encoder layer kwargs
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation=activation, 
            layer_norm_eps=layer_norm_eps, 
            batch_first=batch_first, 
            norm_first=norm_first, 
            device=device, 
            dtype=dtype,
            # encoder model kwargs
            norm=norm, 
            enable_nested_tensor=enable_nested_tensor, 
            mask_check=mask_check
        )
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x, output_attentions=False, output_last_hidden=False):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.in_layer(x)
        x = self.transformer_encoder(x)
        o = self.fc(x.mean(-2))
        return o
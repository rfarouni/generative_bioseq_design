import torch
from typing import Any
import torch.nn.functional as F
from torch import Tensor, nn
import copy

class RMSNorm(nn.Module):
    def __init__(self, d_model, layer_norm_eps):
        super().__init__()
        self.layer_norm_eps=layer_norm_eps 
        self.d_model=d_model
        self.weight = nn.Parameter(torch.ones(d_model))
        self.register_parameter("weight", self.weight)
        #self.weight = self.weight.to(torch.float32)

    # def _norm(self, x: torch.Tensor):
    #     # (B, seq_len, d_model) -> (B, seq_len, 1)
    #     return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.layer_norm_eps)

    def forward(self, x: torch.Tensor):
        # dim : (B, seq_len, d_model) -> (B, seq_len, d_model)
        #return self.weight * self._norm(x.float()).type_as(x)
    
        y = x / (x.norm(2, dim=-1, keepdim=True) * self.d_model ** (-1.0 / 2) + self.layer_norm_eps)
        return self.weight * y


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network

    Args:
        d_model: the number of expected features in the input (required).
        n_head: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, n_head=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, n_head=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """


    def __init__(self, d_model, n_head, dim_feedforward, dropout, bias, layer_norm_eps) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, 
                                               n_head, 
                                               dropout=dropout,
                                               bias=bias, 
                                               batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        #self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=config.bias)
        #self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=config.bias)

        self.norm1 = RMSNorm(d_model, layer_norm_eps)
        self.norm2 = RMSNorm(d_model, layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    # self-attention block
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(
            self,
            src: Tensor) -> Tensor:

        x = src
   
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._ff_block(x))

        return x

    
class SequenceAttentionEncoder(nn.Module):
    def __init__(self, 
                 d_model: int=120, 
                 n_head: int =1, 
                 n_layer: int= 1, 
                 dim_feedforward: int=2048, 
                 dropout: float=0.02, 
                 bias: bool=False, 
                 layer_norm_eps: float= 1e-5):
        super(SequenceAttentionEncoder, self).__init__()
        self.encoder = self._build_transformer_encoder(d_model, n_head, n_layer,  dim_feedforward, dropout, bias, layer_norm_eps)
        self.fc = nn.Linear(d_model, 1, bias=bias)

    def _build_transformer_encoder(self, d_model, n_head, n_layer, dim_feedforward, dropout, bias, layer_norm_eps):
        # Create a TransformerEncoderLayer instance
        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, bias, layer_norm_eps)
        # Use _get_clones logic to create multiple layers
        mods = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layer)])
        return mods

    def forward(self, x: Tensor) -> Tensor:

        for mod in self.encoder:
            x = mod(x)
      
        logits = self.fc(x)
        return logits

if __name__ == "__main__":
    _ = SequenceAttentionEncoder()        



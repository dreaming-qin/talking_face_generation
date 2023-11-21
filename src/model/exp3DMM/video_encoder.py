from torch import nn
import torch
import torch.nn.functional as F
from src.util.model_util import reset_parameters
import copy
import numpy as np


class VideoEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pos_embed_len=80,
        input_dim=128,
        aggregate_method="average",
        **_
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        reset_parameters(self.encoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        self.increase_embed_dim = nn.Linear(input_dim, d_model)

        self.aggregate_method = None
        if aggregate_method == "self_attention_pooling":
            self.aggregate_method = SelfAttentionPooling(d_model)
        elif aggregate_method == "average":
            pass
        else:
            raise ValueError(f"Invalid aggregate method {aggregate_method}")

    def forward(self, x, pad_mask=None):
        """

        Args:
            x (_type_): (B, num_frames(L), C_exp)
            pad_mask: (B, num_frames)

        Returns:
            style_code: (B, C_model)
        """
        x = self.increase_embed_dim(x)
        # (B, L, C)
        x = x.permute(1, 0, 2)
        # (L, B, C)

        pos = self.pos_embed(x.shape[0])
        pos = pos.permute(1, 0, 2)
        # (L, 1, C)

        style = self.encoder(x, pos=pos, src_key_padding_mask=pad_mask)
        # (L, B, C)

        if self.aggregate_method is not None:
            permute_style = style.permute(1, 0, 2)
            # (B, L, C)
            style_code = self.aggregate_method(permute_style, pad_mask)
            return style_code

        if pad_mask is None:
            style = style.permute(1, 2, 0)
            # (B, C, L)
            style_code = style.mean(2)
            # (B, C)
        else:
            permute_style = style.permute(1, 0, 2)
            # (B, L, C)
            permute_style[pad_mask] = 0
            sum_style_code = permute_style.sum(dim=1)
            # (B, C)
            valid_token_num = (~pad_mask).sum(dim=1).unsqueeze(-1)
            # (B, 1)
            style_code = sum_style_code / valid_token_num
            # (B, C)

        return style_code



class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Sequential(nn.Linear(input_dim, input_dim), Mish(), nn.Linear(input_dim, 1))
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
        N: batch size, T: sequence length, H: Hidden dimension
        input:
            batch_rep : size (N, T, H)
        attention_weight:
            att_w : size (N, T, 1)
        att_mask:
            att_mask: size (N, T): if True, mask this item.
        return:
            utter_rep: size (N, H)
        """

        att_logits = self.W(batch_rep).squeeze(-1)
        # (N, T)
        if att_mask is not None:
            att_mask_logits = att_mask.to(dtype=batch_rep.dtype) * -100000.0
            # (N, T)
            att_logits = att_mask_logits + att_logits

        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Mish.html
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if torch.__version__ >= "1.9":
            return F.mish(input)
        else:
            return mish(input)
        
@torch.jit.script
def mish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return input * torch.tanh(F.softplus(input))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        # q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        # q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(src2, src2, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")



class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src + pos

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, winsize):
        return self.pos_table[:, :winsize].clone().detach()

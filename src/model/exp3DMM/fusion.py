import torch
from torch import nn
import numpy as np


from .dynamic_fc_decoder import DynamicFCDecoderLayer,DynamicFCDecoder
from src.util.model_util import reset_parameters

class Fusion(nn.Module):
    '''
    输入音频特征和视频特征，输出表情3DMM
    '''
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pos_embed_len=80,
        upper_face3d_indices=tuple(list(range(19)) + list(range(46, 51))),
        lower_face3d_indices=tuple(range(19, 46)),
        dynamic_K=None,
        dynamic_ratio=None,
        **_
    ) -> None:
        super().__init__()

        self.upper_face3d_indices = upper_face3d_indices
        self.lower_face3d_indices = lower_face3d_indices

        self.upper_decoder = get_decoder_network(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            num_decoder_layers,
            return_intermediate_dec,
            dynamic_K,
            dynamic_ratio,
        )
        reset_parameters(self.upper_decoder)

        self.lower_decoder = get_decoder_network(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            num_decoder_layers,
            return_intermediate_dec,
            dynamic_K,
            dynamic_ratio,
        )
        reset_parameters(self.lower_decoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        tail_hidden_dim = d_model // 2
        self.upper_tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, len(upper_face3d_indices)),
        )
        self.lower_tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, len(lower_face3d_indices)),
        )


    def forward(self, content, style_code):
        """
        Args:
            content : (B, len, window, audio dim)
            style_code : (B, len, window, video dim)
            其中window设置为11
        Returns:
            face3d: (B, len, 3dmm dim)
        """
        B, N, W, C = content.shape
        style = style_code.permute(2, 0, 1, 3).reshape(W, B * N, C)
        # (W, B*N, C)

        content = content.permute(2, 0, 1, 3).reshape(W, B * N, C)
        # (W, B*N, C)
        tgt = torch.zeros_like(style)
        pos_embed = self.pos_embed(W)
        pos_embed = pos_embed.permute(1, 0, 2)

        upper_face3d_feat = self.upper_decoder(tgt, content, pos=pos_embed, query_pos=style)[0]
        # (W, B*N, C)
        upper_face3d_feat = upper_face3d_feat.permute(1, 0, 2).reshape(B, N, W, C)[:, :, W // 2, :]
        # (B, N, C)
        upper_face3d = self.upper_tail_fc(upper_face3d_feat)
        # (B, N, C_exp)

        lower_face3d_feat = self.lower_decoder(tgt, content, pos=pos_embed, query_pos=style)[0]
        lower_face3d_feat = lower_face3d_feat.permute(1, 0, 2).reshape(B, N, W, C)[:, :, W // 2, :]
        lower_face3d = self.lower_tail_fc(lower_face3d_feat)
        C_exp = len(self.upper_face3d_indices) + len(self.lower_face3d_indices)
        face3d = torch.zeros(B, N, C_exp).to(upper_face3d)
        face3d[:, :, self.upper_face3d_indices] = upper_face3d
        face3d[:, :, self.lower_face3d_indices] = lower_face3d
        return face3d

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

def get_decoder_network(
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    activation,
    normalize_before,
    num_decoder_layers,
    return_intermediate_dec,
    dynamic_K,
    dynamic_ratio
):
    d_style = d_model
    decoder_layer = DynamicFCDecoderLayer(
        d_model,
        nhead,
        d_style,
        dynamic_K,
        dynamic_ratio,
        dim_feedforward,
        dropout,
        activation,
        normalize_before,
    )
    norm = nn.LayerNorm(d_model)
    decoder = DynamicFCDecoder(decoder_layer, num_decoder_layers, norm, return_intermediate_dec)

    return decoder
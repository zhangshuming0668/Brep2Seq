import torch
from torch import nn
import torch.nn.functional as F

from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .module_utils.masking import _make_batch_first
from .module_utils.macro import *

class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.d_model = cfg.d_model
        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src


class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()
        self.n_args = n_args
        self.args_dim = args_dim
        self.command_fcn = nn.Linear(d_model, n_commands)      #[seq_len, batch_size, d_model]->[seq_len, batch_size, n_commands]
        self.args_fcn = nn.Linear(d_model, n_args * args_dim)  #[seq_len, batch_size, d_model]->[seq_len, batch_size, n_args * args_dim]

    def forward(self, out):
        S, N, _ = out.shape

        command_logits = self.command_fcn(out)  # Shape [seq_len, batch_size, n_commands]
        args_logits = self.args_fcn(out)  # Shape [seq_len, batch_size, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [seq_len, batch_size, n_args, args_dim]

        return command_logits, args_logits


class SeqDecoder(nn.Module):
    def __init__(self, cfg):
        super(SeqDecoder, self).__init__()

        self.embedding = ConstEmbedding(cfg, MAX_N_MAIN + MAX_N_SUB)
        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)
        self.fcn_primitive = FCN(cfg.d_model, N_MAIN_COMMANDS + 2, N_ARGS_MAIN, ARGS_DIM)
        self.fcn_feature = FCN(cfg.d_model, N_SUB_COMMANDS + 2, N_ARGS_SUB, ARGS_DIM)


    def forward(self, z, encode_mode=False):         # z [batch_size, 256]
        z = torch.unsqueeze(z, dim=0)
        if encode_mode: return _make_batch_first(z)

        src = self.embedding(z)   # src [seq_len, batch_size, 256]
        y = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None) # out [seq_len, batch_size, 256]
        y_p = y[:MAX_N_MAIN]
        y_f = y[MAX_N_MAIN:]

        commands_primitive, args_primitive = self.fcn_primitive(y_p)
        output_primitive = (commands_primitive, args_primitive)
        output_primitive = _make_batch_first(*output_primitive)

        commands_feature, args_feature = self.fcn_feature(y_f)
        output_feature = (commands_feature, args_feature)
        output_feature = _make_batch_first(*output_feature)

        res = {
            "commands_primitive": output_primitive[0],
            "args_primitive": output_primitive[1],
            "commands_feature": output_feature[0],
            "args_feature": output_feature[1]
        }
        return res
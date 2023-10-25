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


class NonLinearTransform(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.linear0 = nn.Linear(input_dim, 512, bias=False)
        self.bn0 = nn.BatchNorm1d(512)
        self.dp0 = nn.Dropout(p=dropout)
        
        self.linear1_0 = nn.Linear(512, 512, bias=False)
        self.bn1_0 = nn.BatchNorm1d(512)
        self.dp1_0 = nn.Dropout(p=dropout)        
        self.linear1_1 = nn.Linear(512, 512, bias=False)
        self.bn1_1 = nn.BatchNorm1d(512)
        self.dp1_1 = nn.Dropout(p=dropout)
               
        self.linear2 = nn.Linear(512, output_dim)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = self.linear0(inp)           #[seq_len, batch_size, d_model]->[seq_len, batch_size, 512]
        x = x.permute(1, 2, 0)          #[seq_len, batch_size, 512]->[batch_size, 512, seq_len]
        x = F.leaky_relu(self.bn0(x))    
        x = self.dp0(x)                                   
              
        x = x.permute(2, 0, 1)          #[batch_size, 512, seq_len]->[seq_len, batch_size, 512]
        x = self.linear1_0(x)           #[seq_len, batch_size, 512]->[seq_len, batch_size, 512]
        x = x.permute(1, 2, 0)          #[seq_len, batch_size, 512]->[batch_size, 512, seq_len]
        x = F.leaky_relu(self.bn1_0(x))    
        x = self.dp1_0(x)
        
        x = x.permute(2, 0, 1)          #[batch_size, 512, seq_len]->[seq_len, batch_size, 512]
        x = self.linear1_1(x)           #[seq_len, batch_size, 512]->[seq_len, batch_size, 512]
        x = x.permute(1, 2, 0)          #[seq_len, batch_size, 512]->[batch_size, 512, seq_len]
        x = F.leaky_relu(self.bn1_1(x))
        x = self.dp1_1(x)
        
        x = x.permute(2, 0, 1)          #[batch_size, 512, seq_len]->[seq_len, batch_size, 512]
        x = self.linear2(x)             #[seq_len, batch_size, 512]->[seq_len, batch_size, output_dim]
        return x


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
        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, 2*cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)
        self.fcn_main = FCN(cfg.d_model, N_MAIN_COMMANDS + 2, N_ARGS_MAIN, ARGS_DIM)
        self.fcn_sub = FCN(cfg.d_model, N_SUB_COMMANDS + 2, N_ARGS_SUB, ARGS_DIM)

        #test
        # self.embedding_main = ConstEmbedding(cfg, MAX_N_MAIN)
        # self.embedding_sub = ConstEmbedding(cfg, MAX_N_SUB)
        #
        # decoder_layer_main = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        # decoder_norm_main = LayerNorm(cfg.d_model)
        # self.decoder_main = TransformerDecoder(decoder_layer_main, cfg.n_layers_decode, decoder_norm_main)
        #
        # decoder_layer_sub = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        # decoder_norm_sub = LayerNorm(cfg.d_model)
        # self.decoder_sub = TransformerDecoder(decoder_layer_sub, cfg.n_layers_decode, decoder_norm_sub)
        #test

        self.fcn_main = FCN(cfg.d_model, N_MAIN_COMMANDS + 2, N_ARGS_MAIN, ARGS_DIM)
        self.fcn_sub = FCN(cfg.d_model, N_SUB_COMMANDS + 2, N_ARGS_SUB, ARGS_DIM)


    def forward(self, z, encode_mode=False):         # z [1, batch_size, 256]
        if encode_mode: return _make_batch_first(z)

        src = self.embedding(z)   # src [seq_len, batch_size, 256]
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None) # out [seq_len, batch_size, 256]
        out_main = out[:MAX_N_MAIN]
        out_sub = out[MAX_N_MAIN:]

        #test
        # z_main, z_sub = z.chunk(2, dim=-1)
        # src_main = self.embedding_main(z_main)
        # out_main = self.decoder_main(src_main, z_main, tgt_mask=None, tgt_key_padding_mask=None)
        #
        # src_sub = self.embedding_sub(z_sub)
        # out_sub = self.decoder_sub(src_sub, z_sub, tgt_mask=None, tgt_key_padding_mask=None)
        # test

        command_logits_main, args_logits_main = self.fcn_main(out_main)
        out_logits_main = (command_logits_main, args_logits_main)
        out_logits_main = _make_batch_first(*out_logits_main)

        command_logits_sub, args_logits_sub = self.fcn_sub(out_sub)
        out_logits_sub = (command_logits_sub, args_logits_sub)
        out_logits_sub = _make_batch_first(*out_logits_sub)

        res = {
            "command_logits_main": out_logits_main[0],
            "args_logits_main": out_logits_main[1],
            "command_logits_sub": out_logits_sub[0],
            "args_logits_sub": out_logits_sub[1]
        }
        return res
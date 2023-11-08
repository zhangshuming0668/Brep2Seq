# -*- coding: utf-8 -*-
import os
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

from .modules.brep_encoder import BrepEncoder
from .modules.seq_decoder import SeqDecoder
from .modules.loss_func.domain_discriminator import DomainDiscriminator
from .modules.loss_func.dann import DomainAdversarialLoss
from .modules.loss_func.dan import MultipleKernelMaximumMeanDiscrepancy
from .modules.loss_func.kernels import GaussianKernel
from .modules.module_utils.masking import _get_padding_mask, _get_visibility_mask
from .modules.module_utils.macro import *
from .modules.module_utils.vec2json import vec_to_json


class BreptoSeq(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the model.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.primitive_encoder = BrepEncoder(
            num_in_degree=64,  # number of in degree types in the graph
            num_out_degree=64,  # number of out degree types in the graph
            num_spatial=128,  # number of spatial types in the graph
            num_edge_dis=128,  # number of edge dis types in the graph
            edge_type="multi_hop",  # edge type in the graph "multi_hop"
            multi_hop_max_dist=8,  # max distance of multi-hop edges
            num_encoder_layers=args.n_layers_encode,  # num encoder layers
            embedding_dim=args.dim_z,  # encoder embedding dimension
            ffn_embedding_dim=args.dim_feedforward,  # encoder embedding dimension for FFN
            num_attention_heads=args.n_heads,  # num encoder attention heads
            dropout=args.dropout,  # dropout probability
            attention_dropout=args.attention_dropout,  # dropout probability for"attention weights"
            activation_dropout=args.act_dropout,  # dropout probability after"activation in FFN"
            encoder_normalize_before=True,  # apply layernorm before each encoder block
            pre_layernorm=True,
            apply_params_init=True,  # use custom param initialization for Graphormer
            activation_fn="gelu",  # activation function to use
        )

        self.feature_encoder = BrepEncoder(
            num_in_degree=64,  # number of in degree types in the graph
            num_out_degree=64,  # number of out degree types in the graph
            num_spatial=128,  # number of spatial types in the graph
            num_edge_dis=128,  # number of edge dis types in the graph
            edge_type="multi_hop",  # edge type in the graph "multi_hop"
            multi_hop_max_dist=8,  # max distance of multi-hop edges
            num_encoder_layers=args.n_layers_encode,  # num encoder layers
            embedding_dim=args.dim_z,  # encoder embedding dimension
            ffn_embedding_dim=args.dim_feedforward,  # encoder embedding dimension for FFN
            num_attention_heads=args.n_heads,  # num encoder attention heads
            dropout=args.dropout,  # dropout probability
            attention_dropout=args.attention_dropout,  # dropout probability for"attention weights"
            activation_dropout=args.act_dropout,  # dropout probability after"activation in FFN"
            encoder_normalize_before=True,  # apply layernorm before each encoder block
            pre_layernorm=True,
            apply_params_init=True,  # use custom param initialization for Graphormer
            activation_fn="gelu",  # activation function to use
        )

        self.seq_decoder = SeqDecoder(args)

        self.loss_func_weights = 2.0
        self.loss_func_primitive = CADLoss(loss_type="primitive")
        self.loss_func_feature = CADLoss(loss_type="feature")

        if args.similarity_loss == "dann":
            domain_discri = DomainDiscriminator(args.dim_z, hidden_size=1024)
            self.loss_func_similarity = DomainAdversarialLoss(domain_discri)
        else:
            kernels = (GaussianKernel(alpha=0.2), GaussianKernel(alpha=0.35),
                        GaussianKernel(alpha=0.5), GaussianKernel(alpha=0.75),
                        GaussianKernel(alpha=1.0),
                        GaussianKernel(alpha=1.5), GaussianKernel(alpha=2.),
                        GaussianKernel(alpha=3.5), GaussianKernel(alpha=5.)
                       )
            self.loss_func_similarity = MultipleKernelMaximumMeanDiscrepancy(kernels, linear=False)

        self.loss_func_diff_1 = DiffLoss()
        self.loss_func_diff_2 = DiffLoss()
        self.loss_func_diff_3 = DiffLoss()

        self.acc_cmds_primitive = []
        self.acc_args_primitive = []
        self.acc_cmds_feature = []
        self.acc_args_feature = []


    def training_step(self, batch, batch_idx):
        self.primitive_encoder.train()
        self.feature_encoder.train()
        self.seq_decoder.train()

        _, z_p = self.primitive_encoder(batch)
        _, z_f = self.feature_encoder(batch)
        z = z_p + z_f
        output = self.seq_decoder(z)

        # Loss recons
        loss_primitive = self.loss_func_primitive(output, batch["label_commands_primitive"], batch["label_args_primitive"], self.loss_func_weights)
        loss_primitive = sum(loss_primitive.values())
        self.log("loss_primitive", loss_primitive, on_step=True, on_epoch=True)
        loss_feature = self.loss_func_feature(output, batch["label_commands_feature"], batch["label_args_feature"], self.loss_func_weights)
        loss_feature = sum(loss_feature.values())
        self.log("loss_feature", loss_feature, on_step=True, on_epoch=True)

        # Loss similarity & differenec
        z_p_0, z_p_1 = z_p.chunk(2, dim=0)
        z_f_0, z_f_1 = z_f.chunk(2, dim=0)
        self.loss_func_similarity.train()
        loss_similarity = self.loss_func_similarity(z_p_0, z_p_1)
        self.log("loss_similarity", loss_similarity, on_step=True, on_epoch=True)
        loss_difference_1 = self.loss_func_diff_1(z_f_0, z_f_1)
        loss_difference_2 = self.loss_func_diff_2(z_p_0, z_f_0)
        loss_difference_3 = self.loss_func_diff_3(z_p_1, z_f_1)
        loss_difference = loss_difference_1 + loss_difference_2 + loss_difference_3
        self.log("loss_difference", loss_difference, on_step=True, on_epoch=True)

        loss = loss_primitive + loss_feature + 0.25 * loss_similarity + 0.25 * loss_difference
        return loss


    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.primitive_encoder.eval()
        self.feature_encoder.eval()
        self.seq_decoder.eval()

        _, z_p = self.primitive_encoder(batch)
        _, z_f = self.feature_encoder(batch)
        z = z_p + z_f
        output = self.seq_decoder(z)

        # Acc
        self.cal_acc_primitive(output, batch["label_commands_primitive"], batch["label_args_primitive"])
        self.cal_acc_feature(output, batch["label_commands_feature"], batch["label_args_feature"])

        # Loss recons
        loss_primitive = self.loss_func_primitive(output, batch["label_commands_primitive"], batch["label_args_primitive"], self.loss_func_weights)
        loss_primitive = sum(loss_primitive.values())
        loss_feature = self.loss_func_feature(output, batch["label_commands_feature"], batch["label_args_feature"], self.loss_func_weights)
        loss_feature = sum(loss_feature.values())
        loss = loss_primitive + loss_feature
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss


    def validation_epoch_end(self, val_step_outputs):
        self.logger.experiment.add_scalars('acc_primitive',
                                           {"opr": np.mean(self.acc_cmds_primitive),
                                            "param": np.mean(self.acc_args_primitive)},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('acc_feature',
                                           {"opr": np.mean(self.acc_cmds_feature),
                                            "param": np.mean(self.acc_args_feature)},
                                           global_step=self.current_epoch)
        self.acc_cmds_primitive = []
        self.acc_args_primitive = []
        self.acc_cmds_feature = []
        self.acc_args_feature = []


    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.primitive_encoder.eval()
        self.feature_encoder.eval()
        self.seq_decoder.eval()

        _, z_p = self.primitive_encoder(batch)
        _, z_f = self.feature_encoder(batch)
        z = z_p + z_f
        output = self.seq_decoder(z)

        # output predicted result-------------------------------------------------------------------
        commands_primitive = torch.argmax(torch.softmax(output['commands_primitive'], dim=-1), dim=-1)  # (N, S)
        args_primitive = torch.argmax(torch.softmax(output['args_primitive'], dim=-1), dim=-1) - 1  # (N, S, n_args)
        commands_primitive = commands_primitive.long().detach().cpu().numpy()  # (N, S)
        args_primitive = args_primitive.long().detach().cpu().numpy()  # (N, S, n_args)

        commands_feature = torch.argmax(torch.softmax(output['commands_feature'], dim=-1), dim=-1)  # (N, S)
        args_feature = torch.argmax(torch.softmax(output['args_feature'], dim=-1), dim=-1) - 1  # (N, S, n_args)
        commands_feature = commands_feature.long().detach().cpu().numpy()  # (N, S)
        args_feature = args_feature.long().detach().cpu().numpy()  # (N, S, n_args)

        # 将结果转为json文件--------------------------------------------------------------------------
        batch_size = np.size(commands_primitive, 0)
        for i in range(batch_size):
            end_pos = MAX_N_MAIN - np.sum((commands_primitive[i][:] == EOS_IDX).astype(np.int))
            primitive_type = commands_primitive[i][:end_pos + 1]  # (Seq)
            primitive_param = args_primitive[i][:end_pos + 1][:]  # (Seq, n_args)

            end_pos = MAX_N_SUB - np.sum((commands_feature[i][:] == EOS_IDX).astype(np.int))
            feature_type = commands_feature[i][:end_pos + 1]  # (Seq)
            feature_param = args_feature[i][:end_pos + 1][:]  # (Seq, n_args)

            file_name = "rebuild_{}.json".format(str(i))
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/predicted_seq")
            if not os.path.exists(file_path): os.makedirs(file_path)
            vec_to_json(primitive_type, primitive_param, feature_type, feature_param, os.path.join(file_path,file_name))


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                               threshold=0.0001, threshold_mode='rel',
                                                               min_lr=0.000001, cooldown=2, verbose=False)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"}
                }


    def optimizer_step(self,
                        epoch,
                        batch_idx,
                        optimizer,
                        optimizer_idx,
                        optimizer_closure,
                        on_tpu,
                        using_native_amp,
                        using_lbfgs,
                        ):
        # update params
        optimizer.step(closure=optimizer_closure)
        # manually warm up lr without a scheduler
        if self.trainer.global_step < 10000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 10000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 0.0001

    def cal_acc_primitive(self, output, label_commands, label_args):
        out_commands = torch.argmax(torch.softmax(output['commands_primitive'], dim=-1), dim=-1)
        out_commands = out_commands.long().detach().cpu().numpy()  # (N, S)
        out_args = torch.argmax(torch.softmax(output['args_primitive'], dim=-1), dim=-1) - 1
        out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)
        gt_commands = label_commands.squeeze(1).long().detach().cpu().numpy()  # (N, S)
        gt_args = label_args.squeeze(1).long().detach().cpu().numpy()  # (N, S, n_args)

        # acc_opr
        commands_comp = (out_commands == gt_commands).astype(np.int)
        self.acc_cmds_primitive.append(np.mean(commands_comp))

        # acc_param
        acc_pos = np.where(out_commands == gt_commands)
        args_comp = (np.abs(out_args - gt_args) < 5).astype(np.int)
        self.acc_args_primitive.append(np.mean(args_comp[acc_pos]))


    def cal_acc_feature(self, output, label_commands, label_args):
        out_commands = torch.argmax(torch.softmax(output['commands_feature'], dim=-1), dim=-1)
        out_commands = out_commands.long().detach().cpu().numpy()  # (N, S)
        out_args = torch.argmax(torch.softmax(output['args_feature'], dim=-1), dim=-1) - 1
        out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)
        gt_commands = label_commands.squeeze(1).long().detach().cpu().numpy()  # (N, S)
        gt_args = label_args.squeeze(1).long().detach().cpu().numpy()  # (N, S, n_args)

        # acc opr
        commands_comp = (out_commands == gt_commands).astype(np.int)
        self.acc_cmds_feature.append(np.mean(commands_comp))

        # acc param
        acc_pos = np.where(out_commands == gt_commands)
        args_comp = (np.abs(out_args - gt_args) < 5).astype(np.int)
        self.acc_args_feature.append(np.mean(args_comp[acc_pos]))
        

class CADLoss(nn.Module):
    def __init__(self, loss_type="primitive"):
        super().__init__()
        assert loss_type in ("primitive", "feature")
        self.loss_type = loss_type
        if loss_type == "primitive":
            self.n_commands = N_MAIN_COMMANDS + 2
            self.args_dim = ARGS_DIM
            self.register_buffer("cmd_args_mask", torch.tensor(MAIN_CMD_ARGS_MASK))
        else:
            self.n_commands = N_SUB_COMMANDS + 2
            self.args_dim = ARGS_DIM
            self.register_buffer("cmd_args_mask", torch.tensor(SUB_CMD_ARGS_MASK))


    def forward(self, output, label_commands, label_args, args_weight):
        # Target & predictions
        tgt_commands, tgt_args = label_commands, label_args
        if self.loss_type == "primitive":
            command_logits, args_logits = output["commands_primitive"], output["args_primitive"]
        else:
            command_logits, args_logits = output["commands_feature"], output["args_feature"]

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask_ = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        mask = self.cmd_args_mask[tgt_commands.long()]

        loss_cmd = F.cross_entropy(command_logits[padding_mask_.bool()].reshape(-1, self.n_commands),
                                   tgt_commands[padding_mask_.bool()].reshape(-1).long())

        loss_args = F.cross_entropy(args_logits[mask.bool()].reshape(-1, self.args_dim),
                                    tgt_args[mask.bool()].reshape(-1).long() + 1)

        loss_cmd = loss_cmd
        loss_args = args_weight * loss_args

        res = {"loss_cmds": loss_cmd, "loss_args": loss_args}
        return res


class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()


    def forward(self, input1, input2):
        input1_ = torch.mean(input1, dim=0)
        input1 = input1 - input1_
        input1 = torch.nn.functional.normalize(input1, p=2, dim=1)

        input2_ = torch.mean(input2, dim=0)
        input2 = input2 - input2_
        input2 = torch.nn.functional.normalize(input2, p=2, dim=1)

        input1 = torch.transpose(input1, 0, 1)
        correlation_matrix = torch.matmul(input1, input2)

        diff_loss = torch.mean(torch.square(correlation_matrix)) * 1.0
        if diff_loss < 0.0:
            diff_loss = 0.0
        return diff_loss

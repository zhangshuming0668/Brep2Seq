# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from geomloss import SamplesLoss

from .modules.brep_encoder import BrepEncoder
from .modules.seq_decoder import SeqDecoder
from .modules.module_utils.masking import _get_padding_mask, _get_visibility_mask
from .modules.module_utils.macro import *
from .modules.domain_adv.domain_discriminator import DomainDiscriminator
from .modules.domain_adv.dann import DomainAdversarialLoss
from .modules.domain_adv.dan import MultipleKernelMaximumMeanDiscrepancy
from .modules.domain_adv.kernels import GaussianKernel
from .modules.module_utils.vec2json import vec_to_json, output_z


class BreptoSeq(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the model.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.main_encoder = BrepEncoder(
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
            # apply layernorm before self-attention and ffn. Without this, post layernorm will used
            apply_params_init=True,  # use custom param initialization for Graphormer
            activation_fn="gelu",  # activation function to use
        )

        self.sub_encoder = BrepEncoder(
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
            # apply layernorm before self-attention and ffn. Without this, post layernorm will used
            apply_params_init=True,  # use custom param initialization for Graphormer
            activation_fn="gelu",  # activation function to use
        )

        self.decoder = SeqDecoder(args)

        self.loss_func_weights = 5.0
        self.loss_func_main = CADLoss(loss_type="mainfeat")
        self.loss_func_sub = CADLoss(loss_type="subfeat")

        # DANN方法计算loss_similarity
        self.domain_discri = DomainDiscriminator(args.dim_z, hidden_size=512)
        self.domain_adv = DomainAdversarialLoss(self.domain_discri)

        # MMD方法计算loss_similarity
        self.kernels = (GaussianKernel(alpha=0.2), GaussianKernel(alpha=0.35),
                        GaussianKernel(alpha=0.5), GaussianKernel(alpha=0.75),
                        GaussianKernel(alpha=1.0),
                        GaussianKernel(alpha=1.5), GaussianKernel(alpha=2.),
                        GaussianKernel(alpha=3.5), GaussianKernel(alpha=5.)
                        )
        self.mmd = MultipleKernelMaximumMeanDiscrepancy(self.kernels, linear=False)

        self.distance = SamplesLoss("sinkhorn", blur=0.05)

        self.loss_func_diff_1 = DiffLoss()
        self.loss_func_diff_2 = DiffLoss()
        self.loss_func_diff_3 = DiffLoss()

        self.loss_cmd_main = []
        self.loss_args_main = []
        self.loss_cmd_sub = []
        self.loss_args_sub = []
        self.val_loss_cmd_main = []
        self.val_loss_args_main = []
        self.val_loss_cmd_sub = []
        self.val_loss_args_sub = []
        
        #------------------------------
        self.commands_acc = []
        self.seqs_acc = []
        #------------------------------
        
        #------------------------------
        self.sub_commands_acc = []
        self.sub_seqs_acc = []
        #------------------------------

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.main_encoder.train()
        self.sub_encoder.train()
        self.decoder.train()
        self.domain_discri.train()
        self.domain_adv.train()

        _, z_main = self.main_encoder(batch)  #[batch_size, 256]
        _, z_sub = self.sub_encoder(batch)    #[batch_size, 256]

        z = torch.cat([z_main, z_sub], dim=-1)
        z = torch.unsqueeze(z, dim=0)         #[1, batch_size, 512]
        logits = self.decoder(z)

        loss_main = self.loss_func_main(logits, batch["label_commands_main"], batch["label_args_main"], self.loss_func_weights)
        # ----------------------------------------------------------------------
        loss_cmd_copy = loss_main["loss_cmd"].detach().cpu().numpy()
        loss_args_copy = loss_main["loss_args"].detach().cpu().numpy()
        self.loss_cmd_main.append(loss_cmd_copy)
        self.loss_args_main.append(loss_args_copy)
        # ----------------------------------------------------------------------
        loss_main = sum(loss_main.values())

        loss_sub = self.loss_func_sub(logits, batch["label_commands_sub"], batch["label_args_sub"], self.loss_func_weights)
        # ----------------------------------------------------------------------
        loss_cmd_copy_ = loss_sub["loss_cmd"].detach().cpu().numpy()
        loss_args_copy_ = loss_sub["loss_args"].detach().cpu().numpy()
        self.loss_cmd_sub.append(loss_cmd_copy_)
        self.loss_args_sub.append(loss_args_copy_)
        # ----------------------------------------------------------------------
        loss_sub = sum(loss_sub.values())

        # 拆分中间特征-----------------------------------------------------------
        z_main_b, z_main_a = z_main.chunk(2, dim=0) # a--only principal primitive; b--with detail feature
        z_sub_b, z_sub_a = z_sub.chunk(2, dim=0)

        # loss_similarity-------------------------------------------------------
        # DANN方法
        loss_similarity = self.domain_adv(z_main_a, z_main_b)
        domain_acc = self.domain_adv.domain_discriminator_accuracy
        self.log("loss_similarity", loss_similarity, on_step=True, on_epoch=True)
        self.log("acc_transfer", domain_acc, on_step=True, on_epoch=True)
        # MMD方法
        # self.mmd.train()
        # loss_similarity = self.mmd(z_main_complete, z_main_mian)
        # loss_similarity = loss_similarity
        # self.log("loss_similarity", loss_similarity, on_step=False, on_epoch=True)
        # loss_similarity-------------------------------------------------------
        Wasserstein_D = self.distance(z_main_a, z_main_b)
        self.log("Wasserstein_D", Wasserstein_D, on_step=True, on_epoch=True)

        loss_difference_1 = self.loss_func_diff_1(z_sub_a, z_sub_b)
        loss_difference_2 = self.loss_func_diff_2(z_main_a, z_sub_a)
        loss_difference_3 = self.loss_func_diff_3(z_main_b, z_sub_b)

        loss_difference = loss_difference_1 + loss_difference_2 + loss_difference_3
        # loss_difference = loss_difference_2 + loss_difference_3
        self.log("loss_difference", loss_difference, on_step=True, on_epoch=True)

        loss = loss_sub + loss_main + 0.25 * loss_similarity + 0.5 * loss_difference

        # cross-reconstruction
        z_main_1, z_main_2 = z_main.chunk(2, dim=0)
        z_main_cross = torch.cat([z_main_2, z_main_1], dim=0)
        z_cross = torch.cat([z_main_cross, z_sub], dim=-1)
        z_cross = torch.unsqueeze(z_cross, dim=0)  # [1, batch_size, 512]
        logits_cross = self.decoder(z_cross)
        loss_main_cross = self.loss_func_main(logits_cross, batch["label_commands_main"], batch["label_args_main"], self.loss_func_weights)
        loss_sub_cross = self.loss_func_sub(logits_cross, batch["label_commands_sub"], batch["label_args_sub"], self.loss_func_weights)
        loss_cross = sum(loss_main_cross.values()) + sum(loss_sub_cross.values())
        loss = loss + loss_cross
        # cross-reconstruction

        return loss


    def training_epoch_end(self, training_step_outputs):
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=False, on_epoch=True)

        loss_cmd_main = np.mean(self.loss_cmd_main)
        loss_args_main = np.mean(self.loss_args_main) / self.loss_func_weights
        loss_cmd_sub = np.mean(self.loss_cmd_sub)
        loss_args_sub = np.mean(self.loss_args_sub) / self.loss_func_weights
        self.logger.experiment.add_scalars('loss_train',
                                           {"cmd_main": loss_cmd_main,
                                            "args_main": loss_args_main,
                                            "cmd_sub": loss_cmd_sub,
                                            "args_sub": loss_args_sub},
                                           global_step=self.current_epoch)
        self.logger.experiment.add_scalars('loss',
                                           {"train": loss_cmd_main + loss_args_main + loss_cmd_sub + loss_args_sub},
                                           global_step=self.current_epoch)
        self.loss_cmd_main = []
        self.loss_args_main = []
        self.loss_cmd_sub = []
        self.loss_args_sub = []


    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.main_encoder.eval()
        self.sub_encoder.eval()
        self.decoder.eval()
        self.domain_discri.eval()
        self.domain_adv.eval()

        _, z_main = self.main_encoder(batch)
        _, z_sub = self.sub_encoder(batch)

        z = torch.cat([z_main, z_sub], dim=-1)
        z = torch.unsqueeze(z, dim=0)
        logits = self.decoder(z)

        loss_main = self.loss_func_main(logits, batch["label_commands_main"], batch["label_args_main"], self.loss_func_weights)
        # ----------------------------------------------------------------------
        loss_cmd_copy = loss_main["loss_cmd"].detach().cpu().numpy()
        loss_args_copy = loss_main["loss_args"].detach().cpu().numpy()
        self.val_loss_cmd_main.append(loss_cmd_copy)
        self.val_loss_args_main.append(loss_args_copy)
        # ----------------------------------------------------------------------
        loss_main = sum(loss_main.values())

        loss_sub = self.loss_func_sub(logits, batch["label_commands_sub"], batch["label_args_sub"], self.loss_func_weights)
        # ----------------------------------------------------------------------
        loss_cmd_copy_ = loss_sub["loss_cmd"].detach().cpu().numpy()
        loss_args_copy_ = loss_sub["loss_args"].detach().cpu().numpy()
        self.val_loss_cmd_sub.append(loss_cmd_copy_)
        self.val_loss_args_sub.append(loss_args_copy_)
        # ----------------------------------------------------------------------
        loss_sub = sum(loss_sub.values())

        loss = loss_main + loss_sub
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.cal_acc_main(logits, batch["label_commands_main"], batch["label_args_main"])
        self.cal_acc_sub(logits, batch["label_commands_sub"], batch["label_args_sub"])

        return loss


    def validation_epoch_end(self, val_step_outputs):
        loss_cmd_main = np.mean(self.val_loss_cmd_main)
        loss_args_main = np.mean(self.val_loss_args_main) / self.loss_func_weights
        loss_cmd_sub = np.mean(self.val_loss_cmd_sub)
        loss_args_sub = np.mean(self.val_loss_args_sub) / self.loss_func_weights
        self.logger.experiment.add_scalars('loss_val',
                                           {"cmd_main": loss_cmd_main,
                                            "args_main": loss_args_main,
                                            "cmd_sub": loss_cmd_sub,
                                            "args_sub": loss_args_sub},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('loss',
                                           {"val": loss_cmd_main + loss_args_main + loss_cmd_sub + loss_args_sub},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('val_acc_cmd',
                                           {"commands": np.mean(self.commands_acc),
                                            "seqs": np.mean(self.seqs_acc)},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('val_acc_cmd_feature',
                                           {"commands": np.mean(self.sub_commands_acc),
                                            "seqs": np.mean(self.sub_seqs_acc)},
                                           global_step=self.current_epoch)

        self.val_loss_cmd_main = []
        self.val_loss_args_main = []
        self.val_loss_cmd_sub = []
        self.val_loss_args_sub = []

        #---------------------------------
        self.commands_acc = []
        self.seqs_acc = []
        #---------------------------------
        
        #---------------------------------
        self.sub_commands_acc = []
        self.sub_seqs_acc = []
        #---------------------------------


    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.main_encoder.eval()
        self.sub_encoder.eval()
        self.decoder.eval()
        self.domain_discri.eval()
        self.domain_adv.eval()

        _, z_main = self.main_encoder(batch)
        _, z_sub = self.sub_encoder(batch)

        z = torch.cat([z_main, z_sub], dim=-1)
        z = torch.unsqueeze(z, dim=0)
        logits = self.decoder(z)

        # 结果处理------------------------------------------------------------------------------------
        out_main_commands = torch.argmax(torch.softmax(logits['command_logits_main'], dim=-1), dim=-1)  # (N, S)
        out_main_args = torch.argmax(torch.softmax(logits['args_logits_main'], dim=-1), dim=-1) - 1  # (N, S, n_args)
        out_main_commands = out_main_commands.long().detach().cpu().numpy()  # (N, S)
        out_main_args = out_main_args.long().detach().cpu().numpy()  # (N, S, n_args)

        out_sub_commands = torch.argmax(torch.softmax(logits['command_logits_sub'], dim=-1), dim=-1)  # (N, S)
        out_sub_args = torch.argmax(torch.softmax(logits['args_logits_sub'], dim=-1), dim=-1) - 1  # (N, S, n_args)
        out_sub_commands = out_sub_commands.long().detach().cpu().numpy()  # (N, S)
        out_sub_args = out_sub_args.long().detach().cpu().numpy()  # (N, S, n_args)

        # 将结果转为json文件--------------------------------------------------------------------------
        batch_size = np.size(out_main_commands, 0)
        for i in range(batch_size):
            # 计算每个commands的实际长度
            end_index = MAX_N_MAIN - np.sum((out_main_commands[i][:] == EOS_IDX).astype(np.int))
            # masked出实际commands
            main_commands = out_main_commands[i][:end_index + 1]  # (Seq)
            # masked出实际args
            main_args = out_main_args[i][:end_index + 1][:]  # (Seq, n_args)

            end_index = MAX_N_SUB - np.sum((out_sub_commands[i][:] == EOS_IDX).astype(np.int))
            # masked出实际commands
            sub_commands = out_sub_commands[i][:end_index + 1]  # (Seq)
            # masked出实际args
            sub_args = out_sub_args[i][:end_index + 1][:]  # (Seq, n_args)

            # file_name = str(batch["id"][i].long().detach().cpu().numpy()).zfill(8)
            file_name = "rebuild_" + str(batch["id"][i].long().detach().cpu().numpy())

            vec_to_json(main_commands, main_args, sub_commands, sub_args, file_name)

            # 输出隐向量分布情况
            # z_main = logits_main["tgt_z"].detach().cpu().numpy()
            # z_sub = logits_sub["tgt_z"].detach().cpu().numpy()
            # output_z(z_main, z_sub)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.0015)  #batch_size=128 lr=0.0015
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.0015)

        # 学习策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,  # 10
                                                               threshold=0.0001, threshold_mode='rel',
                                                               min_lr=0.000001, cooldown=1, verbose=False)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"}
                }

    # 逐渐增大学习率
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
                pg["lr"] = lr_scale * 0.001

    def cal_acc_main(self, logits_main, label_commands, label_args):
        out_commands = torch.argmax(torch.softmax(logits_main['command_logits_main'], dim=-1), dim=-1)
        out_commands = out_commands.long().detach().cpu().numpy()  # (N, S)
        out_args = torch.argmax(torch.softmax(logits_main['args_logits_main'], dim=-1), dim=-1) - 1
        out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)
        gt_commands = label_commands.squeeze(1).long().detach().cpu().numpy()  # (N, S)
        gt_args = label_args.squeeze(1).long().detach().cpu().numpy()  # (N, S, n_args)

        # 计算整个commands的准确率
        same_commands = 0  # commands完全相同的个数
        for i in range(np.size(out_commands, 0)):
            end_comp = (gt_commands[i][:] == EOS_IDX).astype(np.int)
            end_index = MAX_N_MAIN - np.sum(end_comp)  # 计算每个commands的实际长度
            com_comp = (out_commands[i][:end_index + 1] == gt_commands[i][:end_index + 1]).astype(np.int)
            if (np.sum(com_comp) == end_index + 1):
                same_commands = same_commands + 1
        self.seqs_acc.append(same_commands / np.size(out_commands, 0))

        commands_comp = (out_commands == gt_commands).astype(np.int)
        self.commands_acc.append(np.mean(commands_comp))

    def cal_acc_sub(self, logits_sub, label_commands, label_args):
        out_commands = torch.argmax(torch.softmax(logits_sub['command_logits_sub'], dim=-1), dim=-1)
        out_commands = out_commands.long().detach().cpu().numpy()  # (N, S)
        out_args = torch.argmax(torch.softmax(logits_sub['args_logits_sub'], dim=-1), dim=-1) - 1
        out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)
        gt_commands = label_commands.squeeze(1).long().detach().cpu().numpy()  # (N, S)
        gt_args = label_args.squeeze(1).long().detach().cpu().numpy()  # (N, S, n_args)

        # 计算整个commands的准确率
        same_commands = 0  # commands完全相同的个数
        for i in range(np.size(out_commands, 0)):
            end_comp = (gt_commands[i][:] == EOS_IDX).astype(np.int)
            end_index = MAX_N_SUB - np.sum(end_comp)  # 计算每个commands的实际长度
            com_comp = (out_commands[i][:end_index + 1] == gt_commands[i][:end_index + 1]).astype(np.int)
            if (np.sum(com_comp) == end_index + 1):
                same_commands = same_commands + 1
        self.sub_seqs_acc.append(same_commands / np.size(out_commands, 0))

        commands_comp = (out_commands == gt_commands).astype(np.int)
        self.sub_commands_acc.append(np.mean(commands_comp))
        

class CADLoss(nn.Module):
    def __init__(self, loss_type="mainfeat"):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "mainfeat":
            self.n_commands = N_MAIN_COMMANDS + 2
            self.args_dim = ARGS_DIM
            self.register_buffer("cmd_args_mask", torch.tensor(MAIN_CMD_ARGS_MASK))
        elif loss_type == "subfeat":
            self.n_commands = N_SUB_COMMANDS + 2
            self.args_dim = ARGS_DIM
            self.register_buffer("cmd_args_mask", torch.tensor(SUB_CMD_ARGS_MASK))

    def forward(self, output, label_commands, label_args, args_weight):
        # Target & predictions
        tgt_commands, tgt_args = label_commands, label_args
        if self.loss_type == "mainfeat":
            command_logits, args_logits = output["command_logits_main"], output["args_logits_main"]
        else:
            command_logits, args_logits = output["command_logits_sub"], output["args_logits_sub"]

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask_ = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        mask = self.cmd_args_mask[tgt_commands.long()]  # mask与标签commands所对应的参数矩阵对应，相应used参数位置为1，其余为0

        loss_cmd = F.cross_entropy(command_logits[padding_mask_.bool()].reshape(-1, self.n_commands),
                                   tgt_commands[padding_mask_.bool()].reshape(-1).long())

        loss_args = F.cross_entropy(args_logits[mask.bool()].reshape(-1, self.args_dim),
                                    tgt_args[mask.bool()].reshape(-1).long() + 1)  # 这里+1是因为标签used参数没有-1
        # args_logits和tgt_args都mask，mask有两个作用，一个是对齐计算结果和标签，一个是剔除无用数据，提取其中的非-1数据，

        loss_cmd = loss_cmd
        loss_args = args_weight * loss_args

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        # batch_size = input1.size(0)
        # input1 = input1.view(batch_size, -1)
        # input2 = input2.view(batch_size, -1)

        # input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        # input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        # input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        # input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        # diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

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

# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

from modules.latentGAN import Generator, Discriminator
from brep2seq import BreptoSeq
from modules.module_utils.macro import *
from modules.module_utils.vec2json import vec_to_json

class GAN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = args.batch_size
        self.dim_z = args.dim_z
        self.dim_n = args.dim_n
        self.gp_lambda = 10

        brep_seq_model = BreptoSeq.load_from_checkpoint(args.checkpoint)
        self.main_encoderdecoder = brep_seq_model.main_encoderdecoder
        self.sub_encoderdecoder = brep_seq_model.sub_encoderdecoder

        self.generator = Generator(args.dim_n, args.dim_h, args.dim_z)
        self.discriminator = Discriminator(args.dim_h, args.dim_z*2)
        self.adversarial_loss = F.binary_cross_entropy

    def calc_gradient_penalty(self, netD, real_z, fake_z):
        alpha = torch.rand([self.batch_size, 1], device=real_z.device)
        alpha = alpha.expand(real_z.size())

        interpolates = alpha * real_z.detach() + ((1 - alpha) * fake_z.detach())
        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size(), device=real_z.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda # LAMBDA
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.main_encoderdecoder.eval()
        self.sub_encoderdecoder.eval()

        # train discriminator
        if optimizer_idx == 0:
            # A ---------------------------------------------------------
            self.generator.eval()
            self.discriminator.train()
            #---------------------------------------------------------
            # generate real z
            logits_sub = self.sub_encoderdecoder(batch)
            z_sub_real = logits_sub["tgt_z"]
            logits_main = self.main_encoderdecoder(batch)
            z_main_real = logits_main["tgt_z"]
            z_real = torch.cat([z_main_real, z_sub_real], dim=1).detach()
            # how well can it label as real?
            mone = -1 * torch.ones([1], device=batch["node_data"].device)
            logits_real = self.discriminator(z_real)
            logits_real = logits_real.mean(dim=0, keepdim=True)
            real_loss = logits_real * mone

            #---------------------------------------------------------
            # generate fake z
            one = torch.ones([1], device=batch["node_data"].device)
            # n = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            # generated_z = self.generator(n).detach()
            n_1 = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            n_2 = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            generated_z = self.generator(n_1, n_2).detach()

            logits_fake = self.discriminator(generated_z)
            logits_fake = logits_fake.mean(dim=0, keepdim=True)
            fake_loss = logits_fake * one
            #---------------------------------------------------------
            gradient_penalty = self.calc_gradient_penalty(self.discriminator, z_real, generated_z)

            Wasserstein_D = logits_real - logits_fake
            self.log("Wasserstein_D", Wasserstein_D, prog_bar=False)

            d_loss = fake_loss + real_loss + gradient_penalty
            self.log("D_loss", d_loss, prog_bar=False)
            self.log("D_loss_1", fake_loss, prog_bar=False)
            self.log("D_loss_2", real_loss, prog_bar=False)
            return d_loss

            # B ---------------------------------------------------------
            # self.generator.eval()
            # self.discriminator.train()
            # logits_sub = self.sub_encoderdecoder(batch)
            # z_sub_real = logits_sub["tgt_z"]
            # logits_main = self.main_encoderdecoder(batch)
            # z_main_real = logits_main["tgt_z"]
            # z_real = torch.cat([z_sub_real, z_main_real], dim=1).detach()
            # valid = torch.ones(self.batch_size, device=batch["node_data"].device)
            # real_loss = self.adversarial_loss(self.discriminator(z_real), valid)
            #
            # n = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            # generated_z = self.generator(n).detach()
            # fake = torch.zeros(self.batch_size, device=batch["node_data"].device)
            # fake_loss = self.adversarial_loss(self.discriminator(generated_z), fake)
            #
            # gradient_penalty = self.calc_gradient_penalty(self.discriminator, z_real, generated_z)
            #
            # d_loss = real_loss + fake_loss + gradient_penalty
            # self.log("d_loss_real", real_loss, prog_bar=True)
            # self.log("d_loss_fake", fake_loss, prog_bar=True)
            # return d_loss

        # train generator
        if optimizer_idx == 1:
            # A ---------------------------------------------------------
            self.generator.train()
            self.discriminator.eval()
            # generate fake z
            # n = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            # generated_z = self.generator(n)
            n_1 = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            n_2 = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            generated_z = self.generator(n_1, n_2)

            # ground truth result (ie: all fake)
            one = torch.ones([1], device=batch["node_data"].device)
            mone = -1 * one
            logits_fake = self.discriminator(generated_z)
            logits_fake = logits_fake.mean(dim=0, keepdim=True)
            g_loss = logits_fake * mone
            self.log("G_loss", g_loss, prog_bar=False)
            return g_loss

            # B ---------------------------------------------------------
            # n = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            # generated_z = self.generator(n)
            # valid = torch.ones(self.batch_size, device=batch["node_data"].device)
            # g_loss = self.adversarial_loss(self.discriminator(generated_z), valid)
            # self.log("g_loss_fake", g_loss, prog_bar=True)
            # return g_loss


    def configure_optimizers(self):
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.00005, betas=(0.5, 0.9))  #default 0.5 0.9
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.00005, betas=(0.5, 0.9))
        return (
            {'optimizer': opt_d, 'frequency': 2},
            {'optimizer': opt_g, 'frequency': 1}
        )

    def test_step(self, batch, batch_idx):
        self.main_encoderdecoder.eval()
        self.sub_encoderdecoder.eval()
        self.generator.eval()

        # 1. sample noise
        # n_1 = torch.randn(batch["node_data"].size()[0] self.dim_n, device=batch["node_data"].device)
        # n_2 = torch.randn(batch["node_data"].size()[0], self.dim_n, device=batch["node_data"].device)
        # generated_z = self.generator(n_1, n_2)

        # 2. main feature + random sample sub feature
        logits_main = self.main_encoderdecoder(batch)
        logits_sub = self.sub_encoderdecoder(batch)
        z_real = torch.cat([logits_main["tgt_z"], logits_sub["tgt_z"]], dim=1)
        z_main_real = logits_main["tgt_z"]
        for i in range(1, z_main_real.size()[0]):
            z_main_real[i] = z_main_real[0]
        n_1 = torch.randn(z_main_real.size()[0], self.dim_n, device=batch["node_data"].device)
        n_2 = torch.randn(z_main_real.size()[0], self.dim_n, device=batch["node_data"].device)
        generated_z = self.generator(n_1, n_2, z_main_real, True)
        generated_z[0] = z_real[0]

        # 3. Interpolation
        # logits_main = self.main_encoderdecoder(batch)
        # logits_sub = self.sub_encoderdecoder(batch)
        # generated_z = torch.cat([logits_main["tgt_z"], logits_sub["tgt_z"]], dim=1)
        # for i in range(generated_z.size()[0]-2):
        #     generated_z[i+1] = (1.0 - 0.1*(i+1))*generated_z[0] + 0.1*(i+1)*generated_z[-1]

        z_main, z_sub = generated_z.chunk(2, dim=1)
        z_main = torch.unsqueeze(z_main, dim=0)
        z_sub = torch.unsqueeze(z_sub, dim=0)

        logits_main = self.main_encoderdecoder.decoder(z_main)
        logits_sub = self.sub_encoderdecoder.decoder(z_sub)

        #结果处理------------------------------------------------------------------------------------
        out_main_commands = torch.argmax(torch.softmax(logits_main['command_logits'], dim=-1), dim=-1) # (N, S)     
        out_main_args = torch.argmax(torch.softmax(logits_main['args_logits'], dim=-1), dim=-1) - 1    # (N, S, n_args)  
        out_main_commands = out_main_commands.long().detach().cpu().numpy()  # (N, S)
        out_main_args = out_main_args.long().detach().cpu().numpy()  # (N, S, n_args)
        
        out_sub_commands = torch.argmax(torch.softmax(logits_sub['command_logits'], dim=-1), dim=-1) # (N, S)     
        out_sub_args = torch.argmax(torch.softmax(logits_sub['args_logits'], dim=-1), dim=-1) - 1    # (N, S, n_args)  
        out_sub_commands = out_sub_commands.long().detach().cpu().numpy()  # (N, S)
        out_sub_args = out_sub_args.long().detach().cpu().numpy()  # (N, S, n_args)
             
        #将结果转为json文件--------------------------------------------------------------------------
        batch_size = np.size(out_main_commands, 0)
        for i in range(batch_size):
            #计算每个commands的实际长度
            end_index = MAX_N_MAIN - np.sum((out_main_commands[i][:] == EOS_IDX).astype(np.int))        
            #masked出实际commands
            main_commands = out_main_commands[i][:end_index+1] # (Seq)            
            #masked出实际args
            main_args = out_main_args[i][:end_index+1][:] # (Seq, n_args)    
            
            end_index = MAX_N_SUB - np.sum((out_sub_commands[i][:] == EOS_IDX).astype(np.int))        
            #masked出实际commands
            sub_commands = out_sub_commands[i][:end_index+1] # (Seq)            
            #masked出实际args
            sub_args = out_sub_args[i][:end_index+1][:] # (Seq, n_args)   
            
            file_index = batch_idx * batch_size + i
            vec_to_json(main_commands, main_args, sub_commands, sub_args, file_index)
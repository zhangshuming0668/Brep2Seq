# -*- coding: utf-8 -*-
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.autograd as autograd

from .brep2seq import BreptoSeq
from .modules.latentGAN import Generator, Discriminator
from .modules.module_utils.macro import *
from .modules.module_utils.vec2json import vec_to_json

class GAN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = args.batch_size
        self.dim_z = args.dim_z
        self.dim_n = args.dim_n
        self.gp_lambda = 10

        brep_seq_model = BreptoSeq.load_from_checkpoint(args.pretrained)
        self.primitive_encoder = brep_seq_model.primitive_encoder
        self.feature_encoder = brep_seq_model.feature_encoder
        self.seq_decoder = brep_seq_model.seq_decoder

        self.generator = Generator(args.dim_n, 1024, args.dim_z)
        self.discriminator = Discriminator(1024, args.dim_z)
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
        self.primitive_encoder.eval()
        self.feature_encoder.eval()
        self.seq_decoder.eval()

        # step 1. train discriminator
        if optimizer_idx == 0:
            self.generator.eval()
            self.discriminator.train()
            # generate real z
            _, z_p_real = self.primitive_encoder(batch)
            _, z_f_real = self.feature_encoder(batch)
            z_real = (z_p_real+z_f_real).detach()
            # how well can it label as real?
            mone = -1 * torch.ones([1], device=batch["node_data"].device)
            logits_real = self.discriminator(z_real)
            logits_real = logits_real.mean(dim=0, keepdim=True)
            real_loss = logits_real * mone

            # generate fake z
            one = torch.ones([1], device=batch["node_data"].device)
            n_1 = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            n_2 = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            z_fake = self.generator(n_1, n_2).detach()
            logits_fake = self.discriminator(z_fake)
            logits_fake = logits_fake.mean(dim=0, keepdim=True)
            fake_loss = logits_fake * one
            gradient_penalty = self.calc_gradient_penalty(self.discriminator, z_real, z_fake)

            Wasserstein_D = logits_real - logits_fake
            self.log("Wasserstein_D", Wasserstein_D, prog_bar=False)
            d_loss = fake_loss + real_loss + gradient_penalty
            self.log("D_loss", d_loss, prog_bar=False)
            return d_loss

        # step 2. train generator
        if optimizer_idx == 1:
            self.generator.train()
            self.discriminator.eval()
            # generate fake z
            n_1 = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            n_2 = torch.randn(self.batch_size, self.dim_n, device=batch["node_data"].device)
            z_fake = self.generator(n_1, n_2)

            # ground truth result (ie: all fake)
            one = torch.ones([1], device=batch["node_data"].device)
            mone = -1 * one
            logits_fake = self.discriminator(z_fake)
            logits_fake = logits_fake.mean(dim=0, keepdim=True)
            g_loss = logits_fake * mone
            self.log("G_loss", g_loss, prog_bar=False)
            return g_loss


    def configure_optimizers(self):
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))  #default 0.5 0.9
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
        return (
            {'optimizer': opt_d, 'frequency': 2},
            {'optimizer': opt_g, 'frequency': 1}
        )

    def generate(self, n_samples, batch_size):
        self.primitive_encoder.eval()
        self.feature_encoder.eval()
        self.seq_decoder.eval()
        self.generator.eval()
        for batch_index in range(n_samples // batch_size):
            # sample noise
            n_1 = torch.randn(batch_size, self.dim_n, device=self.device)
            n_2 = torch.randn(batch_size, self.dim_n, device=self.device)
            generated_z = self.generator(n_1, n_2)

            output = self.seq_decoder(generated_z)
            # output predicted result-------------------------------------------------------------------
            commands_primitive = torch.argmax(torch.softmax(output['commands_primitive'], dim=-1), dim=-1)  # (N, S)
            args_primitive = torch.argmax(torch.softmax(output['args_primitive'], dim=-1), dim=-1) - 1  # (N, S, n_args)
            commands_primitive = commands_primitive.long().detach().cpu().numpy()  # (N, S)
            args_primitive = args_primitive.long().detach().cpu().numpy()  # (N, S, n_args)

            commands_feature = torch.argmax(torch.softmax(output['commands_feature'], dim=-1), dim=-1)  # (N, S)
            args_feature = torch.argmax(torch.softmax(output['args_feature'], dim=-1), dim=-1) - 1  # (N, S, n_args)
            commands_feature = commands_feature.long().detach().cpu().numpy()  # (N, S)
            args_feature = args_feature.long().detach().cpu().numpy()  # (N, S, n_args)

            # output json files--------------------------------------------------------------------------
            for i in range(batch_size):
                end_pos = MAX_N_MAIN - np.sum((commands_primitive[i][:] == EOS_IDX).astype(np.int))
                primitive_type = commands_primitive[i][:end_pos + 1]  # (Seq)
                primitive_param = args_primitive[i][:end_pos + 1][:]  # (Seq, n_args)

                end_pos = MAX_N_SUB - np.sum((commands_feature[i][:] == EOS_IDX).astype(np.int))
                feature_type = commands_feature[i][:end_pos + 1]  # (Seq)
                feature_param = args_feature[i][:end_pos + 1][:]  # (Seq, n_args)

                file_name = "generation_{}.json".format(str(batch_size * batch_index + i))
                file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "results/generated_seq")
                if not os.path.exists(file_path): os.makedirs(file_path)
                vec_to_json(primitive_type, primitive_param, feature_type, feature_param,
                            os.path.join(file_path, file_name))
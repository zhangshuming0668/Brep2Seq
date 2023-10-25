import torch.nn as nn
import torch

class Generator(nn.Module):

    def __init__(self, n_dim, h_dim, z_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # self.model = nn.Sequential(
        #     *block(n_dim, h_dim, normalize=False),
        #     *block(h_dim, h_dim),
        #     *block(h_dim, h_dim),
        #     *block(h_dim, h_dim),
        #     nn.Linear(h_dim, z_dim),
        #     nn.Tanh(),
        # )

        self.gen_main = nn.Sequential(
            *block(n_dim, h_dim, normalize=False),
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
            nn.Linear(h_dim, z_dim),
            nn.Tanh(),
        )
        self.gen_sub = nn.Sequential(
            *block(2*n_dim, h_dim, normalize=False),
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
            nn.Linear(h_dim, z_dim),
            nn.Tanh(),
        )
        self.downsample = nn.Sequential(
            *block(z_dim, h_dim, normalize=False),
            *block(h_dim, h_dim),
            nn.Linear(h_dim, n_dim),
        )


    def forward(self, noise_1, noise_2, z_main_in=None, real_z_main=False):
        # output = self.model(noise)

        z_main = self.gen_main(noise_1)
        if(real_z_main):
            z_main_ = self.downsample(z_main_in)
            input_2 = torch.cat([z_main_, noise_2], dim=1)
            z_sub = self.gen_sub(input_2)
            output = torch.cat([z_main_in, z_sub], dim=1)
        else:
            z_main_ = self.downsample(z_main)
            input_2 = torch.cat([z_main_, noise_2], dim=1)
            z_sub = self.gen_sub(input_2)
            output = torch.cat([z_main, z_sub], dim=1)

        return output

class Discriminator(nn.Module):

    def __init__(self, h_dim, z_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, 1),
        )

    def forward(self, inputs):
        output = self.model(inputs)
        return output.view(-1)

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

        self.gen_z_p = nn.Sequential(
            *block(n_dim, h_dim, normalize=False),
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
            nn.Linear(h_dim, z_dim),
            nn.Tanh(),
        )
        self.gen_z_f = nn.Sequential(
            *block(n_dim, h_dim, normalize=False),
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


    def forward(self, noise_1, noise_2, real_z_p=None):
        if(real_z_p):
            z_p_ = self.downsample(real_z_p)
            input_2 = z_p_ + noise_2
            z_f = self.gen_z_f(input_2)
            output = real_z_p + z_f
        else:
            z_p = self.gen_z_p(noise_1)
            z_p_ = self.downsample(z_p)
            input_2 = z_p_ + noise_2
            z_f = self.gen_z_f(input_2)
            output = z_p + z_f
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

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


# baseline NN
class NN_baseline(nn.Module):
    def __init__(self, nlayers=3, whh=None, in_size=256, lkns=0):

        super(NN_baseline, self).__init__()

        self.lkns=lkns
        self.nlayers=nlayers
        if whh is True:
            self.input_layer = nn.Linear(5, in_size)
        else:
            self.input_layer = nn.Linear(4, in_size)

        for i in range(self.nlayers-2):
            exec_name = "self.hidden_layer01=nn.Linear(in_size,in_size)"
            exec_name = exec_name[:18]+str(i+1)+exec_name[19:]
            exec(exec_name)

        self.output_layer = nn.Linear(in_size, 105)

    def forward(self, x):

        output = F.leaky_relu(self.input_layer(x), negative_slope=self.lkns)

        for i in range(self.nlayers - 2):
            exec_name = "output=F.leaky_relu(self.hidden_layer01(output),negative_slope=self.lkns)"
            exec_name = exec_name[:38]+str(i+1)+exec_name[39:]
            exec(exec_name)

        output = torch.mul(torch.tanh(self.output_layer(output)), 1.06)

        return output


class cVAE(nn.Module):
    def __init__(self, nlayers=3, whh = None, in_size=256, code_dim=10, lkns=0):

        super(cVAE, self).__init__()
        self.nlayers=nlayers
        self.lkns=lkns
        if whh is True:
            self.encoder_input_layer = nn.Linear(105 + 5, in_size)
            self.decoder_input_layer = nn.Linear(code_dim + 5, in_size)
        else:
            self.encoder_input_layer = nn.Linear(105 + 4, in_size)
            self.decoder_input_layer = nn.Linear(code_dim + 4, in_size)


        # for encoder
        for i in range(self.nlayers-2):
            exec_name = "self.encoder_hidden_layer01=nn.Linear(in_size,in_size)"
            exec_name = exec_name[:26] + str(i + 1) + exec_name[27:]
            exec(exec_name)

        self.encoder_output_mu = nn.Linear(in_size, code_dim)
        self.encoder_output_sigma = nn.Linear(in_size, code_dim)

        # for decoder
        for i in range(self.nlayers-2):
            exec_name = "self.decoder_hidden_layer01=nn.Linear(in_size,in_size)"
            exec_name = exec_name[:26] + str(i + 1) + exec_name[27:]
            exec(exec_name)

        self.decoder_output_layer=nn.Linear(in_size, 105)

    def encoder(self, x, wh):

        input_feature = torch.cat([x, wh], axis=1)

        output = F.leaky_relu(self.encoder_input_layer(input_feature), negative_slope=self.lkns)

        for i in range(self.nlayers - 2):
            exec_name = "output=F.leaky_relu(self.encoder_hidden_layer01(output),negative_slope=self.lkns)"
            exec_name = exec_name[:46] + str(i + 1) + exec_name[47:]
            exec(exec_name)

        z_mu = self.encoder_output_mu(output)

        z_logvar = self.encoder_output_sigma(output)

        return z_mu, z_logvar

    def reparameterize(self, z_mu, z_logvar):

        std = z_logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        if torch.cuda.is_available():
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps)
        return eps.mul(std).add_(z_mu)

    def decoder(self, z_code, wh):

        input_code =  torch.cat([z_code, wh], axis=1)

        output = F.leaky_relu(self.decoder_input_layer(input_code),  negative_slope=self.lkns)

        for i in range(self.nlayers - 2):
            exec_name = "output=F.leaky_relu(self.decoder_hidden_layer01(output),negative_slope=self.lkns)"
            exec_name = exec_name[:46] + str(i + 1) + exec_name[47:]
            exec (exec_name)

        reconstructed = torch.mul(torch.tanh(self.decoder_output_layer(output)),1.06)

        return reconstructed

    def forward(self, x, wh):

        z_mu, z_logvar = self. encoder(x, wh)

        z_code = self.reparameterize(z_mu, z_logvar)

        return self.decoder(z_code, wh), z_mu, z_logvar

class cGAN_D(nn.Module):
    def __init__(self, nlayers=3, whh=None, in_size=256, lkns=0):
        super(cGAN_D, self).__init__()
        self.nlayers=nlayers
        self.lkns=lkns

        if whh is True:
            self.D_input_layer = nn.Linear(105 + 5, in_size)
        else:
            self.D_input_layer = nn.Linear(105+4, in_size)

        for i in range(self.nlayers - 2):
            exec_name = "self.D_hidden_layer01=nn.Linear(in_size, in_size)"
            exec_name = exec_name[:20] + str(i + 1) + exec_name[21:]
            exec (exec_name)

        self.D_output_layer = nn.Linear(in_size, 1)
    def forward(self,x, wh):
        input_code = torch.cat([x, wh], axis=1)

        output = F.leaky_relu(self.D_input_layer(input_code), negative_slope=self.lkns)

        for i in range(self.nlayers - 2):
            exec_name = "output=F.leaky_relu(self.D_hidden_layer01(output), negative_slope=self.lkns)"
            exec_name = exec_name[:40] + str(i + 1) + exec_name[41:]
            exec (exec_name)
        output = F.leaky_relu(self.D_hidden_layer01(output), negative_slope=self.lkns)

        output = torch.sigmoid(self.D_output_layer(output))

        return output
class cGAN_G(nn.Module): # concatecate directly
    def __init__(self, nlayers=3, whh=None, in_size=256, code_dim=10, lkns=0):
        super(cGAN_G, self).__init__()
        self.lkns=lkns
        self.nlayers=nlayers
        if whh is True:
            self.G_input_layer = nn.Linear( code_dim + 5, in_size)
        else:
            self.G_input_layer = nn.Linear( code_dim + 4, in_size)

        for i in range(self.nlayers - 2):
            exec_name = "self.G_hidden_layer01=nn.Linear(in_size,in_size)"
            exec_name = exec_name[:20] + str(i + 1) + exec_name[21:]
            exec (exec_name)
        self.G_output_layer = nn.Linear(in_size, 105)

    def forward(self, z, wh):
        input_code = torch.cat([z, wh], axis=1)

        output = F.leaky_relu(self.G_input_layer(input_code), negative_slope=self.lkns)

        for i in range(self.nlayers - 2):
            exec_name = "output=F.leaky_relu(self.G_hidden_layer01(output), negative_slope=self.lkns)"
            exec_name = exec_name[:40] + str(i + 1) + exec_name[41:]
            exec (exec_name)

        output = torch.mul(torch.tanh(self.G_output_layer(output)),1.06)

        return output



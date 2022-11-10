import torch
import torch.nn as nn
from torch.nn import init
from torchsummary import summary
from math import sqrt
import numpy as np
from monai.networks.nets import ViT
from typing import Tuple
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter
import torch.nn.utils.spectral_norm as spectral_norm






class CNN(nn.Module):
    def __init__(self, in_channels=4, inplace=True):
        super(CNN, self).__init__()
        NS = 0.2
        filters = [64, 128, 256, 512, 1024, 2048]
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, filters[0], 3, 2, 1, bias=False),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  80 x 96 x 64
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(filters[0], filters[1], 3, 2, 1, bias=False),
            nn.BatchNorm3d(filters[1]),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  40 x 48 x 32
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(filters[1], filters[2], 3, 2, 1, bias=False),
            nn.BatchNorm3d(filters[2]),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  20 x 24 x 16
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(filters[2], filters[3], 3, 2, 1, bias=False),
            nn.BatchNorm3d(filters[3]),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  10 x 12 x 8
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(filters[3], filters[4], 3, 2, 1, bias=False),
            nn.BatchNorm3d(filters[4]),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  5 x 6 x 4
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(filters[4], filters[5], 3, 2, 1, bias=False),
            nn.BatchNorm3d(filters[5]),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  3 x 3 x 2
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            m.weight = nn.init.kaiming_normal_(m.weight, a=0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.zero_()


    def forward(self, x):
        N = x.size(0)
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        output = torch.cat((x.view(N, -1),
                            out1.view(N, -1),
                            out2.view(N, -1),
                            out3.view(N, -1),
                            out4.view(N, -1),
                            out5.view(N, -1),
                            out6.view(N, -1)),
                            dim=1)

        return output





'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1       [-1, 64, 80, 96, 64]           6,912
       BatchNorm3d-2       [-1, 64, 80, 96, 64]             128
         LeakyReLU-3       [-1, 64, 80, 96, 64]               0
            Conv3d-4      [-1, 128, 40, 48, 32]         221,184
       BatchNorm3d-5      [-1, 128, 40, 48, 32]             256
         LeakyReLU-6      [-1, 128, 40, 48, 32]               0
            Conv3d-7      [-1, 256, 20, 24, 16]         884,736
       BatchNorm3d-8      [-1, 256, 20, 24, 16]             512
         LeakyReLU-9      [-1, 256, 20, 24, 16]               0
           Conv3d-10       [-1, 64, 80, 96, 64]           6,912
           Conv3d-11       [-1, 512, 10, 12, 8]       3,538,944
      BatchNorm3d-12       [-1, 512, 10, 12, 8]           1,024
        LeakyReLU-13       [-1, 512, 10, 12, 8]               0
           Conv3d-14        [-1, 1024, 5, 6, 4]      14,155,776
      BatchNorm3d-15        [-1, 1024, 5, 6, 4]           2,048
        LeakyReLU-16        [-1, 1024, 5, 6, 4]               0
      BatchNorm3d-17       [-1, 64, 80, 96, 64]             128
           Conv3d-18        [-1, 2048, 3, 3, 2]      56,623,104
        LeakyReLU-19       [-1, 64, 80, 96, 64]               0
      BatchNorm3d-20        [-1, 2048, 3, 3, 2]           4,096
        LeakyReLU-21        [-1, 2048, 3, 3, 2]               0
              CNN-22             [-1, 57667584]               0
           Conv3d-23      [-1, 128, 40, 48, 32]         221,184
      BatchNorm3d-24      [-1, 128, 40, 48, 32]             256
        LeakyReLU-25      [-1, 128, 40, 48, 32]               0
           Conv3d-26      [-1, 256, 20, 24, 16]         884,736
      BatchNorm3d-27      [-1, 256, 20, 24, 16]             512
        LeakyReLU-28      [-1, 256, 20, 24, 16]               0
           Conv3d-29       [-1, 512, 10, 12, 8]       3,538,944
      BatchNorm3d-30       [-1, 512, 10, 12, 8]           1,024
        LeakyReLU-31       [-1, 512, 10, 12, 8]               0
           Conv3d-32        [-1, 1024, 5, 6, 4]      14,155,776
      BatchNorm3d-33        [-1, 1024, 5, 6, 4]           2,048
        LeakyReLU-34        [-1, 1024, 5, 6, 4]               0
           Conv3d-35        [-1, 2048, 3, 3, 2]      56,623,104
      BatchNorm3d-36        [-1, 2048, 3, 3, 2]           4,096
        LeakyReLU-37        [-1, 2048, 3, 3, 2]               0
              CNN-38             [-1, 57667584]               0
================================================================
Total params: 150,877,440
Trainable params: 150,877,440
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 60.00
Forward/backward pass size (MB): 2799.75
Params size (MB): 575.55
Estimated Total Size (MB): 3435.30
----------------------------------------------------------------
'''



class SingleL1_CNN(nn.Module):
    def __init__(self, in_channels=4, inplace=True):
        super(SingleL1_CNN, self).__init__()
        NS = 0.2
        filters = [64, 128, 256, 512, 1024, 2048]
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, filters[0], 3, 2, 1, bias=False),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  80 x 96 x 64
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(filters[0], filters[1], 3, 2, 1, bias=False),
            nn.BatchNorm3d(filters[1]),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  40 x 48 x 32
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(filters[1], filters[2], 3, 2, 1, bias=False),
            nn.BatchNorm3d(filters[2]),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  20 x 24 x 16
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(filters[2], filters[3], 3, 2, 1, bias=False),
            nn.BatchNorm3d(filters[3]),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  10 x 12 x 8
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(filters[3], filters[4], 3, 2, 1, bias=False),
            nn.BatchNorm3d(filters[4]),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  5 x 6 x 4
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(filters[4], filters[5], 3, 2, 1, bias=False),
            nn.BatchNorm3d(filters[5]),
            nn.LeakyReLU(negative_slope=NS, inplace=inplace),
            # nn.Dropout3d(0.2)
            # state size.  3 x 3 x 2
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            m.weight = nn.init.kaiming_normal_(m.weight, a=0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.zero_()


    def forward(self, x):
        N = x.size(0)
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        output = out6.view(N, -1)

        return output




# class SpectralNorm(object):

#     def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
#         self.name = name
#         self.dim = dim
#         if n_power_iterations <= 0:
#             raise ValueError('Expected n_power_iterations to be positive, but '
#                              'got n_power_iterations={}'.format(n_power_iterations))
#         self.n_power_iterations = n_power_iterations
#         self.eps = eps

#     def compute_weight(self, module):
#         weight = getattr(module, self.name + '_orig')
#         u = getattr(module, self.name + '_u')
#         weight_mat = weight
#         if self.dim != 0:
#             # permute dim to front
#             weight_mat = weight_mat.permute(self.dim,
#                                             *[d for d in range(weight_mat.dim()) if d != self.dim])
#         height = weight_mat.size(0)
#         weight_mat = weight_mat.reshape(height, -1)
#         with torch.no_grad():
#             for _ in range(self.n_power_iterations):
#                 # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
#                 # are the first left and right singular vectors.
#                 # This power iteration produces approximations of `u` and `v`.
#                 v = normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
#                 u = normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)

#         sigma = torch.dot(u, torch.matmul(weight_mat, v))
#         weight = weight / sigma
#         return weight, u

#     def remove(self, module):
#         weight = getattr(module, self.name)
#         delattr(module, self.name)
#         delattr(module, self.name + '_u')
#         delattr(module, self.name + '_orig')
#         module.register_parameter(self.name, torch.nn.Parameter(weight))

#     def __call__(self, module, inputs):
#         if module.training:
#             weight, u = self.compute_weight(module)
#             setattr(module, self.name, weight)
#             setattr(module, self.name + '_u', u)
#         else:
#             r_g = getattr(module, self.name + '_orig').requires_grad
#             getattr(module, self.name).detach_().requires_grad_(r_g)

#     @staticmethod
#     def apply(module, name, n_power_iterations, dim, eps):
#         fn = SpectralNorm(name, n_power_iterations, dim, eps)
#         weight = module._parameters[name]
#         height = weight.size(dim)

#         u = normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
#         delattr(module, fn.name)
#         module.register_parameter(fn.name + "_orig", weight)
#         # We still need to assign weight back as fn.name because all sorts of
#         # things may assume that it exists, e.g., when initializing weights.
#         # However, we can't directly assign as it could be an nn.Parameter and
#         # gets added as a parameter. Instead, we register weight.data as a
#         # buffer, which will cause weight to be included in the state dict
#         # and also supports nn.init due to shared storage.
#         module.register_buffer(fn.name, weight.data)
#         module.register_buffer(fn.name + "_u", u)

#         module.register_forward_pre_hook(fn)
#         return fn


# def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
#     r"""Applies spectral normalization to a parameter in the given module.
#     .. math::
#          \mathbf{W} &= \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
#          \sigma(\mathbf{W}) &= \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
#     Spectral normalization stabilizes the training of discriminators (critics)
#     in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
#     with spectral norm :math:`\sigma` of the weight matrix calculated using
#     power iteration method. If the dimension of the weight tensor is greater
#     than 2, it is reshaped to 2D in power iteration method to get spectral
#     norm. This is implemented via a hook that calculates spectral norm and
#     rescales weight before every :meth:`~Module.forward` call.
#     See `Spectral Normalization for Generative Adversarial Networks`_ .
#     .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957
#     Args:
#         module (nn.Module): containing module
#         name (str, optional): name of weight parameter
#         n_power_iterations (int, optional): number of power iterations to
#             calculate spectal norm
#         eps (float, optional): epsilon for numerical stability in
#             calculating norms
#         dim (int, optional): dimension corresponding to number of outputs,
#             the default is 0, except for modules that are instances of
#             ConvTranspose1/2/3d, when it is 1
#     Returns:
#         The original module with the spectal norm hook
#     Example::
#         >>> m = spectral_norm(nn.Linear(20, 40))
#         Linear (20 -> 40)
#         >>> m.weight_u.size()
#         torch.Size([20])
#     """
#     if dim is None:
#         if isinstance(module, (torch.nn.ConvTranspose1d,
#                                torch.nn.ConvTranspose2d,
#                                torch.nn.ConvTranspose3d)):
#             dim = 1
#         else:
#             dim = 0
#     SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
#     return module



class CGAN_Discriminator(nn.Module):
    def __init__(self, n_classes):
        super(CGAN_Discriminator, self).__init__()
        NS = 0.2
        #  D = ( (D+ 2*pad - dilation*(ks - 1) - 1)//stride + 1)
        self.model = nn.Sequential(
            spectral_norm(nn.Conv3d(n_classes, 64, 4, 2, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(NS, inplace=True),
            nn.Dropout3d(0.2),
            # (N, 64, 80, 96, 80)
            spectral_norm(nn.Conv3d(64, 128, 4, 2, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(NS, inplace=True),
            nn.Dropout3d(0.2),
            # (N, 128, 40, 48, 40)
            spectral_norm(nn.Conv3d(128, 256, 4, 2, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(NS, inplace=True),
            nn.Dropout3d(0.2),
            # (N, 256, 20, 24, 20)
            spectral_norm(nn.Conv3d(256, 512, 4, 2, 1)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(NS, inplace=True),
            nn.Dropout3d(0.2),
            # (N, 512, 10, 12, 10)
            spectral_norm(nn.Conv3d(512, 512, 4, 1, 1)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(NS, inplace=True),
            nn.Dropout3d(0.2),
            # (n, 512, 9, 11, 9)
            spectral_norm(nn.Conv3d(512, 1, 4, 1, 1)),
            # (n, 1, 8, 10, 8)
        )
        self.linear = nn.Linear(1*8*10*8, 1)

    def forward(self, img, labels):
        '''
            labels: (N, 1, x, y, z)
        '''
        x = torch.cat((img, labels), 1)         #( N, c, x, y, z)
        out = self.model(x)           # (N, c, w, h, d)
        out = self.linear(out.view(out.size(0), -1))        # (N, 1)
        return out

  

'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1       [-1, 64, 80, 96, 64]           3,520
       BatchNorm3d-2       [-1, 64, 80, 96, 64]             128
         LeakyReLU-3       [-1, 64, 80, 96, 64]               0
            Conv3d-4       [-1, 64, 80, 96, 64]           3,520
           Dropout-5       [-1, 64, 80, 96, 64]               0
       BatchNorm3d-6       [-1, 64, 80, 96, 64]             128
         LeakyReLU-7       [-1, 64, 80, 96, 64]               0
            Conv3d-8      [-1, 128, 40, 48, 32]         221,312
           Dropout-9       [-1, 64, 80, 96, 64]               0
      BatchNorm3d-10      [-1, 128, 40, 48, 32]             256
        LeakyReLU-11      [-1, 128, 40, 48, 32]               0
          Dropout-12      [-1, 128, 40, 48, 32]               0
           Conv3d-13      [-1, 128, 40, 48, 32]         221,312
           Conv3d-14      [-1, 256, 20, 24, 16]         884,992
      BatchNorm3d-15      [-1, 256, 20, 24, 16]             512
        LeakyReLU-16      [-1, 256, 20, 24, 16]               0
          Dropout-17      [-1, 256, 20, 24, 16]               0
      BatchNorm3d-18      [-1, 128, 40, 48, 32]             256
           Conv3d-19       [-1, 512, 10, 12, 8]       3,539,456
        LeakyReLU-20      [-1, 128, 40, 48, 32]               0
      BatchNorm3d-21       [-1, 512, 10, 12, 8]           1,024
        LeakyReLU-22       [-1, 512, 10, 12, 8]               0
          Dropout-23       [-1, 512, 10, 12, 8]               0
          Dropout-24      [-1, 128, 40, 48, 32]               0
           Conv3d-25        [-1, 1024, 5, 6, 4]      14,156,800
           Conv3d-26      [-1, 256, 20, 24, 16]         884,992
      BatchNorm3d-27        [-1, 1024, 5, 6, 4]           2,048
        LeakyReLU-28        [-1, 1024, 5, 6, 4]               0
          Dropout-29        [-1, 1024, 5, 6, 4]               0
      BatchNorm3d-30      [-1, 256, 20, 24, 16]             512
        LeakyReLU-31      [-1, 256, 20, 24, 16]               0
          Dropout-32      [-1, 256, 20, 24, 16]               0
           Conv3d-33       [-1, 512, 10, 12, 8]       3,539,456
      BatchNorm3d-34       [-1, 512, 10, 12, 8]           1,024
        LeakyReLU-35       [-1, 512, 10, 12, 8]               0
          Dropout-36       [-1, 512, 10, 12, 8]               0
           Conv3d-37        [-1, 1024, 5, 6, 4]      14,156,800
      BatchNorm3d-38        [-1, 1024, 5, 6, 4]           2,048
        LeakyReLU-39        [-1, 1024, 5, 6, 4]               0
          Dropout-40        [-1, 1024, 5, 6, 4]               0
           Linear-41                    [-1, 1]         122,881
CGAN_Discriminator-42                    [-1, 1]               0
           Linear-43                    [-1, 1]         122,881
CGAN_Discriminator-44                    [-1, 1]               0
================================================================
Total params: 37,865,858
Trainable params: 37,865,858
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 15.00
Forward/backward pass size (MB): 2557.50
Params size (MB): 144.45
Estimated Total Size (MB): 2716.95
----------------------------------------------------------------
'''

class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        
    def forward(self, x):
        return self.norm(x)

class TransSegAN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Tuple[int, int, int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = 3
        self.patch_size = (16, 16, 16)
        self.classification = False

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.norm = CustomNorm('bn', hidden_size)


    def forward(self, x):
        N = x.shape[0]
        output, hidden_states_out = self.vit(x)         # (b, w*h*d, c)
        for i in range(len(hidden_states_out)):
            to_cat = self.norm(hidden_states_out[i].permute(0, 2, 1)).permute(0, 2, 1).contiguous()
            output = torch.cat((output.view(N, -1), to_cat.view(N, -1)), dim=1)
        
        return output








if __name__=='__main__':
    if False:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = CGAN_Discriminator(n_classes=1).to(device)
        net = nn.DataParallel(net.to(device), device_ids=[0,1,2,3],output_device=0)
        summary(net, (1, 160, 192, 128))

    if False:
        m = spectral_norm(nn.Conv3d(3, 64, 4, 2, 1))
        print(m.weight_u)

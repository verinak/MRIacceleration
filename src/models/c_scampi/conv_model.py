import torch
import torch.nn as nn
import numpy as np

from utils.cartesian.transforms import image2kspace_torch, cartesian_backward
from utils.data_utils import toComplex, toReal
from utils.cartesian.sampling import data_consistency


class ConvDecoder(nn.Module):
    def __init__(self, input_dim, num_layers, num_channels, num_output_channels, out_size, in_size):
        super(ConvDecoder, self).__init__()

        kernel_size = 3
        strides = [1] * (num_layers - 1)

        # Compute upsampling sizes
        scale_x = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1))
        scale_y = (out_size[1] / in_size[1]) ** (1. / (num_layers - 1))
        hidden_size = [
            (int(np.ceil(scale_x ** n * in_size[0])), int(np.ceil(scale_y ** n * in_size[1])))
            for n in range(1, num_layers - 1)
        ] + [out_size]

        self.net = nn.Sequential()

        for i in range(num_layers - 1):
            self.net.add_module(str(len(self.net)), nn.Upsample(size=hidden_size[i], mode='nearest'))

            in_ch = input_dim if i == 0 else num_channels
            conv = nn.Conv2d(in_ch, num_channels, kernel_size, strides[i], padding=(kernel_size - 1) // 2, bias=True)

            self.net.add_module(str(len(self.net)), conv)
            self.net.add_module(str(len(self.net)), nn.ReLU())
            self.net.add_module(str(len(self.net)), nn.BatchNorm2d(num_channels, affine=True))

        self.net.add_module(str(len(self.net)), nn.Conv2d(num_channels, num_channels, kernel_size, strides[-1], padding=(kernel_size - 1) // 2, bias=True))
        self.net.add_module(str(len(self.net)), nn.ReLU())
        self.net.add_module(str(len(self.net)), nn.BatchNorm2d(num_channels, affine=True))
        self.net.add_module(str(len(self.net)), nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))

    def forward(self, x, scale_out=1):
        return self.net(x) * scale_out


class DipConvDecoder(nn.Module):
    """Implementation of ConvDecoder for Deep Image Prior in MRI."""

    def __init__(self, input_dim, num_layers, num_channels, num_output_channels, out_size, in_size,
                 mask, produce=True, apply_data_consistency=False, k0=None,
                 coilmap=None, scale_out=1.0):
        super(DipConvDecoder, self).__init__()

        self.model = ConvDecoder(input_dim, num_layers, num_channels, num_output_channels, out_size, in_size)
        self.mask = mask
        self.produce = produce
        self.data_consistency = apply_data_consistency
        self.k0 = k0
        self.coilmap = coilmap
        self.scale_out = scale_out

        if self.data_consistency:
            assert self.k0 is not None, "When data_consistency is selected, DipConvDecoder needs a k0 to be passed!"

    def forward(self, x, **kwargs):
        logits = self.model(x, scale_out=self.scale_out)

        if self.data_consistency and self.produce:
            logits = toComplex(logits, dim=1)
            if self.coilmap is not None:
                logits = self.coilmap * logits.broadcast_to(self.coilmap.shape)
            logits = image2kspace_torch(logits, [0, 0, 1, 1])
            logits = toReal(logits, dim=1)

            logits = data_consistency(logits, self.k0, self.mask.int())
            logits = toComplex(logits, dim=1)
            logits = cartesian_backward(logits, self.coilmap)

        return logits

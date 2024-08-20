# This code was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# Below is the code for the model
# Contact: medchaoukiziara@gmail.com || me@chaouki.pro


import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Encoder_EffNet, Decoder_EffNet, Encoder_ResNet, Decoder_ResNet, HistLayer

import torch
import torch.nn as nn

class DepthHist(nn.Module):

    """ DepthHist Model: 
        This model supports two configurations: 
            1. Simple Mode: 
                This is a straightforward encoder-decoder approach. 
                Only the encoder and decoder are used.
                Outputs are passed through a convolutional layer and returned directly. 
            2. Non-Simple Mode: 
                This is an advanced model that uses encoder-decoder along with a histogram layer for depth computation.
                After passing through the encoder-decoder, the output is further processed using a histogram layer. 
                Depth is calculated by multiplying the output with the histogram and summing along a specific dimension. 
        Attributes: 
            - backend: The backbone model used for encoding. 
            - bins: The number of bins used in the histogram (only for non-simple mode). 
            - simple: A boolean flag indicating whether to use the simple mode. 
    """
    
    def __init__(self, backend, bins, simple):
        super(DepthHist, self).__init__()
        # Initialize the encoder based on the backend provided
        self.encoder = Encoder_EffNet(backend) if backend is not None else Encoder_ResNet()

        # Initialize the decoder based on the backend provided
        self.decoder = Decoder_EffNet(num_classes=128) if backend is not None else Decoder_ResNet()

        # Determine if the model is in simple mode or not
        self.simple = simple

        # Number of bins for the output
        self.bins = bins

        # Initialize the output convolution layer and histogram if not in simple mode
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, 1 if self.simple else bins, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # If not in simple mode, initialize the histogram layer
        if not self.simple:
            print(f"This model will use {bins} bins")
            self.Histogram = HistLayer(bins=bins)
        else : 
            print(f"This model will use a basic Encoder Decoder Arch without bining")


    def forward(self, x, **kwargs):
        # Pass through the encoder and decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, **kwargs)
        
        # Pass the decoded output through the convolution layer
        unet_out = self.conv_out(decoded)

        # Return the appropriate output based on whether the model is in simple mode or not
        return self._compute_output(unet_out, x, decoded)

    def _compute_output(self, unet_out, x, decoded):
        if self.simple:
            return unet_out
        else:
            # Compute histogram and depth for non-simple mode
            histogram = self.Histogram(unet_out, x, decoded)
            depth = torch.sum(unet_out * histogram, dim=1, keepdim=True)
            return depth

    @classmethod
    def build(cls, bins, simple, backbone, **kwargs):
        # Method to build the model with the appropriate backbone
        print('Building Encoder-Decoder model..', end='\n')
        if backbone == "efficientnet":
            basemodel_name = 'tf_efficientnet_b5_ap'
            print(f'\n\nLoading base model {basemodel_name}\n', end=' ')
            basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True, verbose=False)
            print('Done.')

            # Remove global pooling and classifier layers
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()
        else:
            print(f'\n\nLoading base model ResNet \n', end=' ')
            print('Done.')
            basemodel = None

        # Instantiate the class with the appropriate configuration
        return cls(basemodel, bins, simple, **kwargs)

    def get_1x_lr_params(self):
        # Return parameters of the encoder for lower learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):
        # Return parameters of the decoder and other layers for higher learning rate
        modules = [self.decoder, self.conv_out]
        if not self.simple:
            modules.append(self.Histogram)

        for m in modules:
            yield from m.parameters()


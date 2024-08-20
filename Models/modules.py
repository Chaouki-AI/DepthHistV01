# This Code was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# here is the main modules used to build the model
# Contact: medchaoukiziara@gmail.com || me@chaouki.pro

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class HistLayer(nn.Module):
    def __init__(self, bins=81):
        super(HistLayer, self).__init__()
        self.bins = bins
        
        # QKV projection layers
        self.query_conv    = nn.Conv2d(3    , bins, kernel_size=3, stride=3, padding=0)
        self.key_conv      = nn.Conv2d(128  , bins, kernel_size=3, stride=3, padding=0)
        self.value_conv    = nn.Conv2d(bins , bins, kernel_size=3, stride=3, padding=0)
        self.t             = nn.Parameter(torch.randn(1))
        self.temperature   = nn.Parameter(torch.randn(1, bins, bins))
        self.pos_encodings = nn.Parameter(torch.rand(15000, bins), requires_grad=True)
                
        # Parameters
        self.epsilon = 1e-6

    
    def forward(self, input, rgb, decoded):
        b, c, h, w = input.size()
        
        # QKV attention mechanism
        key   = self.key_conv(decoded) 
        _, _, hk, wk = key.size()
        
        value = self.value_conv(input).flatten(2) 
        query = F.interpolate(self.query_conv(rgb), key.shape[-2:]).flatten(2) 
        key   = key.flatten(2)
       
        query = (query + self.pos_encodings[:query.shape[2], :].T.unsqueeze(0)).view(b, self.bins, -1)
        key   = (key   + self.pos_encodings[:key.shape[2], :].T.unsqueeze(0)).view(b, self.bins, -1).permute(0, 2, 1)
        value = (value + self.pos_encodings[:value.shape[2], :].T.unsqueeze(0)).view(b, self.bins, -1)

        
        attention_scores = torch.bmm(query, key) / self.temperature
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        weighted_value = torch.bmm(attention_weights, value)
        weighted_value = weighted_value.view(b, self.bins, hk, wk)
        weighted_value = F.interpolate(weighted_value, (h, w))


        # Generic operation replacing method
        normalized_diff = (input - weighted_value)**2
        histogram = torch.cumsum(normalized_diff, dim=1)

        return histogram
            
class UpSample_EffNet(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSample_EffNet, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

class Decoder_EffNet(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(Decoder_EffNet, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSample_EffNet(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSample_EffNet(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSample_EffNet(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSample_EffNet(skip_input=features // 8 + 16 + 8, output_features=features // 16)

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[
            11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)

        return out

class Encoder_EffNet(nn.Module):
    def __init__(self, backend):
        super(Encoder_EffNet, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features

class Encoder_ResNet(nn.Module):
    def __init__(self, num_layers=152, num_input_images=1):
        super(Encoder_ResNet, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.encoder = models.resnet152(weights='IMAGENET1K_V1')
        #self.encoder = models.resnext50_32x4d( weights='IMAGENET1K_V1')

        self.conv  = nn.Conv2d(out_channels=64, in_channels=64 , kernel_size = 3, padding = 1, stride= 2, padding_mode = 'reflect')
        self.conv1 = nn.Conv2d(out_channels=128, in_channels=64, kernel_size = 3, padding = 1, stride= 1, padding_mode = 'reflect')
        self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = input_image
        x = self.encoder.conv1(x)
       # print(x.shape)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.conv(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        self.features[0]  = self.conv1(x)
        return self.features
    
class UpSample_ResNet(nn.Module):
    def __init__(self, skip_dim, output_dim):
        super(UpSample_ResNet, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_dim + output_dim, output_dim, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_dim),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_dim),
                                  nn.LeakyReLU())
        
        self.up =  nn.Sequential(nn.ConvTranspose2d(skip_dim, skip_dim, 3, stride=2, padding=1),
                                  nn.BatchNorm2d(skip_dim),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        x = self.up(x)
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)
    
class Decoder_ResNet(nn.Module):
    def __init__(self):
        super(Decoder_ResNet, self).__init__()
        
        self.uper1 = UpSample_ResNet(2048, 1024)
        self.uper2 = UpSample_ResNet(1024, 512)
        self.uper3 = UpSample_ResNet(512, 256)
        self.uper4 = UpSample_ResNet(256, 128)
        
    def forward(self, features):
        u1 = self.uper1(features[-1], features[-2])
        u2 = self.uper2(u1, features[-3])
        u3 = self.uper3(u2, features[-4])
        u4 = self.uper4(u3, features[-5])
        return u4

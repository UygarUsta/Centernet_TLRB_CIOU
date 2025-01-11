import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Transformer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model).float()
        self.pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)




class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act='relu'):
        super(Conv, self).__init__()
        if act is not None:
            if act == 'relu':
                self.convs = nn.Sequential(
                    nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.ReLU(inplace=True) if act else nn.Identity()
                )
            elif act == 'leaky':
                self.convs = nn.Sequential(
                    nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
                )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return self.convs(x)


class UpSample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, 
                                                mode=self.mode, align_corners=self.align_corner)


class ResizeConv(nn.Module):
    def __init__(self, c1, c2, act='relu', size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(ResizeConv, self).__init__()
        self.upsample = UpSample(size=size, scale_factor=scale_factor, mode=mode, align_corner=align_corner)
        self.conv = Conv(c1, c2, k=1, act=act)

    def forward(self, x):
        x = self.conv(self.upsample(x))
        return x



class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        super(DETR, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = Transformer(d_model=hidden_dim, nhead=nheads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        
        # Head for heatmap prediction
        self.heatmap_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.wh = nn.Conv2d(hidden_dim, 2, kernel_size=1)
        
        # Head for regression (bounding box and center offsets)
        self.offset = nn.Conv2d(hidden_dim, 2, kernel_size=1)
        self.deconv = UpSample(scale_factor=2)
        self.bottleneck = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.resize = ResizeConv(256,256,scale_factor=2)
        
    def forward(self, x):
        # Backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Shape: (batch_size, 2048, H, W)
        h = self.conv(x)  # Shape: (batch_size, hidden_dim, H, W)
        H, W = h.shape[-2:]
        h_flat = h.flatten(2).permute(2, 0, 1)  # Shape: (H*W, batch_size, hidden_dim)

        h_flat = self.positional_encoding(h_flat)

        h_trans = self.transformer(h_flat, h_flat)  # Shape remains: (H*W, batch_size, hidden_dim)
        h = h_trans.permute(1, 2, 0).view(-1, hidden_dim, H, W)  # Shape: (batch_size, hidden_dim, H, W)

        
        h = self.resize(h)
        h = self.resize(h)
        h = self.resize(h)
        h = self.bottleneck(h)
        
        # Heatmap prediction
        heatmap = self.heatmap_head(h).sigmoid()  # Shape: (batch_size, num_classes, H, W)
        
        # Regression prediction (bounding box dimensions and center offsets)
        offset = self.offset(h)  # Shape: (batch_size, 2, H, W)
        wh = self.wh(h)
        
        return heatmap,wh,offset
    

#num_classes = 91  # Number of classes in the COCO dataset
hidden_dim = 256
nheads = 8
num_encoder_layers = 6
num_decoder_layers = 6
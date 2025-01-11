import torch
import torch.nn as nn
import torchvision.models as models
import timm
import math
import torchvision 

input_width = 512 
input_height = 512 

name = "convnext_atto"
basemodel = timm.create_model(name, features_only=True, pretrained=True)
o = basemodel(torch.rand(1,3,input_width,input_height))
fsize = []
for x in o:
    #print(x.size(2))
    if x.size(3) == input_width // 8:
        #print("Size is 160",x.shape)
        fsize.append(x.size(1))
        
    if x.size(3) == input_width // 16:
        #print("Size is 80",x.shape)
        fsize.append(x.size(1))
    
    if x.size(3) == input_width // 32:
        #print("Size is 40",x.shape)
        fsize.append(x.size(1))


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()

        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                     1 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator
                                          )
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[:, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))



class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=2):
        super(Upsampling, self).__init__()
        # deconv basic config
        if ksize == 4:
            padding = 1
            output_padding = 0
        elif ksize == 3:
            padding = 1
            output_padding = 1
        elif ksize == 2:
            padding = 0
            output_padding = 0

        self.conv = DeformableConv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.up = nn.ConvTranspose2d(out_channels, out_channels, ksize, stride=stride, padding=padding, output_padding=output_padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        fill_up_weights(self.up)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv(x)))
        x = torch.relu(self.bn2(self.up(x)))
        return x



class centernet(nn.Module):
    def __init__(self,classes):
        super(centernet, self).__init__()
        
        self.classes = classes
        # Resnet-18 as backbone.
        #basemodel = timm.create_model('dla34', features_only=True, pretrained=True)
        #torchvision.models.resnet18(pretrained=True)
        
        # DO NOT FREEZE ResNet weights
        #for param in basemodel.parameters():
        #    param.requires_grad = False
        num_ch = 256
        head_conv = 64
        
        # Select only first layers up when you reach 80x45 dimensions with 256 channels
        self.base_model = basemodel #nn.Sequential(*list(basemodel.children())[:-3])

        self.low_level = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest') 
        self.mid_level = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.high_level = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_ = torchvision.ops.FeaturePyramidNetwork(fsize, 256)
        self.fixer = nn.Conv2d(fsize[-1], num_ch, kernel_size=3, padding=1)

        
        self.outc = nn.Sequential(
                nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
                #DeformableConv2d(num_ch, head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, len(self.classes), kernel_size=1, stride=1))
        
        self.outo = nn.Sequential(
                nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
                #DeformableConv2d(num_ch, head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, 2, kernel_size=1, stride=1))
        
        self.outr = nn.Sequential(
                nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
                #DeformableConv2d(num_ch, head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, 2, kernel_size=1, stride=1))
        
#         self.outa = nn.Sequential(
#                 nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
#                 #DeformableConv2d(num_ch, head_conv),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(head_conv, 8, kernel_size=1, stride=1))

        self.upsample1 = Upsampling(256, 256, ksize=4, stride=2) # 32 -> 16
        self.upsample2 = Upsampling(256, 128, ksize=4, stride=2) # 16 -> 8
        self.upsample3 = Upsampling(128, 256, ksize=4, stride=2) #  8 -> 4
        
        self.downsm = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3,stride=2,padding=1),nn.ReLU())
        
        
    def forward(self, x):
        # [b, 3, 720, 1280]
        
        o = self.base_model(x)

        # ll_ml_hl = []
        # for x in o:
        #     #print(x.size(2))
        #     if x.size(3) == input_width // 8:
        #         #print("Size is 160",x.shape)
        #         ll_ml_hl.append(x)
                
        #     if x.size(3) == input_width // 16:
        #         #print("Size is 80",x.shape)
        #         ll_ml_hl.append(x)
        
        #     if x.size(3) == input_width // 32:
        #         #print("Size is 40",x.shape)
        #         ll_ml_hl.append(x)

        # low_level = ll_ml_hl[0]
        # mid_level = ll_ml_hl[1]
        # high_level = ll_ml_hl[2]

        # high = self.upsampler(self.high_level(high_level)) #256,80,80
        # mid = self.mid_level(mid_level) + high #256,80,80
        # low = self.upsampler(mid) + self.low_level(low_level) #256,160,160 

        # x_feats = OrderedDict()
        # x_feats['feat0'] = low_level
        # x_feats['feat2'] = mid_level
        # x_feats['feat3'] = high_level

        # x = low
        #fpn_res = self.fpn_(x_feats)
        #x = high_level
        x = o[-1] #fpn_res['feat3']
        x = self.fixer(x)
        
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        
        
    
        
        heatmap = self.outc(x).sigmoid() 
        offset = self.outo(x)
        wh = self.outr(x)
        #outcorner = self.outa(x)
        
        return heatmap, wh, offset #, outcorner
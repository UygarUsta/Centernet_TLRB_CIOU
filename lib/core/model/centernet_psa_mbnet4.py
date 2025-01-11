import sys

sys.path.append('.')
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from lib.core.model.shufflenet import shufflenet_v2_x1_0
from lib.core.model.resnet import resnet18

from lib.core.model.fpn import Fpn

from lib.core.model.utils import normal_init
from lib.core.model.utils import SeparableConv2d


from train_config import config as cfg

import timm

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        struct = 'Mobilenetv4'
        if 'Mobilenetv2' in struct:
            self.model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True, exportable=True)
        elif 'ShuffleNetV2' in struct:
            self.model = shufflenet_v2_x1_0(pretrained=True)
        elif 'Resnet18' in struct:
            self.model = resnet18(pretrained=True)
        elif 'Mobilenetv4' in struct:
            self.model = timm.create_model('mobilenetv4_conv_small_050', pretrained=True, features_only=True, exportable=True)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        fms = self.model(inputs)
        return fms[-4:]

class CenterNetHead(nn.Module):
    def __init__(self, nc, head_dims=[128, 128, 128]):
        super().__init__()
        self.cls = SeparableConv2d(head_dims[0], nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.wh = SeparableConv2d(head_dims[0], 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.offset = SeparableConv2d(head_dims[0], 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.iou = SeparableConv2d(head_dims[0], 1, kernel_size=3, stride=1, padding=1, bias=True)
        normal_init(self.cls.pointwise, 0, 0.01, -2.19)
        normal_init(self.wh.pointwise, 0, 0.01, 0)

    def forward(self, inputs):
        cls = self.cls(inputs).sigmoid_()
        wh = self.wh(inputs)
        offset = self.offset(inputs)
        iou = self.iou(inputs).sigmoid_().squeeze(1)
        return cls, offset, wh, iou

class CenterNet(nn.Module):
    def __init__(self, nc, inference=False, coreml=False):
        super().__init__()
        self.nc = nc
        self.down_ratio = cfg.MODEL.global_stride
        self.inference = inference
        self.coreml_ = coreml
        self.backbone = Net()
        self.psa = Attention(cfg.MODEL.backbone_feature_dims[-1])
        self.fpn = Fpn(head_dims=cfg.MODEL.head_dims, input_dims=cfg.MODEL.backbone_feature_dims)
        self.head = CenterNetHead(self.nc, head_dims=cfg.MODEL.head_dims)

        if self.down_ratio == 8:
            self.extra_conv = nn.Sequential(
                SeparableConv2d(cfg.MODEL.backbone_feature_dims[-2], cfg.MODEL.backbone_feature_dims[-1],
                                kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(cfg.MODEL.backbone_feature_dims[-1]),
                nn.ReLU(inplace=True))
        else:
            self.extra_conv = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs):
        fms = self.backbone(inputs)
        fms[-1] = self.psa(fms[-1])  # Apply PSA to the last feature map

        if self.extra_conv is not None:
            extra_fm = self.extra_conv(fms[-1])
            fms.append(extra_fm)
            fms = fms[1:]

        fpn_fm = self.fpn(fms)
        cls, wh, offset , iou = self.head(fpn_fm)

        if not self.inference:
            return cls, wh * 16, offset, iou
        else:
            detections = self.decode(cls, wh * 16, self.down_ratio)
            return detections

    def decode(self, heatmap, wh, stride, K=100):
        def nms(heat, kernel=3):
            ##fast

            heat = heat.permute([0, 2, 3, 1])
            heat, clses = torch.max(heat, dim=3)

            heat = heat.unsqueeze(1)
            scores = torch.sigmoid(heat)

            hmax = nn.MaxPool2d(kernel, 1, padding=1)(scores)
            keep = (scores == hmax).float()

            return scores * keep, clses
        def get_bboxes(wh):

            ### decode the box
            shifts_x = torch.arange(0, (W - 1) * stride + 1, stride,
                                   dtype=torch.int32)

            shifts_y = torch.arange(0, (H - 1) * stride + 1, stride,
                                   dtype=torch.int32)

            y_range, x_range = torch.meshgrid(shifts_y, shifts_x)

            base_loc = torch.stack((x_range, y_range, x_range, y_range), axis=0)  # (h, w，4)

            base_loc = torch.unsqueeze(base_loc, dim=0).to(self.device)

            wh = wh * torch.tensor([1, 1, -1, -1],requires_grad=False).reshape([1, 4, 1, 1]).to(self.device)
            pred_boxes = base_loc - wh

            return pred_boxes

        batch, cat, H, W = heatmap.size()


        score_map, label_map = nms(heatmap)
        pred_boxes=get_bboxes(wh)


        score_map = torch.reshape(score_map, shape=[batch, -1])

        top_score,top_index=torch.topk(score_map,k=K)

        top_score = torch.unsqueeze(top_score, 2)


        if self.coreml_:
            pred_boxes = torch.reshape(pred_boxes, shape=[batch, 4, -1])
            pred_boxes = pred_boxes.permute([0, 2, 1])
            top_index_bboxes=torch.stack([top_index,top_index,top_index,top_index],dim=2)


            pred_boxes = torch.gather(pred_boxes,dim=1,index=top_index_bboxes)

            label_map = torch.reshape(label_map, shape=[batch, -1])
            label_map = torch.gather(label_map,dim=1,index=top_index)
            label_map = torch.unsqueeze(label_map, 2)

            pred_boxes = pred_boxes.float()
            label_map = label_map.float()


            detections = torch.cat([pred_boxes, top_score, label_map], dim=2)

        else:
            pred_boxes = torch.reshape(pred_boxes, shape=[batch, 4, -1])
            pred_boxes=pred_boxes.permute([0,2,1])

            pred_boxes = pred_boxes[:,top_index[0],:]

            label_map = torch.reshape(label_map, shape=[batch, -1])
            label_map = label_map[:,top_index[0]]
            label_map = torch.unsqueeze(label_map, 2)


            pred_boxes = pred_boxes.float()
            label_map = label_map.float()

            detections = torch.cat([ pred_boxes,top_score, label_map], dim=2)

        return detections









if __name__ == '__main__':
    import torch
    import torchvision  
    import time

    
    
    model = CenterNet(10,inference=False)
    #model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True, exportable=True)
    ### load your weights
    model.eval()
    # x = torch.randn(1, 3, 512, 512)  # Example input
    # output = model(x)
    # print("outputs:")
    # print(output[0].shape) 
    # print(output[1].shape) 
    # print(output[2].shape) 
    # print(output[3].shape)
    # print(output[4].shape) 
    # print("------------")

    # model = timm.create_model('mobilenetv4_conv_small_050', pretrained=True, features_only=True, exportable=True)
    # output = model(x)
    # print("outputs:")
    # print(output[0].shape) 
    # print(output[1].shape) 
    # print(output[2].shape) 
    # print(output[3].shape)
    # print(output[4].shape) 
    # print("------------")
    
    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    # Move model to CPU

    psa = Attention(256)
    x = torch.randn(1, 256, 64, 64)  # Example input
    output = psa(x)
    print(output.shape) 


    batch_size = 1
    input_height = 320
    input_width = 320
    device = torch.device('cpu')
    model.to(device)
    dummy_input = torch.randn(batch_size, 3, input_height, input_width).to(device)
    print("Start Tracing")
    model = torch.jit.trace(model, dummy_input)
    print("End Tracing")

    # Create dummy input data
    # Quantize the model for faster CPU inference
    model_quantized = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    model_quantized.eval()
    model_quantized.to(device)



    # Warm-up runs (to exclude initialization overhead)
    with torch.no_grad():
        for _ in range(10):
            _ = model_quantized(dummy_input)
            print(_[0].shape)

    # Timing settings
    num_runs = 100
    start_time = time.time()

    # Run the model multiple times and measure the total time
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model_quantized(dummy_input)

    end_time = time.time()
    total_time = end_time - start_time
    fps = num_runs / total_time

    print(f"Total inference time for {num_runs} runs: {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    quit(0)

    torch.onnx.export(model,
                      dummy_input,
                      "centernet.onnx",
                      opset_version=11,
                      input_names=['image'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      )

    import onnx
    from onnxsim import simplify

    # load your predefined ONNX model
    model = onnx.load("centernet.onnx")

    # convert model
    model_simp, check = simplify(model)
    f = model_simp.SerializeToString()
    file = open("centernet.onnx", "wb")
    file.write(f)
    assert check, "Simplified ONNX model could not be validated"

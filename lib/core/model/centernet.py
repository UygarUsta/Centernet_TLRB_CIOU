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

class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        struct = 'Mobilenetv2'
        if 'Mobilenetv2' in struct:
            self.model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True,exportable=True)
        elif 'ShuffleNetV2' in struct:
            self.model = shufflenet_v2_x1_0(pretrained=True)
        elif 'Resnet18' in struct:
            self.model = resnet18(pretrained=True)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        # do preprocess

        # Convolution layers
        fms = self.model(inputs)

        # for ff in fms:
        #     print(ff.size())

        return fms[-4:]

class CenterNetHead(nn.Module):
    def __init__(self,nc,head_dims=[128,128,128] ):
        super().__init__()



        self.cls =SeparableConv2d(head_dims[0], nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.wh =SeparableConv2d(head_dims[0], 4, kernel_size=3, stride=1, padding=1, bias=True)
        # self.offset =SeparableConv2d(head_dims[0], 2, kernel_size=3, stride=1, padding=1, bias=True)
        # self.iou_head = nn.Conv2d(head_dims[0], 1, kernel_size=3, stride=1, padding=1, bias=True)

        normal_init(self.cls.pointwise, 0, 0.01,-2.19)
        normal_init(self.wh.pointwise, 0, 0.01, 0)



    def forward(self, inputs):


        cls = self.cls(inputs).sigmoid_()
        wh = self.wh(inputs)
        # offset = self.offset(inputs)
        # iou_aware_head = self.iou_head(inputs).sigmoid_().squeeze(1) #[B, H, W]
        

        return cls,wh

class CenterNet(nn.Module):
    def __init__(self,nc,inference=False,coreml=False ):
        super().__init__()

        self.nc = nc
        self.down_ratio=cfg.MODEL.global_stride

        ### control params
        self.inference = inference

        self.coreml_ = coreml



        ###model structure
        self.backbone = Net()

        self.fpn=Fpn(head_dims=cfg.MODEL.head_dims,input_dims=cfg.MODEL.backbone_feature_dims)

        self.head = CenterNetHead(self.nc,head_dims=cfg.MODEL.head_dims)



        if self.down_ratio==8:
            self.extra_conv=nn.Sequential(SeparableConv2d(cfg.MODEL.backbone_feature_dims[-2],cfg.MODEL.backbone_feature_dims[-1],
                                                    kernel_size=3,stride=2,padding=1),
                                          nn.BatchNorm2d(cfg.MODEL.backbone_feature_dims[-1]),
                                          nn.ReLU(inplace=True))
        else:
            self.extra_conv=None



        self.device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    def forward(self, inputs):

        fms = self.backbone(inputs)

        if self.extra_conv is not None:

            extra_fm=self.extra_conv(fms[-1])
            fms.append(extra_fm)
            fms=fms[1:]

        fpn_fm=self.fpn(fms)

        cls, wh = self.head(fpn_fm)


        if not self.inference:
            return cls,wh*16    #,angle
        else:
            detections = self.decode(cls, wh*16, self.down_ratio)
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

    ### load your weights
    model.eval()
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

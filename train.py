from dataloader import centernet_dataset_collate,CenternetDataset
from model import *
import datetime
import os
from data_utils  import xml_to_coco_json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from loss import get_lr_scheduler,set_optimizer_lr
from glob import glob
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import matplotlib.pyplot as plt
from calc_coco_val import gt_check
from centernet import CenterNet_Resnet50
from torch.utils.data import DataLoader
from utils_fit import worker_init_fn,fit_one_epoch,seed_everything
from functools import partial
from CenternetPlus.centernet_plus import CenterNetPlus
from infer_utils import load_model,hardnet_load_model
#from dlamodel import get_pose_net
#from centernet_resnet import resnet_18

folder = "/home/rivian/Desktop/Datasets/derpetv5_xml"

input_shape = (512,512)
batch_size = 8#16
epochs = 200
num_workers = 8
optimizer_type = "adam"
lr_decay_type = "cos"

train_images = glob(os.path.join(folder,"train_images","*.jpg")) + glob(os.path.join(folder,"train_images","*.png")) + glob(os.path.join(folder,"train_images","*.JPG"))
val_images = glob(os.path.join(folder,"val_images","*.jpg")) + glob(os.path.join(folder,"val_images","*.png")) + glob(os.path.join(folder,"val_images","*.JPG"))

train_annotatons = sorted(glob(os.path.join(folder,"train_images","*.xml")))
val_annotations =  sorted(glob(os.path.join(folder,"val_images","*.xml")))

train_images = sorted(train_images)
val_images = sorted(val_images)

val_outputs = xml_to_coco_json(os.path.join(folder,"val_images"), 'val_output_coco.json')
cocoGt = COCO("val_output_coco.json")

classes = []
for (i,v) in cocoGt.cats.items():
    classes.append(v["name"])


print(f"CLASSES = {classes}")

with open("classes.txt","w") as f:
    for i in classes:
        f.write(str(i))
        f.write("\n")

pretrained = True
model_type = "shufflenet"
model_path = "centernet_mobilenetv2_stride4.pth" #"best_epoch_weights_ciou_iou_aware_mbv4_psa_shufflenet_coco_mini_map_117.pth"

if model_type == "shufflenet":
    from lib.core.model.centernet import CenterNet
    model = CenterNet(nc=len(classes))
    if model_path != "":
        model = load_model(model,model_path)



if model_type == "resnet50":
    model = CenterNet_Resnet50(len(classes), pretrained = pretrained)
    model_path = "best_epoch_weights_0.154map_cocomini_resnet50_bubliing.pth"
    device = "cuda"
    if model_path != "":
        model = load_model(model,model_path)

if model_type == "CenterNetPlus":
    model = CenterNetPlus(True,len(classes),'r18')
    model_path = "best_epoch_weights_cplus_r18_cocomini_0.162map.pth"
    device = "cuda"
    if model_path != "":
        model = load_model(model,model_path)


if model_type == "detr":
    from detr.detrmodel import *
    hidden_dim = 256
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    model = DETR(len(classes),hidden_dim,nheads,num_encoder_layers,num_decoder_layers)
    model_path = ""
    device = "cuda"
    if model_path != "":
        model = load_model(model,model_path)

if model_type == "timmodel":
    from timmodel.timmodel import *
    model = centernet(classes)
    model_path = ""
    device = "cuda"
    if model_path != "":
        model = load_model(model,model_path)


if model_type == "DLA":
    from dlamodel import * 
    device = "cuda"
    model = get_pose_net("34",{"hm":len(classes),"wh":2,"offset":2})
    if model_path != "":
        model = load_model(model,model_path)   

if model_type == "hardnet":
    from hardnet import get_pose_net
    device = "cuda"
    model = get_pose_net(85,{"hm":len(classes),"wh":2,"offset":2})
    if model_path != "":
        model = hardnet_load_model(model,model_path)


# if model_type == "resnet18":
#     model = resnet_18(classes)


train_dataset = CenternetDataset(train_images,train_annotatons,input_shape,classes,len(classes),train=True)
val_dataset = CenternetDataset(val_images,val_annotations,input_shape,classes,len(classes),train=False)


cuda = True

fp16 = True
if fp16:
    from torch.cuda.amp import GradScaler as GradScaler
    scaler = GradScaler()
else:
    scaler = None

model_train = model.train()

cudnn.benchmark = True
model_train = model_train.cuda()
#model_train = torch.compile(model_train)

Init_lr = 5e-4
Min_lr  = Init_lr * 0.01
weight_decay = 0
momentum = 0.9
nbs             = 64
lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 5e-2
lr_limit_min    = 2.5e-4 if optimizer_type == 'adam' else 5e-4
Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)


#---------------------------------------#
#   根据optimizer_type选择优化器
#---------------------------------------#
optimizer = {
    'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
    'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
}[optimizer_type]

#---------------------------------------#
#   获得学习率下降的公式
#---------------------------------------#
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epochs)

#---------------------------------------#
#   判断每一个世代的长度
#---------------------------------------#
num_train   = len(train_images)
num_val = len(val_images)
epoch_step      = num_train // batch_size
epoch_step_val  = num_val // batch_size

train_sampler   = None
val_sampler     = None
local_rank      = 0
rank            = 0
seed            = 11
seed_everything(seed)


gen  = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=centernet_dataset_collate, sampler=train_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
gen_val = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=centernet_dataset_collate, sampler=val_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

# for i in range(30):
#     img, hm,wh,regmask = train_dataset[i]

#     img_visualize = gt_check(img,hm,wh)
#     cv2.imshow("img",img_visualize)
#     ch = cv2.waitKey(0)
#     if ch == ord("q"): break 

save_period = 5
best_mean_AP = 0
for epoch in range(epochs):
    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    mean_ap = fit_one_epoch(model_train,model,optimizer,epoch,epoch_step, epoch_step_val, gen, gen_val,epochs,cuda,fp16,scaler,save_period,cocoGt,classes,folder,best_mean_AP)
    if mean_ap > best_mean_AP:
        best_mean_AP = mean_ap

print("Best Mean AP:",best_mean_AP)


##sanity check
# for i in range(30):
#     img, hm,wh,regmask = train_dataset[i]

#     img_visualize = gt_check(img,hm,wh)
#     plt.imshow(img_visualize)
#     plt.imshow(cv2.resize(hm[:,:,0],(img_visualize.shape[1],img_visualize.shape[0])),cmap="jet",alpha=0.4)
#     plt.show()

#     # plt.imshow(np.transpose(img,(1,2,0)))
#     # plt.show()
#     # plt.imshow(wh[:,:,0])
#     # plt.show()

#     # plt.imshow(offset[:,:,0])
#     # plt.show()

#     # plt.imshow(regmask)
#     # plt.show()

# quit(0)



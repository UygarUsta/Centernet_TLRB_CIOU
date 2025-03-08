import math

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from data_utils import extract_coordinates
from gaussan_functions import gaussian2D,gaussian_radius,draw_gaussian
import os 

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def resize_numpy(image, size, letterbox_image):
    image = np.array(image,dtype='float32')
    iw, ih = image.shape[1], image.shape[0]
    w, h = size

    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.full((h, w, 3), 128, dtype=np.uint8)
        top = (h - nh) // 2
        left = (w - nw) // 2
        new_image[top:top+nh, left:left+nw, :] = resized_image
    else:
        new_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return new_image

def preprocess_input(image):
    image   = np.array(image,dtype = np.float32)[:, :, ::-1]
    mean    = [0.40789655, 0.44719303, 0.47026116]
    std     = [0.2886383, 0.27408165, 0.27809834]
    return (image / 255. - mean) / std

def preprocess_input_simple(image):
    image   = image[:, :, ::-1] #np.array(image,dtype = np.float32)[:, :, ::-1]
    #mean    = [0.40789655, 0.44719303, 0.47026116]
    #std     = [0.2886383, 0.27408165, 0.27809834]
    return image / 255. #(image / 255. - mean) / std



class CenternetDataset(Dataset):
    def __init__(self, image_path,annotation_lines, input_shape,classes,num_classes, train):
        super(CenternetDataset, self).__init__()
        self.image_path = image_path
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)

        self.input_shape        = input_shape
        self.output_shape       = (int(input_shape[0]/4) , int(input_shape[1]/4))
        self.classes = classes
        self.num_classes        = num_classes
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #index = index % self.length

        #-------------------------------------------------#
        #   进行数据增强
        #-------------------------------------------------#
        image, box      = self.get_random_data(self.image_path[index],self.annotation_lines[index], self.input_shape, random = self.train)

        batch_hm        = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        #batch_wh        = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg       = np.zeros((self.output_shape[0], self.output_shape[1], 4), dtype=np.float32)
        batch_reg_mask  = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        neighbor_size = 1
        if len(box) != 0:
            boxes = np.array(box[:, :4],dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1)
        H, W = batch_hm.shape[:2]
        for i in range(len(box)):
            bbox    = boxes[i].copy()
            cls_id  = int(box[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                #-------------------------------------------------#
                #   计算真实框所属的特征点
                #-------------------------------------------------#
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                xmin,ymin,xmax,ymax = bbox[:4]
                area = (1 / self.bbox_areas_log_np(bbox[:4])) * 2
                #----------------------------#
                #   绘制高斯热力图
                #----------------------------#
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                for dx in range(-neighbor_size, neighbor_size + 1):
                    for dy in range(-neighbor_size, neighbor_size + 1):
                        nx = ct_int[0] + dx
                        ny = ct_int[1] + dy
        
                        if nx < 0 or nx >= W or ny < 0 or ny >= H:
                            continue
        
                        # tl = y_center - y_min  # Distance to top
                        # tt = x_center - x_min  # Distance to left
                        # br = y_max - y_center  # Distance to bottom
                        # rr = x_max - x_center  # Distance to right
                        adjusted_tlbr = [
                            (nx - xmin),  # Distance from the grid cell to the left edge
                            (ny - ymin),  # Distance from the grid cell to the top edge
                            (xmax - nx),  # Distance from the grid cell to the right edge
                            (ymax - ny)   # Distance from the grid cell to the bottom edge
                        ]
                        batch_reg[ny, nx] = adjusted_tlbr
                        batch_hm[ny, nx, cls_id] = 1
                        batch_reg_mask[ny, nx] = area

        image = np.transpose(preprocess_input(image), (2, 0, 1))

        return image, batch_hm, batch_reg, batch_reg_mask
    
    def bbox_areas_log_np(self,bbox):
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        area = (y_max - y_min + 1) * (x_max - x_min + 1)
        return np.log(area) 


    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image_path,annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        extension = os.path.splitext(image_path)[1]
        #line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(image_path)
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box = extract_coordinates(annotation_line,self.classes)
        #box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                box = np.array(box)
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            box = np.array(box)
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box



class CenternetDatasetMosaicAug(Dataset):
    def __init__(self, image_path, input_shape,classes,num_classes, train):
        super(CenternetDataset, self).__init__()
        self.image_path = image_path
        self.length             = len(self.image_path)

        self.input_shape        = input_shape
        self.output_shape       = (int(input_shape[0]/4) , int(input_shape[1]/4))
        self.classes = classes
        self.num_classes        = num_classes
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #index = index % self.length

        #-------------------------------------------------#
        #   进行数据增强
        #-------------------------------------------------#
        if self.train and np.random.rand() < 0.5:
            # Apply mosaic augmentation
            image, box = self.get_mosaic_data(index)
        else:
            # Load single image
            image, box = self.get_random_data(self.image_path[index], self.input_shape, random=self.train)
        #image, box      = self.get_random_data(self.image_path[index], self.input_shape, random = self.train)

        batch_hm        = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        #batch_wh        = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg       = np.zeros((self.output_shape[0], self.output_shape[1], 4), dtype=np.float32)
        batch_reg_mask  = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        neighbor_size = 1
        if len(box) != 0:
            boxes = np.array(box[:, :4],dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1)
        H, W = batch_hm.shape[:2]
        for i in range(len(box)):
            bbox    = boxes[i].copy()
            cls_id  = int(box[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                #-------------------------------------------------#
                #   计算真实框所属的特征点
                #-------------------------------------------------#
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                xmin,ymin,xmax,ymax = bbox[:4]
                area = (1 / self.bbox_areas_log_np(bbox[:4])) * 2
                #----------------------------#
                #   绘制高斯热力图
                #----------------------------#
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                for dx in range(-neighbor_size, neighbor_size + 1):
                    for dy in range(-neighbor_size, neighbor_size + 1):
                        nx = ct_int[0] + dx
                        ny = ct_int[1] + dy
        
                        if nx < 0 or nx >= W or ny < 0 or ny >= H:
                            continue

                        adjusted_tlbr = [
                            (nx - xmin),  # Distance from the grid cell to the left edge
                            (ny - ymin),  # Distance from the grid cell to the top edge
                            (xmax - nx),  # Distance from the grid cell to the right edge
                            (ymax - ny)   # Distance from the grid cell to the bottom edge
                        ]
                        batch_reg[ny, nx] = adjusted_tlbr
                        batch_hm[ny, nx, cls_id] = 1
                        batch_reg_mask[ny, nx] = area

        image = np.transpose(preprocess_input(image), (2, 0, 1))

        return image, batch_hm, batch_reg, batch_reg_mask
    
    def get_mosaic_data(self, index):
        """Mosaic augmentation: Combines 4 images into one."""
        indices = [index] + [np.random.randint(0, self.length) for _ in range(3)]
        images, all_boxes = [], []
        for idx in indices:
            img_path = self.image_path[idx]
            img, box = self.get_random_data(
                img_path,
                input_shape=(self.input_shape[0] // 2, self.input_shape[1] // 2),
                random=True
            )
            images.append(img)
            all_boxes.append(box)

        # Create mosaic image
        mosaic_image = np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        positions = [
            (0, 0),
            (self.input_shape[1] // 2, 0),
            (0, self.input_shape[0] // 2),
            (self.input_shape[1] // 2, self.input_shape[0] // 2)
        ]
        for i in range(4):
            img = images[i]
            x_offset, y_offset = positions[i]
            h, w = img.shape[0], img.shape[1]
            mosaic_image[y_offset:y_offset + h, x_offset:x_offset + w] = img

        # Combine and adjust boxes
        mosaic_boxes = []
        for i in range(4):
            boxes = all_boxes[i]
            if len(boxes) == 0:
                continue
            boxes = boxes.copy()
            x_off, y_off = positions[i]
            boxes[:, [0, 2]] += x_off
            boxes[:, [1, 3]] += y_off
            mosaic_boxes.append(boxes)
        mosaic_boxes = np.concatenate(mosaic_boxes, axis=0) if len(mosaic_boxes) > 0 else np.array([])

        # Apply flip
        if self.rand() < 0.5:
            mosaic_image = mosaic_image[:, ::-1]
            if len(mosaic_boxes) > 0:
                mosaic_boxes[:, [0, 2]] = self.input_shape[1] - mosaic_boxes[:, [2, 0]]

        # Apply color jitter
        image_data = mosaic_image.astype(np.uint8)
        r = np.random.uniform(-1, 1, 3) * [0.1, 0.7, 0.4] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

        # Filter boxes
        if len(mosaic_boxes) > 0:
            mosaic_boxes[:, 0:2] = np.clip(mosaic_boxes[:, 0:2], 0, self.input_shape[1])
            mosaic_boxes[:, 2] = np.clip(mosaic_boxes[:, 2], 0, self.input_shape[1])
            mosaic_boxes[:, 3] = np.clip(mosaic_boxes[:, 3], 0, self.input_shape[0])
            box_w = mosaic_boxes[:, 2] - mosaic_boxes[:, 0]
            box_h = mosaic_boxes[:, 3] - mosaic_boxes[:, 1]
            mosaic_boxes = mosaic_boxes[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, mosaic_boxes
    
    def bbox_areas_log_np(self,bbox):
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        area = (y_max - y_min + 1) * (x_max - x_min + 1)
        return np.log(area) 


    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image_path, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        extension = os.path.splitext(image_path)[1]
        #line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(image_path)
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        if os.path.isfile(image_path.replace(extension,".xml")):
            annotation_line = image_path.replace(extension,".xml")
            box = extract_coordinates(annotation_line,self.classes)
        else:
            box = []
        #box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                box = np.array(box)
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            box = np.array(box)
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box


class CenternetDatasetTTF(Dataset):
    def __init__(self, image_path, input_shape, classes, num_classes, train):
        super(CenternetDatasetTTF, self).__init__()
        self.image_path = image_path
        self.length = len(self.image_path)

        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0]/4), int(input_shape[1]/4))
        self.classes = classes
        self.num_classes = num_classes
        self.train = train
        self.alpha = 0.54  # TTFNet'ten alınan değer
        self.beta = 0.54   # TTFNet'ten alınan değer

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image, box = self.get_random_data(self.image_path[index], self.input_shape, random=self.train)

        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_reg = np.zeros((self.output_shape[0], self.output_shape[1], 4), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        fake_heatmap =np.zeros((self.output_shape[0], self.output_shape[1]),dtype=np.float32)
        
        if len(box) > 0:
            boxes = np.array(box[:, :4], dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1)
            
            # TTFNet mantığına göre: "larger boxes have lower priority than small boxes"
            # Kutuların alanlarını hesapla
            box_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            
            # Log ölçekleme uygula (TTFNet'teki wh_area_process='log' mantığı)
            box_areas_log = np.log(np.array(box_areas) + 1e-6)  # 0'a bölünmeyi önle
            
            # Kutuları KÜÇÜKTEN BÜYÜĞE doğru sırala (küçük nesnelere öncelik ver)
            sorted_indices = np.argsort(box_areas_log)  # Küçük alanlar önce
            
            H, W = batch_hm.shape[:2]
            
            # Önce tüm kutular için heatmap'leri oluştur
            for idx in sorted_indices[::-1]:
                bbox = boxes[idx].copy()
                cls_id = int(box[idx, -1])

                # Kutu sınırlarını integer olarak al
                xmin, ymin, xmax, ymax = bbox
                
                # Sınırları özellik haritası boyutlarına kırp
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(W - 1, xmax), min(H - 1, ymax)
                
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    #ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    h, w = ymax - ymin, xmax - xmin

                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))

                    temporary_fake = np.zeros((H, W), dtype=np.float32)
                    h_radius = max(1, int(h / 2 * self.alpha))
                    w_radius = max(1, int(w / 2 * self.alpha))
                    self.draw_truncate_gaussian(temporary_fake, ct_int, h_radius, w_radius)
                    box_target_inds = temporary_fake > 0
                    sum_gaussian = temporary_fake[box_target_inds].sum()
                    area = (xmax - xmin) * (ymax - ymin)
                    processed_area = np.log(area + 1e-6)
                    if sum_gaussian > 0:
                        reg_contribution = (temporary_fake[box_target_inds] * processed_area) / sum_gaussian
                        batch_reg_mask[box_target_inds] = reg_contribution
                    np.maximum(batch_hm[:, :, cls_id], temporary_fake, out=batch_hm[:, :, cls_id])
                    y_indices, x_indices = np.where(box_target_inds)
                    for y, x in zip(y_indices, x_indices):
                        adjusted_tlrb = [
                            x - xmin,
                            y - ymin,
                            xmax - x,
                            ymax - y
                        ]
                        batch_reg[y, x] = adjusted_tlrb
                    
                        
            
        image = np.transpose(preprocess_input(image), (2, 0, 1))
        return image, batch_hm, batch_reg, batch_reg_mask
    
    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self._ttfnet_gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)

        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap
    
    def _ttfnet_gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h
    
    def bbox_areas_log_np(self,bbox):
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        area = (y_max - y_min + 1) * (x_max - x_min + 1)
        return np.log(area) 


    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image_path, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        extension = os.path.splitext(image_path)[1]
        #line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(image_path)
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        if os.path.isfile(image_path.replace(extension,".xml")):
            annotation_line = image_path.replace(extension,".xml")
            box = extract_coordinates(annotation_line,self.classes)
        else:
            box = []
        #box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                box = np.array(box)
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            box = np.array(box)
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box


# DataLoader中collate_fn使用
def centernet_dataset_collate(batch):
    imgs, batch_hms, batch_regs, batch_reg_masks = [], [], [], []

    for img, batch_hm, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        #batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs            = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    batch_hms       = torch.from_numpy(np.array(batch_hms)).type(torch.FloatTensor)
    #batch_whs       = torch.from_numpy(np.array(batch_whs)).type(torch.FloatTensor)
    batch_regs      = torch.from_numpy(np.array(batch_regs)).type(torch.FloatTensor)
    batch_reg_masks = torch.from_numpy(np.array(batch_reg_masks)).type(torch.FloatTensor)
    return imgs, batch_hms, batch_regs, batch_reg_masks



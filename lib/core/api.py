from lib.core.model.centernet import CenterNet

import numpy as np
import math
import cv2
import torch
from train_config import config as cfg

class Detector:
    def __init__(self,model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.model=CenterNet(inference=True)
        print(model_path)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)
    def __call__(self, image, score_threshold=0.5,input_shape=(cfg.DATA.hin,cfg.DATA.win),max_boxes=1000):
        """Detect faces.
        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 5].
        """


        if input_shape is None:
            h, w, c = image.shape
            input_shape = (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)

        else:
            h, w = input_shape
            input_shape = (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)

        image, scale_x, scale_y, dx, dy = self.preprocess(image,
                                                                 target_height=input_shape[0],
                                                                 target_width=input_shape[1])


        if cfg.DATA.channel==1:
            image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            image= np.expand_dims(image, -1)

        image_fornet = np.expand_dims(image, 0)
        image_fornet = np.transpose(image_fornet,axes=[0,3,1,2])

        image_fornet=torch.from_numpy(image_fornet).float().to(self.device)

        with torch.no_grad():
            output=self.model(image_fornet)

        outputs=output.detach().cpu().numpy()

        bboxes=outputs[0]



        bboxes = self.py_nms(np.array(bboxes), iou_thres=None, score_thres=score_threshold,max_boxes=max_boxes)

        ###recorver to raw image
        boxes_scaler = np.array([1 / scale_x,
                                 1  / scale_y,
                                 1 / scale_x,
                                 1  / scale_y,
                                 1.,1.], dtype='float32')

        boxes_bias = np.array([dx ,
                               dy ,
                               dx ,
                               dy , 0.,0.], dtype='float32')
        bboxes = (bboxes - boxes_bias)*boxes_scaler




        # self.stats_graph(self._sess.graph)
        return bboxes


    def preprocess(self, image, target_height, target_width, label=None):

        ###sometimes use in objs detects
        h, w, c = image.shape

        bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype)

        scale_y = target_height / h
        scale_x = target_width / w

        scale = min(scale_x, scale_y)

        image = cv2.resize(image, None, fx=scale, fy=scale)

        h_, w_, _ = image.shape

        dx = (target_width - w_) // 2
        dy = (target_height - h_) // 2
        bimage[dy:h_ + dy, dx:w_ + dx, :] = image

        return bimage, scale, scale, dx, dy

    def py_nms(self, bboxes, iou_thres, score_thres, max_boxes=1000):

        upper_thres = np.where(bboxes[:, 4] > score_thres)[0]

        bboxes = bboxes[upper_thres]
        if iou_thres is None:
            return bboxes

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        order = np.argsort(bboxes[:, 4])[::-1]

        keep=[]
        while order.shape[0] > 0:
            if len(keep)>max_boxes:
                break
            cur = order[0]

            keep.append(cur)

            area = (bboxes[cur, 2] - bboxes[cur, 0]) * (bboxes[cur, 3] - bboxes[cur, 1])

            x1_reain = x1[order[1:]]
            y1_reain = y1[order[1:]]
            x2_reain = x2[order[1:]]
            y2_reain = y2[order[1:]]

            xx1 = np.maximum(bboxes[cur, 0], x1_reain)
            yy1 = np.maximum(bboxes[cur, 1], y1_reain)
            xx2 = np.minimum(bboxes[cur, 2], x2_reain)
            yy2 = np.minimum(bboxes[cur, 3], y2_reain)

            intersection = np.maximum(0, yy2 - yy1) * np.maximum(0, xx2 - xx1)

            iou = intersection / (area + (y2_reain - y1_reain) * (x2_reain - x1_reain) - intersection)

            ##keep the low iou
            low_iou_position = np.where(iou < iou_thres)[0]

            order = order[low_iou_position + 1]

        return bboxes[keep]

import ncnn
import numpy as np
from glob import glob
import cv2
from time import time 

#pnnx best_epoch_weights_mbv2_shufflenet_fe_traced.pth inputshape=[1,3,320,320]

input_width = 320
input_height = 320
MODEL_SCALE = 4

def preprocess_image(img_path, target_size=(320, 320)):
    # Read the image using OpenCV
    if type(img_path) == str:
        img = cv2.imread(img_path)
    else:
        img = img_path
    if img is None:
        raise ValueError(f"Image not found at path: {img_path}")

    # Get original image dimensions
    h, w, c = img.shape

    # Convert the image from BGR to RGB
    img_rgb = img.copy()
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to ncnn Mat and resize
    ncnn_img = ncnn.Mat.from_pixels_resize(
        img_rgb.tobytes(),                # Image data in bytes
        ncnn.Mat.PixelType.PIXEL_RGB,     # Specify that the image is in RGB format
        w,                                # Original width
        h,                                # Original height
        target_size[0],                   # Target width
        target_size[1]                    # Target height
    )

    # Define mean and standard deviation values
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]  # Multiply by 255 to match image scale
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]   # Multiply by 255 to match image scale

    # Calculate normalization values for ncnn
    mean_vals = mean  # Mean values in RGB order
    norm_vals = [1 / s for s in std]  # Normalization values

    # Apply mean subtraction and normalization
    ncnn_img.substract_mean_normalize(mean_vals, norm_vals)

    return ncnn_img
        
def resize_and_pad(image, target_size=(512, 512)):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate the scaling factor
    scale = min(target_width / original_width, target_height / original_height)
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image224
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Pad the image to the target size
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded_image, scale, left, top

def sigmoid(z):
    return 1/(1 + np.exp(-z))


# functions for plotting results
def showbox(img, hm, offset, regr,cls, thresh=0.9):
    boxes, _ = pred2box(hm, offset, regr, thresh=thresh)
    print(boxes)
    sample = img

    for box in boxes:
        center = [int(box[0]), int(box[1])]
        #cos_angle = np.cos(box[4])
        #sin_angle = np.sin(box[4])
        #rot = np.array([[cos_angle, sin_angle], [-sin_angle, cos_angle]])
        reg_w = int(box[2])
        reg_h = int(box[3])
        print(center)

        xmin = center[0] - (reg_w / 2)
        ymin = center[1] - (reg_h / 2)
        xmax = center[0] + (reg_w / 2)
        ymax = center[1] + (reg_h / 2)

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        cv2.rectangle(sample,(xmin,ymin),(xmax,ymax),(255,255,255),2)
        cv2.putText(
	  sample,
	  text = f"{cls}",
	  org = (xmin,ymin),
	  fontFace = cv2.FONT_HERSHEY_DUPLEX,
	  fontScale = 0.8,
	  color = (0, 0, 255),
	  thickness = 1
	)
    return sample


def pred2box(hm, offset, regr, thresh=0.5):
    # make binding box from heatmaps
    # thresh: threshold for logits.

    # get center
    pred = hm > thresh
    pred_center = np.where(hm>thresh)

    # get regressions
    pred_r = regr[:,pred].T
    #pred_angles = cos_sin_hm[:, pred].T

    #print("pred_angle", pred_angle)

    # wrap as boxes
    # [xmin, ymin, width, height]
    # size as original image.
    boxes = []
    scores = hm[pred]

    pred_center = np.asarray(pred_center).T

    for (center, b) in zip(pred_center, pred_r):
        #print(b)
        offset_xy = offset[:, center[0], center[1]]
        #angle = np.arctan2(pred_angle[1], pred_angle[0])
        arr = np.array([(center[1]+offset_xy[0])*MODEL_SCALE, (center[0]+offset_xy[1])*MODEL_SCALE,
                        b[0]*MODEL_SCALE, b[1]*MODEL_SCALE])
        # Clip values between 0 and input_size
        #arr = np.clip(arr, 0, input_size)
        #print("Pred angle", i, pred_angle[i])
        # filter
        #if arr[0]<0 or arr[1]<0 or arr[0]>input_size or arr[1]>input_size:
            #pass
        boxes.append(arr)
    return np.asarray(boxes), scores

def select(hm, threshold):
    """
    Keep only local maxima (kind of NMS).
    We make sure to have no adjacent detection in the heatmap.
    """

    pred = hm > threshold
    pred_centers = np.argwhere(pred)

    for i, ci in enumerate(pred_centers):
        for j in range(i + 1, len(pred_centers)):
            cj = pred_centers[j]
            if np.linalg.norm(ci - cj) <= 2:
                score_i = hm[ci[0], ci[1]]
                score_j = hm[cj[0], cj[1]]
                if score_i > score_j:
                    hm[cj[0], cj[1]] = 0
                else:
                    hm[ci[0], ci[1]] = 0
    return hm

def pred4corner(hm,thresh=0.99):
    threshold = 0.2  # Adjust this threshold as needed
    _, thresholded_heatmap = cv2.threshold(hm, threshold, 1, cv2.THRESH_BINARY)
    
    # Find contours (connected components) in the thresholded heatmap
    contours, _ = cv2.findContours(thresholded_heatmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    keypoints = []
    for cnt in contours:
        # 2. Refine peak location (using contour center)
        try:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            keypoints.append((cx, cy))
        except:
            continue
    return keypoints


def order_points(pts):
    # Initialize a list of coordinates in the order:
    # top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of the points (x + y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left point has the smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right point has the largest sum

    # Difference of the points (y - x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right point has the smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has the largest difference

    return rect


opt = ncnn.Option()
#opt.use_vulkan_compute = True  # Enable if using Vulkan (GPU)
# opt.use_fp16_packed = True
# opt.use_fp16_storage = True
# opt.use_fp16_arithmetic = True

opt.use_int8_storage = True
opt.use_int8_arithmetic = True

net = ncnn.Net()
net.opt = opt
net.opt.num_threads = 4

#mvit_shufflenetv2_pnnx.ncnn
#centernano-int8
#centernano-opt
# Load the converted NCNN model files
net.load_param("best_epoch_weights_mbv2_shufflenet_fe_traced.ncnn.param")
net.load_model("best_epoch_weights_mbv2_shufflenet_fe_traced.ncnn.bin")


test_folder = glob("/home/rivian/Desktop/Datasets/derpetv5_xml/val_images/*.jpg") + glob("/home/rivian/Desktop/Datasets/derpetv5_xml/val_images/*.png") + glob("/home/rivian/Desktop/Datasets/derpetv5_xml/val_images/*.JPG")
threshold = 0.25

import numpy as np
import cv2

def pool_nms(heat, kernel=3):
    """
    Perform non-maximum suppression using a pooling-like approach.
    Works on input tensors of shape (batch, channels, height, width).
    """
    batch, channels, height, width = heat.shape
    hmax = np.zeros_like(heat, dtype=np.float32)
    kernel_matrix = np.ones((kernel, kernel), dtype=np.float32)  # Structuring element

    for b in range(batch):
        for c in range(channels):
            hmax[b, c] = cv2.dilate(heat[b, c], kernel_matrix, iterations=1)

    keep = (hmax == heat).astype(np.float32)
    return heat * keep


def decode_bbox_iou(pred_hms, pred_whs, pred_offsets, confidence=0.3):
    pred_hms = pool_nms(pred_hms)

    b, c, output_h, output_w = pred_hms.shape
    detects = []

    for batch in range(b):
        heat_map = pred_hms[batch].transpose(1, 2, 0).reshape(-1, c)
        pred_wh = pred_whs[batch].transpose(1, 2, 0).reshape(-1, 2)
        pred_offset = pred_offsets[batch].transpose(1, 2, 0).reshape(-1, 2)

        xv, yv = np.meshgrid(np.arange(output_w), np.arange(output_h))
        xv, yv = xv.flatten(), yv.flatten()

        class_conf = np.max(heat_map, axis=-1)
        class_pred = np.argmax(heat_map, axis=-1)
        mask = class_conf > confidence

        pred_wh_mask = pred_wh[mask]
        pred_offset_mask = pred_offset[mask]
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue

        xv_mask = xv[mask].reshape(-1, 1) + pred_offset_mask[:, 0:1]
        yv_mask = yv[mask].reshape(-1, 1) + pred_offset_mask[:, 1:2]

        half_w, half_h = pred_wh_mask[:, 0:1] / 2, pred_wh_mask[:, 1:2] / 2
        bboxes = np.hstack([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h])
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h

        detect = np.hstack([bboxes, class_conf[mask].reshape(-1, 1), class_pred[mask].reshape(-1, 1)])
        detects.append(detect)

    return detects

def centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    box_yx = box_xy[:, ::-1]
    box_hw = box_wh[:, ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2.0 / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.0)
    box_maxes = box_yx + (box_hw / 2.0)
    boxes = np.concatenate([box_mins[:, 0:1], box_mins[:, 1:2], box_maxes[:, 0:1], box_maxes[:, 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def postprocess(prediction, need_nms, image_shape, input_shape, letterbox_image, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]

    for i, detections in enumerate(prediction):
        if len(detections) == 0:
            continue

        unique_labels = np.unique(detections[:, -1])

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]

            if need_nms:
                indices = cv2.dnn.NMSBoxes(
                    detections_class[:, :4].tolist(),
                    detections_class[:, 4].tolist(),
                    confidence_threshold=0.0,
                    nms_threshold=nms_thres
                )
                max_detections = detections_class[indices[:, 0]] if len(indices) > 0 else []
            else:
                max_detections = detections_class

            if len(max_detections) > 0:
                if output[i] is None:
                    output[i] = max_detections
                else:
                    output[i] = np.vstack((output[i], max_detections))

        if output[i] is not None:
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

    return output

a = True #this is better, very close implementation of pytorch 

if not a:
    for i in test_folder:
        image = cv2.imread(i)
        img = image.copy()
        image = cv2.resize(image,(input_width,input_height))
        mat_in = preprocess_image(i)
        mat_vis = np.array(mat_in).transpose(1,2,0)
        # Proceed with inference
        f1 = time()
        ex = net.create_extractor()
        ex.input("in0", mat_in)
        ret, hm = ex.extract("out0")
        ret1, wh = ex.extract("out1")
        ret2, offset = ex.extract("out2")
        f2 = time()
        vis_mat = np.array(hm,dtype=np.float32())
        #vis_mat = cv2.resize(vis_mat,(input_width,input_height))
        
        offset = np.array(offset)
        wh = np.array(wh)
        hm_all = vis_mat
        for hm in hm_all:
            pred = hm > threshold
            pred_centers = np.argwhere(pred)
            for i, ci in enumerate(pred_centers):
                for j in range(i + 1, len(pred_centers)):
                    cj = pred_centers[j]
                    if np.linalg.norm(ci - cj) <= 1:
                        score_i = hm[ci[0], ci[1]]
                        score_j = hm[cj[0], cj[1]]
                        if score_i > score_j:
                            hm[cj[0], cj[1]] = 0
                        else:
                            hm[ci[0], ci[1]] = 0
            pred = hm > threshold
            pred_center = np.where(hm>threshold)

            # get regressions
            pred_r = wh[:,pred].T

            boxes = []
            scores = hm[pred]

            pred_center = np.asarray(pred_center).T

            for (center, b) in zip(pred_center, pred_r):
                offset_xy = offset[:, center[0], center[1]]
                arr = np.array([(center[1]+offset_xy[0])*MODEL_SCALE, (center[0]+offset_xy[1])*MODEL_SCALE,
                                b[0]*MODEL_SCALE, b[1]*MODEL_SCALE])
                boxes.append(arr)
            for box in boxes:
                center = [int(box[0]), int(box[1])]
                reg_w = int(box[2])
                reg_h = int(box[3])

                xmin = center[0] - (reg_w / 2)
                ymin = center[1] - (reg_h / 2)
                xmax = center[0] + (reg_w / 2)
                ymax = center[1] + (reg_h / 2)

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)


                xmin = np.max((0,xmin))
                ymin = np.max((0,ymin))
                xmax = np.min((xmax,img.shape[1]))
                ymax = np.min((ymax,img.shape[0]))


                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),3)

                xmin = xmin * img.shape[1] // input_width
                ymin = ymin * img.shape[0] // input_height
                xmax = xmax * img.shape[1] // input_width
                ymax = ymax * img.shape[0] // input_height

                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),3)
            
        
        
        


        
        
        
        print("FPS:",1/(f2-f1))
        #cv2.imshow("output",image)
        cv2.imshow("output_im",img)
        cv2.imshow("mat_vis",hm)
        #cv2.imshow("vis mat",vis_mat)
        ch = cv2.waitKey(0)
        if ch == ord("q"): break

else:
    for i in test_folder:
        image = cv2.imread(i)
        img = image.copy()
        image_shape = np.array(np.shape(image)[0:2])
        image = cv2.resize(image,(input_width,input_height))
        mat_in = preprocess_image(i)
        mat_vis = np.array(mat_in).transpose(1,2,0)
        # Proceed with inference
        f1 = time()
        ex = net.create_extractor()
        ex.input("in0", mat_in)
        ret, hm = ex.extract("out0")
        ret1, wh = ex.extract("out1")
        ret2, offset = ex.extract("out2")
        f2 = time()
        vis_mat = np.array(hm,dtype=np.float32())
        #vis_mat = cv2.resize(vis_mat,(input_width,input_height))
        
        offset = np.array(offset)[None]
        wh = np.array(wh)[None]
        hm = vis_mat[None]
        outputs = decode_bbox_iou(hm,wh,offset,confidence=0.2)
        results = postprocess(outputs,False,image_shape,(input_height,input_width), False, 0.3) #letterbox true

        #fp2 = time.time()
        #print(f"Postprocessing took: {fp2-fp1} ms")
        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]
        
        for (conf,label,box) in zip(top_conf,top_label,top_boxes):
            ymin = box[0]
            xmin = box[1] 
            ymax = box[2] 
            xmax = box[3] 

            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            class_label = label
            #name = classes[class_label]
            #box_annos.append([xmin,ymin,xmax,ymax,str(name),conf])
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),3)

        print("FPS:",1/(f2-f1))
        cv2.imshow("output_im",img)
        #cv2.imshow("vis mat",vis_mat)
        ch = cv2.waitKey(0)
        if ch == ord("q"): break
    


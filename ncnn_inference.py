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
        
def resize_and_pad(image, target_size=(320, 320)):
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

opt = ncnn.Option()
#opt.use_vulkan_compute = True  # Enable if using Vulkan (GPU)
#opt.use_fp16_packed = True
#opt.use_fp16_storage = True
#opt.use_fp16_arithmetic = True

opt.use_int8_storage = True
opt.use_int8_arithmetic = True

net = ncnn.Net()
net.opt = opt
net.opt.num_threads = 4

#mvit_shufflenetv2_pnnx.ncnn
#centernano-int8
#centernano-opt
# Load the converted NCNN model files
#best_epoch_weights_mbv2_shufflenet_fe_traced.ncnn.param
#best_epoch_weights_mbv2_shufflenet_fe_traced.ncnn.bin

net.load_param("best_epoch_weights_mbv2_shufflenet_fe_traced.ncnn-int8.param")
net.load_model("best_epoch_weights_mbv2_shufflenet_fe_traced.ncnn-int8.bin")

test_folder = glob("/home/rivian/Desktop/Datasets/derpetv5_xml/val_images/*.jpg") + glob("/home/rivian/Desktop/Datasets/derpetv5_xml/val_images/*.png") + glob("/home/rivian/Desktop/Datasets/derpetv5_xml/val_images/*.JPG")
threshold = 0.25

video = False 

f = open("classes.txt","r").readlines()
classes = []
for i in f:
    classes.append(i.strip('\n'))

print(classes)

if not video:

    for i in test_folder:
        image = cv2.imread(i)
        img = image.copy()
        image_shape = np.array(np.shape(image)[0:2])
        image = cv2.resize(image,(input_width,input_height))
        mat_in = preprocess_image(image)
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
            name = classes[class_label]
            #box_annos.append([xmin,ymin,xmax,ymax,str(name),conf])
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),3)
            cv2.putText(img,str(name),(xmin-3,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
            cv2.putText(img,str(conf),(xmax-3,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
        
        print("FPS:",1/(f2-f1))
        cv2.imshow("output_im",img)
        #cv2.imshow("vis mat",vis_mat)
        ch = cv2.waitKey(0)
        if ch == ord("q"): break

else:
    cap = cv2.VideoCapture(0)
    while 1:
        ret,image = cap.read()
        img = image.copy()
        image_shape = np.array(np.shape(image)[0:2])
        image = image[...,::-1]
        image = cv2.resize(image,(input_width,input_height))
        mat_in = preprocess_image(image)
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
        try:
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
                name = classes[class_label]
                #box_annos.append([xmin,ymin,xmax,ymax,str(name),conf])
                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),3)
                cv2.putText(img,str(name),(xmin-3,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
                cv2.putText(img,str(conf),(xmax-3,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
        except:
            pass
        
        print("FPS:",1/(f2-f1))
        cv2.imshow("output_im",img)
        #cv2.imshow("vis mat",vis_mat)
        ch = cv2.waitKey(1)
        if ch == ord("q"): break
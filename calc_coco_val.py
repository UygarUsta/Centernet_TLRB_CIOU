import torch 
import numpy as np
import json 
from dataloader import preprocess_input,resize_image,cvtColor,resize_numpy
from utils_bbox import decode_bbox,postprocess,centernet_correct_boxes_xyxy
import os 
from PIL import Image 
import cv2 
from tqdm import tqdm 
def calculate_eval(model,cocoGt,classes,folder):
    #<class_name> <confidence> <left> <top> <right> <bottom>
    #files = glob(folder + "val_images/*.jpg") + glob(folder + "val_images/*.png")
    input_shape = (512,512)
    coco_format = []
    for i in tqdm(cocoGt.dataset["images"]):     
        id_ = i["id"]
        img_ = os.path.join(folder,"val_images",i["file_name"])
        image =  Image.open(img_) 
        image_shape = np.array(np.shape(image)[0:2])
        image  = cvtColor(image)
        image_data = resize_numpy(image,tuple(input_shape),letterbox_image=False)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        image = np.array(image)
    
        try:
            with torch.no_grad():
                images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor).cuda()
                hm,reg = model(images)

            outputs = decode_bbox(hm,reg,stride = 4,confidence=0.05)
            results = centernet_correct_boxes_xyxy(outputs,input_shape, image_shape, False).cpu()

            for det in results:
                xmin = int(det[0])
                ymin = int(det[1])
                xmax = int(det[2])
                ymax = int(det[3])
                conf = float(det[4])
                label = int(det[5])
                class_label = label
                #cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)
                name = classes[class_label]
                anno = {"image_id": id_, "category_id": int(class_label)+1, "bbox": [xmin, ymin, xmax-xmin, ymax-ymin], "score": float(conf)}
                coco_format.append(anno)

                
        except Exception as e:
            print("Exception during cocoeval",e)
            pass
            #print(f"Could not infer an error occured: {e}")
        #cv2.imwrite("img.jpg",image)
            
    with open('detection_results.json', 'w') as file:
        json.dump(coco_format, file)



def gt_check(img,hm,reg):
    input_shape = (512,512)
    img_copy = np.transpose(np.array(img,dtype=np.float32()),(1,2,0))
    # image =  Image.open(img_) 
    image_shape = np.array(np.shape(img_copy)[0:2])
    print(image_shape)
    # image  = cvtColor(image)  
    # image_data = resize_image(image,input_shape,letterbox_image=True)
    # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    hm = torch.tensor(hm[None]).float().permute(0,3,1,2)
    reg = torch.tensor(reg[None]).float().permute(0,3,1,2)
    #offset = torch.tensor(offset[None]).float().permute(0,3,1,2)

    outputs = decode_bbox(hm,reg,stride = 4, confidence=0.05,cuda=False)
    results = centernet_correct_boxes_xyxy(outputs,input_shape, image_shape, True).cpu()
    #results = postprocess(outputs,True,image_shape,input_shape, False, 0.3) 
    try:
        for det in results:
            xmin = int(det[0])
            ymin = int(det[1])
            xmax = int(det[2])
            ymax = int(det[3])
            conf = float(det[4])
            label = int(det[5])
            class_label = label
            cv2.rectangle(img_copy,(xmin,ymin),(xmax,ymax),(0,255,0),2)
            cv2.putText(img_copy,f"{label} {conf}",(xmin,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    except Exception as e:
        print("Exception is :",e)
    return img_copy

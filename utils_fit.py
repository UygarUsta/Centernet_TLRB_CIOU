import os
import random
import numpy as np 
import torch
from loss import focal_loss, reg_l1_loss,ciou_loss,iou_aware_loss
from tqdm import tqdm
from loss import get_lr
from calc_coco_val import calculate_eval
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def decode_offsets_to_boxes(pred_offsets, stride=4):
    """
    pred_offsets: shape [B, H, W, 4]
                  storing [left, top, right, bottom] per pixel
    stride      : how many pixels in the input space per 1 step in the feature map

    Returns:
        decoded_boxes: shape [B, H, W, 4], with absolute corners in [x_min, y_min, x_max, y_max]
    """
    B, H, W, _ = pred_offsets.shape

    # 1) Create a grid of center coords in the original input space
    #    For pixel (y, x) in feature map, the "center" in input space is (x * stride, y * stride).
    #    We'll flatten them, then reshape back.
    device = pred_offsets.device
    yv, xv = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32)
    )
    xv = xv * stride  # shape (H, W)
    yv = yv * stride

    # 2) Flatten
    xv = xv.view(1, -1)  # shape (1, H*W)
    yv = yv.view(1, -1)

    # 3) Flatten the offsets
    offsets_flat = pred_offsets.view(B, -1, 4)  # (B, H*W, 4)
    # L, T, R, B
    left_offset   = offsets_flat[..., 0]
    top_offset    = offsets_flat[..., 1]
    right_offset  = offsets_flat[..., 2]
    bottom_offset = offsets_flat[..., 3]

    # 4) For each pixel i,
    #    x_min = center_x - (left_offset_i * stride)
    #    x_max = center_x + (right_offset_i * stride)
    #    y_min = center_y - (top_offset_i  * stride)
    #    y_max = center_y + (bottom_offset_i * stride)
    x_min = xv - left_offset  * stride
    x_max = xv + right_offset * stride
    y_min = yv - top_offset   * stride
    y_max = yv + bottom_offset* stride

    # 5) Combine into [x_min, y_min, x_max, y_max]
    decoded = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # (B, H*W, 4)

    # 6) Reshape back to (B, H, W, 4)
    decoded = decoded.view(B, H, W, 4)
    return decoded

def fit_one_epoch(model_train, model,optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, cocoGt,classes,folder,best_mean_AP,local_rank=0):
    total_r_loss    = 0
    total_c_loss    = 0
    total_loss      = 0
    val_loss        = 0
    #total_iou_aware_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
        batch_images, batch_hms, batch_regs, batch_reg_masks = batch

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            hm, pred_reg  = model_train(batch_images)
            c_loss          = focal_loss(hm, batch_hms)
            # wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            # off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
            
            loss            = c_loss + wh_loss + off_loss

            total_loss      += loss.item()
            total_c_loss    += c_loss.item()
            total_r_loss    += wh_loss.item() + off_loss.item()

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                hm, pred_reg  = model_train(batch_images)
                c_loss          = focal_loss(hm, batch_hms)
                #wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                #off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                pred_reg = pred_reg.permute(0, 2, 3, 1).contiguous()

                pred_boxes = decode_offsets_to_boxes(pred_reg, stride=4)
                gt_boxes = decode_offsets_to_boxes(batch_regs, stride=4) 
                

                # Compute CIoU loss
                loss_ciou = ciou_loss(pred_boxes, gt_boxes, batch_reg_masks)


                # with torch.no_grad():
                #     inter_x1 = torch.max(pred_bboxes[:,0], gt_bboxes[:,0])
                #     inter_y1 = torch.max(pred_bboxes[:,1], gt_bboxes[:,1])
                #     inter_x2 = torch.min(pred_bboxes[:,2], gt_bboxes[:,2])
                #     inter_y2 = torch.min(pred_bboxes[:,3], gt_bboxes[:,3])

                #     inter_w = (inter_x2 - inter_x1).clamp(min=0)
                #     inter_h = (inter_y2 - inter_y1).clamp(min=0)
                #     inter_area = inter_w * inter_h

                #     area_pred = (pred_bboxes[:,2]-pred_bboxes[:,0]).clamp(min=0)*(pred_bboxes[:,3]-pred_bboxes[:,1]).clamp(min=0)
                #     area_gt = (gt_bboxes[:,2]-gt_bboxes[:,0]).clamp(min=0)*(gt_bboxes[:,3]-gt_bboxes[:,1]).clamp(min=0)
                #     union = area_pred + area_gt - inter_area + 1e-6
                #     actual_iou = inter_area / union  # shape: [N]

                # # iou_pred shape: [B, H, W], flatten -> [B*H*W]
                # iou_pred_flat = iou_pred.view(-1)
                # loss_iou_aware = iou_aware_loss(iou_pred_flat, actual_iou, mask)




                
                loss            = c_loss   + loss_ciou * 5   # +wh_loss

                total_loss      += loss.item()
                total_c_loss    += c_loss.item()
                total_r_loss    += loss_ciou.item() #+ off_loss.item() + loss_iou_aware.item() * 0.5  #wh_loss.item() + off_loss.item()
                #total_iou_aware_loss += loss_iou_aware.item()

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_r_loss'  : total_r_loss / (iteration + 1), 
                                'total_c_loss'  : total_c_loss / (iteration + 1),
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
            
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
            batch_images, batch_hms, batch_regs, batch_reg_masks = batch

            hm, pred_reg  = model_train(batch_images)
            c_loss          = focal_loss(hm, batch_hms)
            #wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            #off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
            pred_reg = pred_reg.permute(0, 2, 3, 1).contiguous()
            pred_boxes = decode_offsets_to_boxes(pred_reg, stride=4)
            gt_boxes = decode_offsets_to_boxes(batch_regs, stride=4) 
            

            # Compute CIoU loss
            loss_ciou = ciou_loss(pred_boxes, gt_boxes, batch_reg_masks)

            
            # inter_x1 = torch.max(pred_bboxes[:,0], gt_bboxes[:,0])
            # inter_y1 = torch.max(pred_bboxes[:,1], gt_bboxes[:,1])
            # inter_x2 = torch.min(pred_bboxes[:,2], gt_bboxes[:,2])
            # inter_y2 = torch.min(pred_bboxes[:,3], gt_bboxes[:,3])

            # inter_w = (inter_x2 - inter_x1).clamp(min=0)
            # inter_h = (inter_y2 - inter_y1).clamp(min=0)
            # inter_area = inter_w * inter_h

            # area_pred = (pred_bboxes[:,2]-pred_bboxes[:,0]).clamp(min=0)*(pred_bboxes[:,3]-pred_bboxes[:,1]).clamp(min=0)
            # area_gt = (gt_bboxes[:,2]-gt_bboxes[:,0]).clamp(min=0)*(gt_bboxes[:,3]-gt_bboxes[:,1]).clamp(min=0)
            # union = area_pred + area_gt - inter_area + 1e-6
            # actual_iou = inter_area / union  # shape: [N]

            # # iou_pred shape: [B, H, W], flatten -> [B*H*W]
            # iou_pred_flat = iou_pred.view(-1)
            # loss_iou_aware = iou_aware_loss(iou_pred_flat, actual_iou, mask)

            loss            = c_loss   + loss_ciou * 5  #+ loss_iou_aware * 0.5 # +wh_loss

            val_loss        += loss.item()


            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(".", 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
        
            calculate_eval(model,cocoGt,classes,folder)
            try:
                cocoDt = cocoGt.loadRes("detection_results.json")
                cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()

                mean_ap = cocoEval.stats[0]  # This is the mAP at IoU thresholds from .50 to .95
                mean_ap_05 = cocoEval.stats[1]
                mean_ap_075 = cocoEval.stats[2] 
            except:
                mean_ap,mean_ap_05,mean_ap_075 = 0.0,0.0,0.0        

            print(f"Mean Average Precision (mAP) across IoU thresholds [0.50, 0.95]: {mean_ap:.3f}")
            if mean_ap > best_mean_AP:
                print('Save best model to best_epoch_weights.pth')
                torch.save(model.state_dict(), os.path.join(".", "best_epoch_weights.pth"))
        else:
            mean_ap,mean_ap_05,mean_ap_075 = 0.0,0.0,0.0

            
        torch.save(model.state_dict(), os.path.join(".", "last_epoch_weights.pth"))
    return mean_ap


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
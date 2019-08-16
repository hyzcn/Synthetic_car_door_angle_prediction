import os
import numpy as np
import cv2
from vis_dis import *

# Configurations
part_name = 'fl'
data_dir = "datasets/shapenet_car_data/"
data_seg_dir = "datasets/shapenet_car_seg/"
mask_dir = "seg_dict/shapenet_train_seg.npy"
save_dir = "gt_dict/shapenet_car_gt.npy".format(part_name)

seg_dir = "datasets/shapenet_vis_dis/"
seg_dict_dir = "seg_dict/vis_dis_fl_seg.npy"
vis_dir = "vis_dis/vis_dis_fl.npy"

def get_locat(mask):
    left = mask.shape[1]
    right = 0
    top = None
    bottom = None
    for i in range(mask.shape[0]):
        search = np.argwhere(mask[i]==1)
        if len(search)!=0 and top==None:
            top = i
        if len(search)==0 and top!=None and bottom==None:
            bottom = i-1
        if len(search)!=0 and top!=None and i==mask.shape[0]-1 and bottom==None:
            bottom = i
        if len(search)!=0:
            left = min(left, search[0][0])
            right = max(right, search[-1][0])
            
    # print(left, right, top, bottom)
    if top!=None and bottom!=None:
        return (left+right)//2, (top+bottom)//2
    else:
        return None, None

def create_gt(data_dir, data_seg_dir, mask_dir, seg_dir, seg_dict_dir, vis_dir, save_dir):
    seg_mask_dict = read_seg_dict(data_seg_dir, mask_dir)
    vis_dict = read_vis_dict(seg_dir, seg_dict_dir, vis_dir)
    gt_dict = {}
    for file in os.listdir(data_dir):
        type, fl, fr, bl, br, trunk, az, el, dist = file.split('_')
        _, _, bin = vis_dict[az+el]
        x, y = get_locat(seg_mask_dict[file])
        gt_dict[file[:-4]] = [bin, fl, x, y]
    np.save(save_dir, gt_dict)
    return gt_dict

if __name__ == "__main__":
    gt_dict = create_gt(data_dir, data_seg_dir, mask_dir, seg_dir, seg_dict_dir, vis_dir, save_dir)
    print(gtdict)

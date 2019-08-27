import os
import numpy as np
import cv2
from vis_dis import *
from tqdm import tqdm

# Configurations
part_name = 'fl'
data_dir = "../datasets/shapenet_test_fl/"
data_seg_dir = "../datasets/shapenet_test_fl_seg/"
mask_dir = "../seg_dict/shapenet_test_fl_seg.npy"
save_dir = "../gt_dict/shapenet_test_fl_gt.npy".format(part_name)
# Fixed settings
seg_dir = "../datasets/preset_vis_dis/"
seg_dict_dir = "../seg_dict/vis_dis_fl_seg.npy"
vis_dir = "../vis_dis/vis_dis_fl.npy"


def get_locat(mask):
    height = len(mask)
    width = len(mask[0])
    left = mask.shape[1]
    right = 0
    top = None
    bottom = None
    for i in range(mask.shape[0]):
        search = np.argwhere(mask[i]==1)
        if len(search)!=0 and top==None:
            top = i
        if len(search)!=0:
            bottom = i
            left = min(left, search[0][0])
            right = max(right, search[-1][0])
            
    # print(left, right, top, bottom)
    if top!=None and bottom!=None:
        return (left+right)//2/width, (top+bottom)//2/height
    else:
        return None, None

def create_gt(data_dir, data_seg_dir, mask_dir, seg_dir, seg_dict_dir, vis_dir, save_dir):
    print("Reading seg_mask dictionary...")
    seg_mask_dict = read_seg_dict(data_seg_dir, mask_dir)
    print("Reading vis_dis dictionary...")
    vis_dict = read_vis_dict(seg_dir, seg_dict_dir, vis_dir)
    gt_dict = {}
    print("Start generating ground truth...")
    for file in tqdm(os.listdir(data_dir)):
        type, fl, fr, bl, br, trunk, az, el, dist = file.split('_')
        _, _, bin = vis_dict[az+'_'+el]
        x, y = get_locat(seg_mask_dict[file])
        gt_dict[file[:-4]] = [bin, x, y]
    np.save(save_dir, gt_dict)
    return gt_dict

if __name__ == "__main__":
    gt_dict = create_gt(data_dir, data_seg_dir, mask_dir, seg_dir, seg_dict_dir, vis_dir, save_dir)
    # gt_dict = np.load(save_dir).item()
    # print(gt_dict)

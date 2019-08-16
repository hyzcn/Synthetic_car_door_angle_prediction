import cv2
import numpy as np
import os
import re
from tqdm import tqdm
from seg_dict_save import *

# # Testing settings
# seg_dir = "../datasets/test_seg/"
# seg_dict_dir = "../seg_dict/test_seg.npy"
# vis_save_dir = "../vis_dis/test_vis.npy"


# Configurations
part_name = "fl"
seg_dir = "../datasets/shapenet_vis_dis/"
seg_dict_dir = "../seg_dict/vis_dis_fl_seg.npy"
vis_save_dir = "../vis_dis/vis_dis_fl.npy"

vis_thresh = 4000
vis_max = 60
vis_ratio = 0.5


def bin_vis_dis(vis_dict):
    print("Generating visual discriminability...")
    for view, vis in tqdm(vis_dict.items()):
        if vis[1] - vis[0] > vis_max * vis_ratio:
            vis_dict[view].append(True)
        else:
            vis_dict[view].append(False)

def create_vis_dis(seg_mask_dict, vis_save_dir):
    vis_dict = {}
    print("Calculating visible range...")
    for name, mask in tqdm(seg_mask_dict.items()):
        type, fl, fr, bl, br, trunk, az, el, dist, _ = re.split(r'[_.]', name)
        view = "{}_{}".format(az, el)
        area = sum(sum(mask))
        degree = abs(int(fl))
        # print(view, degree, area)
        if view not in vis_dict:
            vis_dict[view] = [0, vis_max]
        if area < 4000: # invisible
            if degree > vis_dict[view][0]:
                vis_dict[view] = [degree, vis_max]

    bin_vis_dis(vis_dict)
    np.save(vis_save_dir, vis_dict)
    return vis_dict

def read_vis_dict(seg_dir, seg_dict_dir, vis_dir):
    if not os.path.isfile(vis_dir):
        seg_mask_dict = read_seg_dict(seg_dir, seg_dict_dir)
        vis_dict = create_vis_dis(seg_mask_dict, vis_dir)
    else:
        vis_dict = np.load(vis_dir).item()
    return vis_dict

def main():
    print("Reading seg_mask_dict...")
    seg_mask_dict = read_seg_dict(seg_dir, seg_dict_dir)
    vis_dict = create_vis_dis(seg_mask_dict, vis_save_dir)
    print(vis_dict)

if __name__ == "__main__":
    main()

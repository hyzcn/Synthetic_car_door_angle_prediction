import cv2
import numpy as np
import os
import re
from tqdm import tqdm
import csv
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

def cal_negative_ratio(vis_dir):
    vis_dict = np.load(vis_dir).item()
    num = 0
    name_list = []
    for name, vis in vis_dict.items():
        if vis[2] == False:
            num += 1
            name_list.append(name)
            print(name, vis)
    return num/len(vis_dict), name_list

def visual_dis(vis_dir):
    vis_dict = np.load(vis_dir).item()
    ans = []
    el_title = [" "]
    for i, az in enumerate(range(0, 361, 10)):
        ans.append([])
        ans[i].append(az)
        for j, el in enumerate(range(0, 91, 10)):
            if i == 0:
                el_title.append(el)
            view = "{}_{}".format(az,el)
            ans[i].append(vis_dict[view][2])

    with open("viewpoint_dis.csv","w") as csvfile:
        over_file = csv.writer(csvfile)
        over_file.writerow(el_title)
        for i, az in enumerate(range(0, 361, 10)):
            over_file.writerow(ans[i])
    print("viewpoint info saved!")


def main():
    print("Reading seg_mask_dict...")
    seg_mask_dict = read_seg_dict(seg_dir, seg_dict_dir)
    vis_dict = create_vis_dis(seg_mask_dict, vis_save_dir)
    print(vis_dict)

if __name__ == "__main__":
    # main()
    # neg_ratio, name_listq = cal_negative_ratio(vis_save_dir)
    # print(neg_ratio)
    visual_dis(vis_save_dir)

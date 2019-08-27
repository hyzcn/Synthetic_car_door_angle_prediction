import os
import numpy as np
import cv2
from tqdm import tqdm
import re
from random_sample import sample_data

# Test Settings
# data_dir = '../datasets/test/'
# seg_dir = '../datasets/test_seg/'
# img_save_dir = "../datasets/test_crop/"
# seg_save_dir = "../datasets/test_seg_crop/"

part_name = 'fl'
sample_iter = 800
data_dir = '../datasets/preset_car_data/'
seg_dir = '../datasets/preset_car_seg/'
img_save_dir = "../datasets/preset_car_crop/"
seg_save_dir = "../datasets/preset_car_seg_crop/"


color_dict = [[0,0,0],[0,0,128],[0,128,0],[0,128,128],[128,0,0],[128,0,128],[128,128,0]]

def crop_mask(img):
    crop = np.zeros([img.shape[0], img.shape[1]], np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if list(img[i][j]) in color_dict:
                crop[i][j] = 1
            else:
                crop[i][j] = 0

    return crop

def get_crop_loc(mask):
    MARGIN = 10
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

    return max(0, top-10), min(height-1, bottom+10), max(0, left-10), min(width-1, right+10)

def gen_crop():
    name_data = []
    print("Start sampling...")
    for i in tqdm(range(sample_iter)):
        fl_spl, fr_spl, bl_spl, br_spl, trunk_spl, az_spl, el_spl, dist_spl = sample_data()
        for file in os.listdir(data_dir):
            if file[-3:] == "png":
                type, fl, fr, bl, br, trunk, az, el, dist, _ = re.split(r'[_.]', file)
                if bl == bl_spl and fr == fr_spl and br == br_spl and trunk == trunk_spl and az == az_spl and el == el_spl and dist == dist_spl:
                    name_data.append(file[:-4])
    print("{} images sampled.".format(len(name_data)))
    for file in tqdm(name_data):
        img = cv2.imread(data_dir+file+".png")
        seg = cv2.imread(seg_dir+file+".png")
        mask = crop_mask(seg)
        top, bottom, left, right = get_crop_loc(mask)
        crop_img = img[top:bottom,left:right,:]
        crop_seg = seg[top:bottom,left:right,:]
        cv2.imwrite(img_save_dir+file+".png", crop_img)
        cv2.imwrite(seg_save_dir+file+".png", crop_seg)

if __name__=="__main__":
    gen_crop()

            
    
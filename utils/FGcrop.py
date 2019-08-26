import os
import numpy as np
import cv2
from tqdm import tqdm

# Test Settings
data_dir = '../datasets/test/'
seg_dir = '../datasets/test_seg/'
img_save_dir = "../datasets/test_crop/"
seg_save_dir = "../datasets/test_seg_crop/"

part_name = 'fl'
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

    return top-10, bottom+10, left-10, right+10

def gen_crop():
    for file in tqdm(os.listdir(data_dir)):
        img = cv2.imread(data_dir+file)
        seg = cv2.imread(seg_dir+file)
        mask = crop_mask(seg)
        top, bottom, left, right = get_crop_loc(mask)
        crop_img = img[top:bottom,left:right,:]
        crop_seg = seg[top:bottom,left:right,:]
        cv2.imwrite(img_save_dir+file, crop_img)
        cv2.imwrite(seg_save_dir+file, crop_seg)

if __name__=="__main__":
    gen_crop()

            
    
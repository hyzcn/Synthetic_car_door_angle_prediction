import os
import numpy as np
import cv2
from tqdm import tqdm
import re
import random
from random_sample import sample_data

# Test Settings
# data_dir = '../datasets/test/'
# seg_dir = '../datasets/test_seg/'
# img_save_dir = "../datasets/test_crop/"
# seg_save_dir = "../datasets/test_seg_crop/"

sample_iter = 10000
data_dir = '../datasets/train/preset_car_data/'
seg_dir = '../datasets/train/preset_car_seg/'
img_save_dir = "../datasets/train/preset_car_crop/"
seg_save_dir = "../datasets/train/preset_car_crop_seg/"




# def crop_mask(img):
#     color_dict = [[0,0,0],[0,0,128],[0,128,0],[0,128,128],[128,0,0],[128,0,128],[128,128,0]]
#     crop = np.zeros([img.shape[0], img.shape[1]], np.uint8)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if list(img[i][j]) in color_dict:
#                 crop[i][j] = 1
#             else:
#                 crop[i][j] = 0

#     return crop

def get_crop_loc(seg_img):
    color = [[0,0,0],[0,0,128],[0,128,0],[0,128,128],[128,0,0],[128,0,128],[128,128,0]]

    height = len(seg_img)
    width = len(seg_img[0])

    mask = ((seg_img[:,:,0] == color[0][2]) & (seg_img[:,:,1] == color[0][1]) & (seg_img[:,:,2] == color[0][0])
            |(seg_img[:,:,0] == color[1][2]) & (seg_img[:,:,1] == color[1][1]) & (seg_img[:,:,2] == color[1][0])
            |(seg_img[:,:,0] == color[2][2]) & (seg_img[:,:,1] == color[2][1]) & (seg_img[:,:,2] == color[2][0])
            |(seg_img[:,:,0] == color[3][2]) & (seg_img[:,:,1] == color[3][1]) & (seg_img[:,:,2] == color[3][0])
            |(seg_img[:,:,0] == color[4][2]) & (seg_img[:,:,1] == color[4][1]) & (seg_img[:,:,2] == color[4][0])
            |(seg_img[:,:,0] == color[5][2]) & (seg_img[:,:,1] == color[5][1]) & (seg_img[:,:,2] == color[5][0])
            |(seg_img[:,:,0] == color[6][2]) & (seg_img[:,:,1] == color[6][1]) & (seg_img[:,:,2] == color[6][0]))

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
    train_params = {
        # "mesh_id":['Hatchback', 'Hybrid', 'Sedan2Door', 'Sedan4Door', 'Suv'],
        "mesh_id":['suv', 'hybrid', 'hatchback', 'sedan2door', 'sedan4door'],
        "fl":[x for x in range(-40, 1, 20)],
        "fr":[x for x in range(0, 41, 20)],
        "bl":[x for x in range(-40, 1, 20)],
        "br":[x for x in range(0, 41, 20)],
        "trunk":[x for x in range(0, 41, 20)],
        "az":[x for x in range(0, 361, 40)],
        "el":[x for x in range(20, 81, 20)],
        "dist":[400, 450],
    }
    name_data = []
    print("Start sampling...")
    for i in tqdm(range(sample_iter)):
        ins = random.sample(train_params['mesh_id'], 1)[0]
        fl, fr, bl, br, trunk, az, el, dist = sample_data()
        train_img = '{ins}_{fl}_{fr}_{bl}_{br}_{trunk}_{az}_{el}_{dist}'.format(**locals())
        name_data.append(train_img)
    print("{} images sampled.".format(len(name_data)))
    for file in tqdm(name_data):
        img = cv2.imread(data_dir+file+".png")
        file = file.replace("Hatchback", "hatchback").replace("Hybrid", "hybrid").replace("Sedan2Door", "sedan2door").replace("Sedan4Door", "sedan4door").replace("Suv", "suv")
        seg = cv2.imread(seg_dir+file+".png")
        top, bottom, left, right = get_crop_loc(seg)
        crop_img = img[top:bottom,left:right,:]
        # crop_seg = seg[top:bottom,left:right,:]
        cv2.imwrite(img_save_dir+file+".png", crop_img)
        # cv2.imwrite(seg_save_dir+file+".png", crop_seg)

if __name__=="__main__":
    gen_crop()

            
    
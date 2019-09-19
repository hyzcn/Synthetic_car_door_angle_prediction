import os
import numpy as np
import cv2
from vis_dis import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configurations
seg_dir = '../datasets/train/preset_car_seg/'
save_dir = '../segsets/train/preset_car_seg/'

def get_locat_np(seg_img):
    color_dict = {'sky':[245, 10, 255], 'floor':[0, 255, 127], 'fl':[0,0,0], 
                'fr':[0,0,128], 'bl':[0,128,0], 'br':[0,128,128],
                'hood':[128,0,0], 'trunk':[128,0,128], 'body':[128,128,0] }

    height = len(seg_img)
    width = len(seg_img[0])
    gtmap = np.array([[0 for x in range(width)] for y in range(height)])

    i = 0
    for key, color in color_dict.items():
        mask = ((seg_img[:,:,0] == color[2]) & (seg_img[:,:,1] == color[1]) & (seg_img[:,:,2] == color[0]))
        x, y = np.where(mask == True)
        gtmap[x, y] = i
        i += 1
    
    # cv2.imwrite('../segsets/test/a.png', gtmap)
    return gtmap




def convert(seg_dir, save_dir):
    for file in tqdm(os.listdir(seg_dir)):
        if not os.path.isfile(save_dir+file[:-4]+'.npy'):
            seg_img = cv2.imread(seg_dir+file)
            gtmap = get_locat_np(seg_img)
            np.save(save_dir+file[:-4]+'.npy', gtmap)

if __name__=='__main__':
    convert(seg_dir, save_dir)
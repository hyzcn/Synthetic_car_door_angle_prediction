import os
import numpy as np
import cv2
from tqdm import tqdm

# Test Settings
# seg_dir = '../datasets/test_seg/'
# save_dir = "../seg_dict/test_seg.npy"

part_name = 'fl'
seg_dir = '../datasets/shapenet_car_seg/'.format(part_name)
save_dir = "../seg_dict/shapenet_train_seg.npy".format(part_name)


color_dict = {'fl':[0,0,0], 'fr':[0,0,128], 'bl':[0,128,0], 'br':[0,128,128],
                'hood':[128,0,0], 'trunk':[128,0,128], 'body':[128,128,0]}

def seg_part(img, color):
    seg = np.zeros([img.shape[0], img.shape[1]], np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][2] == color[0] and img[i][j][1] == color[1] and img[i][j][0] == color[2]:
                seg[i][j] = 0
            else:
                seg[i][j] = 255

    # dilate the target part
    kernel = np.ones((5,5), np.uint8)
    seg_eron = cv2.erode(seg, kernel, iterations=5)
    seg_eron[seg_eron==0] = 1
    seg_eron[seg_eron==255] = 0
    return seg_eron

def read_seg(part_name, path):
    color = color_dict[part_name]
    seg_mask_dict = {}
    for file in tqdm(os.listdir(path)):
        if file[-3:] == "png":
            # print('----seg:{}'.format(file))
            img = cv2.imread(path+file)
            seg_mask = seg_part(img, color)
            seg_mask_dict[file] = seg_mask
    return seg_mask_dict

def save_dict(part_name, seg_dir, save_dir):
    print("Start loading dictionary...")
    seg_dict = read_seg(part_name, seg_dir)
    np.save(save_dir, seg_dict)

def read_seg_dict(part_name, seg_dir, dict_path):
    if not os.path.isfile(dict_path):
        save_dict(part_name, seg_dir, dict_path)
    seg_mask_dict = np.load(dict_path).item()
    return seg_mask_dict

if __name__=='__main__':
    save_dict(part_name, seg_dir, save_dir)
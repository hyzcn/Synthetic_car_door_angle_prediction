import cv2
import numpy as np
from tqdm import tqdm

seg_dir = "../seg_dict/vis_dis_fr_seg.npy"
save_dir = "../seg_vis/"
def show_seg(seg_dir, save_dir):
    print("Reading seg dictionary...")
    seg_dict = np.load(seg_dir).item()
    for name, map in tqdm(seg_dict.items()):
        map[map==1] = 255
        cv2.imwrite(save_dir+name, map)

if __name__=="__main__":
    show_seg(seg_dir, save_dir)



import os
import numpy as np
import cv2
from vis_dis import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configurations
part_name = 'all'
data_dir = "../datasets/all_test/preset_all_random/"
data_seg_dir = "../datasets/all_test/preset_all_random_seg/"
save_dir = "../gt_dict/preset_all_random_{}_gt.npy".format(part_name)
# Fixed settings
if part_name == "fl":
    mask_dir = "../seg_dict/preset_car_crop_seg.npy"
    seg_dir = "../datasets/vis_dis/preset_vis_dis_{}/".format(part_name)
    seg_dict_dir = "../seg_dict/vis_dis_fl_seg.npy"
    vis_dir = "../vis_dis/vis_dis_fl.npy"
else:
    fl_seg_dir = "../datasets/vis_dis/preset_vis_dis_fl/"
    fl_seg_dict_dir = "../seg_dict/vis_dis_fl_seg.npy"
    fl_vis_dir = "../vis_dis/vis_dis_fl.npy"
    fr_seg_dir = "../datasets/vis_dis/preset_vis_dis_fr/"
    fr_seg_dict_dir = "../seg_dict/vis_dis_fr_seg.npy"
    fr_vis_dir = "../vis_dis/vis_dis_fr.npy"
    bl_seg_dir = "../datasets/vis_dis/preset_vis_dis_bl/"
    bl_seg_dict_dir = "../seg_dict/vis_dis_bl_seg.npy"
    bl_vis_dir = "../vis_dis/vis_dis_bl.npy"
    br_seg_dir = "../datasets/vis_dis/preset_vis_dis_br/"
    br_seg_dict_dir = "../seg_dict/vis_dis_br_seg.npy"
    br_vis_dir = "../vis_dis/vis_dis_br.npy"
    trunk_seg_dir = "../datasets/vis_dis/preset_vis_dis_trunk/"
    trunk_seg_dict_dir = "../seg_dict/vis_dis_trunk_seg.npy"
    trunk_vis_dir = "../vis_dis/vis_dis_trunk.npy"

def get_locat_np(seg_dir, part_name):
    color_dict = {'fl':[0,0,0], 'fr':[0,0,128], 'bl':[0,128,0], 'br':[0,128,128],
                'hood':[128,0,0], 'trunk':[128,0,128], 'body':[128,128,0]}

    seg_img = cv2.imread(seg_dir)
    height = len(seg_img)
    width = len(seg_img[0])

    color = color_dict[part_name]

    mask = ((seg_img[:,:,0] == color[2]) & (seg_img[:,:,1] == color[1]) & (seg_img[:,:,2] == color[0]))
    # mask_filename = './' + seg_dir.split('/')[-1] + '_' + part_name + '.png'

    # print(mask_filename)
    # imageio.imwrite(mask_filename, mask)
    # plt.imshow(mask)
    # plt.savefig(mask_filename)

    y, x = np.where(mask == True)
    # bb = [x.min(), x.max(), y.min(), y.max()]
    count = mask.sum()
    if len(y) == 0 or len(x) == 0 or count < 100:
        return 0, 0

    # print(seg_dir, x.min(), x.max(), y.min(), y.max())
    return (x.min() + x.max()) // 2 / width, (y.min() + y.max()) // 2 / height

def get_locat(seg_dir, part_name):
    return get_locat_np(seg_dir, part_name)

def create_gt_single(data_dir, data_seg_dir, mask_dir, seg_dir, seg_dict_dir, vis_dir, save_dir):
    print("Reading seg_mask dictionary...")
    seg_mask_dict = read_seg_dict(part_name, data_seg_dir, mask_dir)
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

def create_gt_all():
    print("Reading vis_dis dictionary...")
    print("--Reading fl vis_dis dict...")
    fl_vis_dict = read_vis_dict(fl_seg_dir, fl_seg_dict_dir, fl_vis_dir)
    print("--Reading fr vis_dis dict...")
    fr_vis_dict = read_vis_dict(fr_seg_dir, fr_seg_dict_dir, fr_vis_dir)
    print("--Reading bl vis_dis dict...")
    bl_vis_dict = read_vis_dict(bl_seg_dir, bl_seg_dict_dir, bl_vis_dir)
    print("--Reading br vis_dis dict...")
    br_vis_dict = read_vis_dict(br_seg_dir, br_seg_dict_dir, br_vis_dir)
    print("--Reading trunk vis_dis dict...")
    trunk_vis_dict = read_vis_dict(trunk_seg_dir, trunk_seg_dict_dir, trunk_vis_dir)
    gt_dict = {}
    print("Start generating ground truth...")
    for file in tqdm(os.listdir(data_dir)):
        type, fl, fr, bl, br, trunk, az, el, dist = file.split('_')
        _, _, fl_bin = fl_vis_dict[az+'_'+el]
        fl_x, fl_y = get_locat(data_seg_dir+file, "fl")
        _, _, fr_bin = fr_vis_dict[az+'_'+el]
        fr_x, fr_y = get_locat(data_seg_dir+file, "fr")
        _, _, bl_bin = bl_vis_dict[az+'_'+el]
        bl_x, bl_y = get_locat(data_seg_dir+file, "bl")
        _, _, br_bin = br_vis_dict[az+'_'+el]
        br_x, br_y = get_locat(data_seg_dir+file, "br")
        _, _, trunk_bin = trunk_vis_dict[az+'_'+el]
        trunk_x, trunk_y = get_locat(data_seg_dir+file, "trunk")

        vis = False
        if vis:
            seg_im = plt.imread(os.path.join(data_seg_dir, file))
            plt.figure()
            plt.imshow(seg_im)
            vis_filename = file + '_vis.png'
            print(vis_filename)
            xs = [fl_x, fr_x, bl_x, br_x, trunk_x]
            xs = [v * 640 for v in xs]
            ys = [fl_y, fr_y, bl_y, br_y, trunk_y]
            ys = [v * 480 for v in ys]
            print(xs)
            print(ys)
            plt.plot(xs, ys, 'r*', markersize=5)
            plt.savefig(vis_filename)

        gt_dict[file[:-4]] = [fl_bin, fl_x, fl_y, fr_bin, fr_x, fr_y, bl_bin, bl_x, bl_y, br_bin, br_x, br_y, trunk_bin, trunk_x, trunk_y]
    np.save(save_dir, gt_dict)
    return gt_dict

if __name__ == "__main__":
    if part_name != "all":
        gt_dict = create_gt_single(data_dir, data_seg_dir, mask_dir, seg_dir, seg_dict_dir, vis_dir, save_dir)
    else:
        gt_dict = create_gt_all()
    # gt_dict = np.load(save_dir).item()
    # print(gt_dict)

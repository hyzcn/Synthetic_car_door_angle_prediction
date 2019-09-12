import os
import numpy as np
import glob
from tqdm import tqdm
import random
from random_sample import get_samples, sample_data


# texture settings
# train_params = {
#     "mesh_id":['Hatchback', 'Hybrid', 'Sedan2Door', 'Sedan4Door', 'Suv'],
#     "fl":[x for x in range(0, 41, 20)],
#     "fr":[x for x in range(0, 41, 20)],
#     "bl":[x for x in range(0, 41, 20)],
#     "br":[x for x in range(0, 41, 20)],
#     "trunk":[x for x in range(0, 41, 20)],
#     "az":[x for x in range(0, 361, 40)],
#     "el":[x for x in range(20, 81, 20)],
#     "dist":[400, 450],
# }
# spatial settings
train_params = {
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

def sample_split_generator(part_name, sample_iter, test_num, train_save_dir, test_save_dir):
    train_name = []
    test_name = []
    for i in tqdm(range(sample_iter)):
        if part_name == "fl":
            sample_names = get_samples(train_params, 0)
            train_name += sample_names
        elif part_name == "all":
            for i in range(5):
                sample_names = get_samples(train_params, i)
                train_name += sample_names
    
    i = 0
    while True:
        if i >= test_num:
            break
        ins = random.sample(train_params['mesh_id'], 1)[0]
        fl, fr, bl, br, trunk, az, el, dist = sample_data()
        test_img = '{ins}_{fl}_{fr}_{bl}_{br}_{trunk}_{az}_{el}_{dist}'.format(**locals())
        if test_img not in train_name:
            test_name.append(test_img)
            i += 1

    train_txt = open(train_save_dir, 'w')
    test_txt = open(test_save_dir, 'w')
    for v in train_name:
        train_txt.write(v+'\n')
    for v in test_name:
        test_txt.write(v+'\n')

    print("train set: {} images".format(len(train_name)))
    print("test set: {} images".format(len(test_name)))

def norm_split_generator(part_name, train_num, test_num, train_save_dir, test_save_dir):
    train_name = []
    test_name = []
    i = 0
    while True:
        if i >= train_num:
            break
        ins = random.sample(train_params['mesh_id'], 1)[0]
        fl, fr, bl, br, trunk, az, el, dist = sample_data()
        train_img = '{ins}_{fl}_{fr}_{bl}_{br}_{trunk}_{az}_{el}_{dist}'.format(**locals())
        train_name.append(train_img)
        i += 1
        
    i = 0
    while True:
        if i >= test_num:
            break
        ins = random.sample(train_params['mesh_id'], 1)[0]
        fl, fr, bl, br, trunk, az, el, dist = sample_data()
        test_img = '{ins}_{fl}_{fr}_{bl}_{br}_{trunk}_{az}_{el}_{dist}'.format(**locals())
        if test_img not in train_name:
            test_name.append(test_img)
            i += 1

    train_txt = open(train_save_dir, 'w')
    test_txt = open(test_save_dir, 'w')
    for v in train_name:
        train_txt.write(v+'\n')
    for v in test_name:
        test_txt.write(v+'\n')

    print("train set: {} images".format(len(train_name)))
    print("test set: {} images".format(len(test_name)))


if __name__ == '__main__':
    # train_save_dir = "../ImageSets/preset_texture_all_train.txt"
    # test_save_dir = "../ImageSets/preset_texture_all_test.txt"
    train_save_dir = "../ImageSets/preset_all_train_norm.txt"
    test_save_dir = "../ImageSets/preset_all_test_norm.txt"
    # sample_split_generator('all', 600, 9720, train_save_dir, test_save_dir)
    norm_split_generator('all', 45000, 9720, train_save_dir, test_save_dir)


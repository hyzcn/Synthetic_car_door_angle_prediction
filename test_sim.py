from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import scipy.io as scio
from PIL import Image 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import random
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from model.model import set_parameter_requires_grad, initialize_model
from model.metric import eveluate_acc
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Dataset settings
data_dir = 'datasets/all_test/shapenet_test_all/*.png'

# Model settings
model_name = 'resnet'
# model_dir = 'params/resnet_ft_all_floor_norm_new.pkl'
model_dir = 'params/loc_pre_final/resnet_ft_all_0.3_0.6_64.pkl'
feature_extract = False

# Number of classes in the dataset
num_classes = 20

# Batch size for training (change depending on how much memory you have)
batch_size = 64

thres = 12

gt_dir = 'gt_dict/shapenet_test_all_all_gt.npy'

def output_test(file, names, labels, result, outputs):
    for i in range(len(names)):
        # visualize txt
        # baseline
        # content = "name:{}  gt:{}  prediction:{}  raws:[{}, {}, {}, {}]\n".format(names[i], labels[i], \
        #             result[i], str(int(round(outputs[i][0]*-60))), str(int(round(outputs[i][1]*60))), \
        #             str(int(round(outputs[i][2]*-60))), str(int(round(outputs[i][3]*60))))
        ## pre_loc
        # content = "name:{}  gt:{}  prediction:{}  raws:[{:.3f}, {}, {:.3f}, {}, {:.3f}, {}, {:.3f}, {}]\n".format(names[i], labels[i], \
        #             result[i], outputs[i][0], str(int(round(outputs[i][1]*-60))), outputs[i][4], \
        #             str(int(round(outputs[i][5]*60))), outputs[i][8], str(int(round(outputs[i][9]*-60))), \
        #             outputs[i][12], str(int(round(outputs[i][13]*60))))
        content = "name:{}  gt:{}  prediction:{}  raws:[{}, {}, {}, {}]\n".format(names[i], labels[i], \
                    result[i], str(int(round(outputs[i][1]*-60))) if outputs[i][0] > 0.5 else 0, \
                    str(int(round(outputs[i][5]*60))) if outputs[i][4] > 0.5 else 0, \
                    str(int(round(outputs[i][9]*-60))) if outputs[i][8] > 0.5 else 0, \
                    str(int(round(outputs[i][13]*60))) if outputs[i][12] > 0.5 else 0)
        
        file.write(content)

def output_aggregate(outputs):
    thre = 0.2
    # baseline
    # x = np.where((outputs[:,0] > thre) | (outputs[:,1] > thre) | (outputs[:,2] > thre) | (outputs[:,3] > thre) == True)
    # pre_loc
    x = np.where((outputs[:,0]>0.5)&(outputs[:,1] > thre) | (outputs[:,4]>0.5)&(outputs[:,5] > thre) \
                | (outputs[:,8]>0.5)&(outputs[:,9] > thre) | (outputs[:,12]>0.5)&(outputs[:,13] > thre) == True)

    results = np.array([False for y in range(outputs.shape[0])])
    results[x] = True

    return torch.FloatTensor(results)


def test_model(model, dataloaders):
    since = time.time()
    file = open("./test_out.txt",'w')
    
    TP = [0,0]
    TN = [0,0]
    FP = [0,0]
    FN = [0,0]
    with torch.set_grad_enabled(False):
        for names, inputs, labels in tqdm(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            model.eval()
            
            outputs = model(inputs)
            aggrs = output_aggregate(outputs.cpu())
            aggrs = aggrs.to(device)
            tps, tns, fps, fns = eveluate_acc(labels.cpu().numpy(), aggrs.cpu().numpy(), 2)

            TP += tps
            TN += tns
            FP += fps
            FN += fns
            
            output_test(file, names, labels.cpu().detach().numpy(), aggrs.cpu().detach().numpy(), outputs.cpu().detach().numpy())

    file.close()
    
    return TP, TN, FP, FN
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataSource, gtSource, mode="test"):
        # Just normalization for validation
        self.names = glob.glob(dataSource)
        self.gts = np.load(gtSource).item()
        self.names.sort()
        self.cal_num()

    def __getitem__(self, index):
        return self.names[index], self.load_image(self.names[index]), self.load_gt(self.names[index].split('/')[-1])
        
    def __len__(self):
        return len(self.names)

    def cal_num(self):
        opened_num = 0
        closed_num = 0
        for name in self.names:
            ins, fl, fr, bl, br, trunk, az, el, dist = name.split("/")[-1].split('_')
            fl_bin, fl_x, fl_y, fr_bin, fr_x, fr_y, bl_bin, bl_x, bl_y, br_bin, br_x, br_y, trunk_bin, trunk_x, trunk_y =  self.gts[name.split("/")[-1].lower()[:-4]]
            # Open: car door exist and degree > thres
            if bool(fl_x) and bool(fl_y) and abs(int(fl)) > thres or bool(fr_x) and bool(fr_y) and abs(int(fr)) > thres \
                or bool(bl_x) and bool(bl_y) and abs(int(bl)) > thres or bool(br_x) and bool(br_y) and abs(int(br)) > thres:
                opened_num += 1
            else:
                closed_num += 1

        print(opened_num, closed_num)
    
    def load_image(self, dir):
        data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        img = cv2.imread(dir)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        return data_transforms(img)
    
    def load_gt(self, name):
        ins, fl, fr, bl, br, trunk, az, el, dist = name.split('_')
        fl_bin, fl_x, fl_y, fr_bin, fr_x, fr_y, bl_bin, bl_x, bl_y, br_bin, br_x, br_y, trunk_bin, trunk_x, trunk_y =  self.gts[name.lower()[:-4]]
        # Open: car door exist and degree > thres
        if bool(fl_x) and bool(fl_y) and abs(int(fl)) > thres or bool(fr_x) and bool(fr_y) and abs(int(fr)) > thres \
            or bool(bl_x) and bool(bl_y) and abs(int(bl)) > thres or bool(br_x) and bool(br_y) and abs(int(br)) > thres:
            return 1
        else:
            return 0


# Detect if we have a GPU available
device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
model_ft.load_state_dict(torch.load(model_dir))
model_ft = nn.DataParallel(model_ft)
if isinstance(model_ft,torch.nn.DataParallel):
        model_ft = model_ft.module
model_ft.to(device)

model_ft.eval()

# Build testset
# Setup train test split
testsets = myDataset(data_dir, gt_dir)
testloader_dict = Data.DataLoader(testsets, batch_size=batch_size, shuffle=False, num_workers=4)
tp, tn, fp, fn = test_model(model_ft, testloader_dict)
print("acc: ", (tp+tn)/(tp+fp+fn+tn))
print("prec: ", tp/(tp+fp))
print("rec: ", tp/(tp+fn))
print("tp: ", tp)
print("tn: ", tn)
print("fp: ", fp)
print("fn: ", fn)
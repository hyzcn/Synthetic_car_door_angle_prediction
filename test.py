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
from sklearn.metrics import accuracy_score
from model import set_parameter_requires_grad, initialize_model
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Dataset settings
data_dir = 'datasets/others/door_val_0309/*/*.png'

# Model settings
model_name = 'resnet'
model_dir = 'params/loc_pre_final/resnet_ft_all_0.3_0.6_64.pkl'
feature_extract = False

# Number of classes in the dataset
num_classes = 20

# Batch size for training (change depending on how much memory you have)
batch_size = 64

def output_test(file, names, labels, result, outputs):
    for i in range(len(names)):
        # visualize txt
        content = "name:{}  gt:{}  prediction:{}  raws:[{}, {}, {}, {}]\n".format(names[i], labels[i], result[i], str(int(round(outputs[i][0]*-40))), \
        str(int(round(outputs[i][1]*40))), str(int(round(outputs[i][2]*-40))), str(int(round(outputs[i][3]*40))))
        
        file.write(content)

def output_aggregate(outputs):
    thre = 0.3
    x = np.where((outputs[:,1] > thre) | (outputs[:,5] > thre) | (outputs[:,9] > thre) | (outputs[:,13] > thre) == True)
    results = np.array([False for y in range(outputs.shape[0])])
    results[x] = True

    return torch.FloatTensor(results)


def test_model(model, dataloaders):
    since = time.time()
    file = open("./test_out.txt",'w')
    
    running_loss = 0
    with torch.set_grad_enabled(False):
        for names, inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            model.eval()
            
            outputs = model(inputs)
            aggrs = output_aggregate(outputs.cpu())
            aggrs = aggrs.to(device)
            print(outputs[:,1::4])
            print(aggrs)
            print(labels)
            loss = accuracy_score(aggrs.cpu(), labels.cpu())
            
            running_loss += loss * inputs.size(0)
            
            # output_test(file, names, labels.cpu().detach().numpy(), aggrs.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        
    loss = running_loss / len(dataloaders.dataset)
    file.close()
    
    return loss
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataSource, mode="test"):
        # Just normalization for validation
        data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.names = glob.glob(dataSource)


    def __getitem__(self, index):
        return self.names[index], self.load_image(self.names[index]), self.load_gt(self.names[index])
        
    def __len__(self):
        return len(self.names)
    
    def load_image(self, dir):
        data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        img = cv2.imread(dir)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        return data_transforms(img)
    
    def load_gt(self, name):
        label = name.split('/')[-2]
        if label == 'Opened':
            return True
        else:
            return False

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
testsets = myDataset(data_dir)
testloader_dict = Data.DataLoader(testsets, batch_size=batch_size, shuffle=True, num_workers=4)
test_loss = test_model(model_ft, testloader_dict)
print("test mae: ", test_loss)
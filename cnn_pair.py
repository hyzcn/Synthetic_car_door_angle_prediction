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
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from model import *
import re
from utils.random_sample import *
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

os.environ['QT_QPA_PLATFORM']='offscreen'

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir = "./data/hymenoptera_data"

# Train/Test mode
command = "test"

# Add crop images to train set
add_crop = False

# Dataset settings
num_images = 97200
sample_iter = 600
test_ratio = 0.1

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
part_name = "all"

# Number of classes in the dataset
num_classes = 5

# Batch size for training (change depending on how much memory you have)
batch_size = 64

# Number of epochs to train for
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# Data range
data_range = 60

# Dir settings
train_dir = 'datasets/train/preset_car_data/'
if add_crop == False:
    crop_dir = None
else:
    crop_dir = 'datasets/train/preset_car_crop/'
test_dir = 'datasets/all_test/preset_all_random/'.format(part_name)
model_dir = 'params/sigmoid/{}_ft_{}.pkl'.format(model_name, part_name)
plot_dir = 'plots/sigmoid/{}_ft_{}.jpg'.format(model_name, part_name)
output_dir = 'outputs/sigmoid/{}_ft_{}.txt'.format(model_name, part_name)
html_dir = "htmls/sigmoid/{}_ft_{}.txt".format(model_name, part_name)

print("-------------------------------------")
print("Config:\nmodel:{}\nnum_classes:{}\nbatch size:{}\nepochs:{}\nsample set:{}\ntest set:{}\nmodel:{}".format(model_name, num_classes, batch_size, num_epochs, train_dir, test_dir, model_dir))
print("-------------------------------------\n")

x_list = []
train_mse = []
val_mse = []
train_mae = []
val_mae = []

def draw_plot():
    # plot
    plt.subplot(121)
    plt.cla()
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("MSE loss")
    plt.plot(x_list,train_mse,"x-",label="train loss")
    plt.plot(x_list,val_mse,"+-",label="val loss")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.subplot(122)
    plt.cla()
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("MAE loss")
    plt.plot(x_list,train_mae,"x-",label="train loss")
    plt.plot(x_list,val_mae,"+-",label="val loss")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.savefig(plot_dir)

def output_test(file, html, names, result):
    for i in range(len(names)):
        # visualize txt
        type_gt, fl_gt, fr_gt, bl_gt, br_gt, trunk_gt, az_gt, el_gt, dist_gt = names[i].split('_')
        # content  = "gt: [ {} {} {} {} {} {} {}]---predictitmutmuxons: [".format(fl_gt, fr_gt, bl_gt, br_gt, trunk_gt, az_gt, el_gt)
        content  = "name: {}---gt: [ {}]---predictions: [".format(names[i], fl_gt)
        content += ' '+str(int(round(result[i][0]*data_range)))
        content += "]\n"
        file.write(content)
        html.write("{} gt:{} pred:{}\n".format(names[i], fl_gt, str(int(round(result[i][0]*data_range)))))

    
def test_model(model, dataloaders, criterion):
    since = time.time()
    file = open(output_dir,'w')
    html = open(html_dir,'w')
    
    running_loss = 0
    running_dist = 0
    for names, inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        model.eval()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        dist = mean_absolute_error(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
        
        running_loss += loss.item() * inputs.size(0)
        running_dist += dist.item() * inputs.size(0)
        
        output_test(file, html, names, outputs.cpu().detach().numpy())
        
    loss = running_loss / len(dataloaders.dataset)
    dist = running_dist / len(dataloaders.dataset)
    file.close()
    html.close()
    
    return loss, dist
    

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        x_list.append(epoch)
        best_loss = 10000

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_dist = 0.0

            # Iterate over data.
            for i, (name, inputs, labels) in tqdm(enumerate(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    dist = mean_absolute_error(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_dist += dist.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_dist = running_dist / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}, Dist: {:.4f}'.format(phase, epoch_loss, epoch_dist*abs(data_range)))
            
            # plot
            if phase == 'train':
                train_mse.append(epoch_loss)
                train_mae.append(epoch_dist*abs(data_range))
            else:
                val_mse.append(epoch_loss)
                val_mae.append(epoch_dist*abs(data_range))

            # deep copy the model
            if phase == 'val' and (epoch == 0 or epoch_loss<best_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.module.state_dict(), model_dir)

        draw_plot()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataSource, mode, test_id=None, cropSource=None):
        # Just normalization for validation
        self.dir_img = dataSource
        self.dir_crop = cropSource
        self.names = self.load_names(mode, test_id)
        print("{} data loaded: {} images".format(mode, len(self.names)))

    def __getitem__(self, index):
        if self.names[index][-3:] == "png":
            return self.names[index], self.load_image(self.dir_crop+self.names[index]), self.load_gt(self.names[index])
        else:
            return self.names[index], self.load_image(self.dir_img+self.names[index]+".png"), self.load_gt(self.names[index])
        
    def __len__(self):
        return len(self.names)
    
    def load_crops(self, dir):
        name_data = []
        for file in os.listdir(dir):
            if file[-3:] == "png":
                name_data.append(file)
        return name_data

    def load_names(self, mode, test_id=None):
        name_data = []
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
        if mode == 'train':
            print("Start sampling...")
            for i in tqdm(range(sample_iter)):
                if part_name == "fl":
                    sample_names = get_samples(train_params, 0)
                    name_data += sample_names
                elif part_name == "all":
                    for i in range(5):
                        sample_names = get_samples(train_params, i)
                        name_data += sample_names
            if self.dir_crop:
                crop_name = self.load_crops(self.dir_crop)
                name_data += crop_name
        elif mode == 'test':
            for file in os.listdir(self.dir_img):
                if file[-3:] == "png":
                    name_data.append(file[:-4])
        elif mode == 'test_baseline':
            n = 0
            for file in os.listdir(self.dir_img):
                if file[-3:] == "png":
                     if n in test_id:
                        name_data.append(file[:-4])
                n += 1
        
        return name_data

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
        if part_name == "all":
            return torch.FloatTensor([abs(int(fl))/data_range, abs(int(fr))/data_range, abs(int(bl))/data_range, abs(int(br))/data_range, abs(int(trunk))/data_range])
        elif part_name == "fl":
            return torch.FloatTensor([abs(int(fl))/data_range])
        elif part_name == "fr":
            return torch.FloatTensor([abs(int(fr))/data_range])
        elif part_name == "bl":
            return torch.FloatTensor([abs(int(bl))/data_range])
        elif part_name == "br":
            return torch.FloatTensor([abs(int(br))/data_range])
        elif part_name == "trunk":
            return torch.FloatTensor([abs(int(trunk))/data_range])
        else:
            print("part name error in loading gt!!!")

# Detect if we have a GPU available
device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")

# Setup the loss fxn
criterion = nn.MSELoss()

if command == "train":
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft = nn.DataParallel(model_ft)


    # Print the model we just instantiated
    # print(model_ft)

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    trainsets = myDataset(dataSource=train_dir, cropSource=crop_dir, mode='train')
    testsets = myDataset(test_dir, 'test')

    #image_datasets = {'train': myDataset([transform_dataset(X_train, data_transforms), Variable(torch.FloatTensor(y_train))]), 'val': myDataset([transform_dataset(X_val, data_transforms), Variable(torch.FloatTensor(y_val))])}
    image_datasets = {'train': trainsets, 'val': testsets}


    # Create training and validation dataloaders
    dataloaders_dict = {x: Data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                # print("\t",name)
                pass

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Train and evaluate
    model_ft = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# Test
# Load model
if command == "test":
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    model_ft.load_state_dict(torch.load(model_dir))
    model_ft = nn.DataParallel(model_ft)
    if isinstance(model_ft,torch.nn.DataParallel):
            model_ft = model_ft.module
    model_ft.to(device)

    model_ft.eval()

    # testsets = myDataset(test_dir, 'test')
    random_list = range(num_images)
    test_id = random.sample(random_list, 9720)
    testsets = myDataset(train_dir, 'test_baseline', test_id)

# Build testset
testloader_dict = Data.DataLoader(testsets, batch_size=batch_size, shuffle=True, num_workers=4)
test_loss, test_dist = test_model(model_ft, testloader_dict, criterion)
print("test mse: ", test_loss)
print("test mae: ", test_dist*abs(data_range))
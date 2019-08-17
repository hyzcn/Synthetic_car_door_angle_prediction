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
from utils.seg_dict_save import *
from model import *
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir = "./data/hymenoptera_data"

# Train/Test mode
command = "train"

# Dataset settings
num_images = 97200
sample_iter = 30
test_ratio = 0.1

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
part_name = "fl"

# Number of classes in the dataset
num_classes = 1+3

# Batch size for training (change depending on how much memory you have)
batch_size = 30

# Number of epochs to train for
num_epochs = 25

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# Data range
data_range = -60

# Dir settings
## Train
train_dir = 'datasets/shapenet_car_data/'
train_gt_dir = 'gt_dict/shapenet_car_gt.npy'.format(part_name)
## Test
test_dir = 'datasets/shapenet_test_{}/'.format(part_name)
test_gt_dir = 'gt_dict/shapenet_test_{}_gt.npy'.format(part_name)
## Model
model_dir = 'params/location/{}_ft_{}.pkl'.format(model_name, part_name)
plot_dir = 'plots/location/'
output_dir = 'outputs/location/{}_ft_{}_same.txt'.format(model_name, part_name)
html_dir = "htmls/location/{}_ft_{}_same.txt".format(model_name, part_name)


print("-------------------------------------")
print("Config:\nmodel:{}\nnum_classes:{}\nbatch size:{}\nepochs:{}\nsample set:{}\ntest set:{}".format(model_name, num_classes, batch_size, num_epochs, train_dir, test_dir))
print("-------------------------------------\n")

x_list = []
train_total = []
val_total = []
train_bin = []
val_bin = []
train_door = []
val_door = []
train_pos = []
val_pos = []
train_door_mae = []
val_door_mae = []
train_pos_mae = []
val_pos_mae = []

def delete_false(labels, outputs):
    index = []
    for i in range(len(labels)):
        if labels[i][0]:
            index.append(i)
    new_labels = []
    new_outputs = []
    for i in index:
        new_labels.append(labels[i])
        new_outputs.append(outputs[i])
    new_labels = torch.FloatTensor(new_labels).to(device)
    new_outputs = torch.FloatTensor(new_outputs).to(device)

    return new_labels, new_outputs

def output_test(file, html, names, labels, outputs):
    for i in range(len(names)):
        # visualize txt
        type_gt, fl_gt, fr_gt, bl_gt, br_gt, trunk_gt, az_gt, el_gt, dist_gt = names[i].split('_')
        content  = "name: {}---gt: [ {}, {}, {}, {}]---predictions: [".format(names[i], bool(labels[i][0]), int(labels[i][1]*data_range), \
                                                                            int(labels[i][2]*640), int(labels[i][3]*480))
        content += '{:.2f}, {}, {}, {}'.format(outputs[i][0], int(round(outputs[i][1]*data_range)), int(round(outputs[i][2]*640)), int(round(labels[i][3]*480)))
        content += "]\n"
        file.write(content)
        html.write("{} gt:{} pred:{}\n".format(names[i], fl_gt, str(int(round(outputs[i][1]*data_range)))))

    
def test_model(model, dataloaders, criterion):
    since = time.time()
    file = open(output_dir,'w')
    html = open(html_dir,'w')
    
    running_loss = {"total_mse": 0.0, "bin_mse": 0.0, "door_mse":0.0, "pos_mse":0.0, "door_mae": 0.0, "pos_mae": 0.0}
    n_num = 0
    for names, inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        model.eval()
        
        outputs = model(inputs)
        loss_bin = criterion(outputs[:,0], labels[:,0])
        n_labels, n_outputs = delete_false(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        loss_door = criterion(n_outputs[:,1], n_labels[:,1])
        loss_pos = criterion(n_outputs[:,2:], n_labels[:,2:])
        loss = 0.1*loss_bin + 0.4*loss_door + 0.5*loss_pos
        dist_door = mean_absolute_error(n_outputs.cpu().detach().numpy()[:,1], n_labels.cpu().detach().numpy()[:,1])
        dist_pos = 0.5*mean_absolute_error(n_outputs.cpu().detach().numpy()[:,2]*640, n_labels.cpu().detach().numpy()[:,2]*640)+\
                                0.5*mean_absolute_error(n_outputs.cpu().detach().numpy()[:,2]*480, n_labels.cpu().detach().numpy()[:,2]*480)
        
        running_loss["bin_mse"] += loss_bin.item() * inputs.size(0)
        running_loss["door_mse"] += loss_door.item() * n_outputs.size(0)
        running_loss["pos_mse"] += loss_pos.item() * n_outputs.size(0)
        running_loss["total_mse"] += loss.item() * n_outputs.size(0)
        running_loss["door_mae"] += dist_door.item() * n_outputs.size(0)
        running_loss["pos_mae"] += dist_pos.item() * n_outputs.size(0)
        n_num += n_outputs.size(0)
        
        output_test(file, html, names, labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
    
    loss_bin = running_loss["bin_mse"] / len(dataloaders.dataset)
    loss_door = running_loss["door_mse"] / n_num
    loss_pos = running_loss["pos_mse"] / n_num
    loss = running_loss["total_mse"] / n_num
    dist_door = running_loss["door_mae"] / n_num
    dist_pos = running_loss["pos_mae"] / n_num
    file.close()
    html.close()
    
    return loss, loss_door, loss_pos, loss_bin, dist_door, dist_pos
        

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    print("Start training...")

    for epoch in tqdm(range(num_epochs)):
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

            running_loss = {"total_mse": 0.0, "bin_mse": 0.0, "door_mse":0.0, "pos_mse":0.0, "door_mae": 0.0, "pos_mae": 0.0}
            # Iterate over data.
            n_num = 0
            for i, (name, inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss_bin = criterion(outputs[:,0], labels[:,0])
                    n_labels, n_outputs = delete_false(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    loss_door = criterion(n_outputs[:,1], n_labels[:,1])
                    loss_pos = criterion(n_outputs[:,2:], n_labels[:,2:])
                    loss = 0.1*loss_bin + 0.4*loss_door + 0.5*loss_pos
                    dist_door = mean_absolute_error(n_outputs.cpu().detach().numpy()[:,1], n_labels.cpu().detach().numpy()[:,1])
                    dist_pos = 0.5*mean_absolute_error(n_outputs.cpu().detach().numpy()[:,2]*640, n_labels.cpu().detach().numpy()[:,2]*640)+\
                                            0.5*mean_absolute_error(n_outputs.cpu().detach().numpy()[:,2]*480, n_labels.cpu().detach().numpy()[:,2]*480)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss["bin_mse"] += loss_bin.item() * inputs.size(0)
                running_loss["door_mse"] += loss_door.item() * outputs.size(0)
                running_loss["pos_mse"] += loss_pos.item() * outputs.size(0)
                running_loss["total_mse"] += loss.item() * outputs.size(0)
                running_loss["door_mae"] += dist_door.item() * outputs.size(0)
                running_loss["pos_mae"] += dist_pos.item() * outputs.size(0)
                n_num += n_outputs.size(0)

            epoch_loss_bin = running_loss["bin_mse"] / len(dataloaders[phase].dataset)
            epoch_loss_door = running_loss["door_mse"] / n_num
            epoch_loss_pos = running_loss["pos_mse"] / n_num
            epoch_loss = running_loss["total_mse"] / n_num
            epoch_dist_door = running_loss["door_mae"] / n_num
            epoch_dist_pos = running_loss["pos_mae"] / n_num

            print('{} Total loss: {:.4f}, Bin loss: {:.4f}, Door loss: {:.4f}, Position loss: {:.4f}, Door dist: {:.4f}, Position dist: {:.4f}'.format(phase, \
                epoch_loss, epoch_loss_bin, epoch_loss_door, epoch_loss_pos, epoch_dist_door*abs(data_range), epoch_dist_pos))
            
            # plot
            if phase == 'train':
                train_total.append(epoch_loss)
                train_bin.append(epoch_loss_bin)
                train_door.append(epoch_loss_door)
                train_pos.append(epoch_loss_pos)
                train_door_mae.append(epoch_dist_door*60)
                train_pos_mae.append(epoch_dist_pos)
            else:
                val_total.append(epoch_loss)
                val_bin.append(epoch_loss_bin)
                val_door.append(epoch_loss_door)
                val_pos.append(epoch_loss_pos)
                val_door_mae.append(epoch_dist_door*60)
                val_pos_mae.append(epoch_dist_pos)

            # deep copy the model
            if phase == 'val' and (epoch == 0 or epoch_loss<best_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.model.state_dict(), model_dir)
        
        draw_plot()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def sample_data():
    fl = [x for x in range(-40, 1, 20)]
    fr = [x for x in range(0, 60, 20)]
    bl = [x for x in range(-40, 1, 20)]
    br = [x for x in range(0, 60, 20)]
    trunk = [x for x in range(0, 60, 20)] 
    az = [x for x in range(0, 361, 40)]
    el = [x for x in range(20, 90, 20)]
    dist = [400, 450]
    fl_spl = random.sample(fl, 1)
    fr_spl = random.sample(fr, 1)
    bl_spl = random.sample(bl, 1)
    br_spl = random.sample(br, 1)
    trunk_spl = random.sample(trunk, 1)
    return str(fl_spl[0]), str(fr_spl[0]), str(bl_spl[0]), str(br_spl[0]), str(trunk_spl[0])
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataSource, gtSource, mode):
        # Just normalization for validation
        self.dir_img = dataSource
        self.names = self.load_names(dataSource, mode)
        self.gt_dict = np.load(gtSource).item()
        print("{} data loaded: {} images".format(mode, len(self.names)))

    def __getitem__(self, index):
        return self.names[index], self.load_image(self.dir_img+self.names[index]+".png"), self.load_gt(self.names[index])
        
    def __len__(self):
        return len(self.names)
    
    def load_names(self, dir, mode):
        name_data = []
        if mode == 'train':
            print("Start sampling...")
            for i in tqdm(range(sample_iter)):
                fl_spl, fr_spl, bl_spl, br_spl, trunk_spl = sample_data()
                for file in os.listdir(dir):
                    if file[-3:] == "png":
                        ins, fl, fr, bl, br, trunk, az, el, dist = file.split('_')
                        if bl == bl_spl and fr == fr_spl and br == br_spl and trunk == trunk_spl:
                            name_data.append(file[:-4])
        else:
            for file in os.listdir(dir):
                if file[-3:] == "png":
                    name_data.append(file[:-4])
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
        bin, x, y = self.gt_dict[name]
        return torch.FloatTensor([bin, int(fl)/data_range, x if x!= None else -1, y if x!= None else -1])


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
    trainsets = myDataset(train_dir, train_gt_dir, 'train')
    testsets = myDataset(test_dir, test_gt_dir, 'test')

    image_datasets = {'train': trainsets, 'val': testsets}


    # Create training and validation dataloaders
    dataloaders_dict = {x: Data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8) for x in ['train', 'val']}

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    # print("Params to learn:")
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

    testsets = myDataset(test_dir, test_gt_dir, 'test')

# Build testset
testloader_dict = Data.DataLoader(testsets, batch_size=batch_size, shuffle=True, num_workers=8)
test_loss, test_door, test_pos, test_bin, dist_door, dist_pos = test_model(model_ft, testloader_dict, criterion)
print("Total test mse: ", test_loss)
print("Test binary mse: ", test_bin)
print("Test door mse: ", test_door)
print("Test position mse: ", test_pos)
print("Test door mae: ", dist_door*abs(data_range))
print("Test position mae: ", dist_pos)

def draw_plot():
    # plot
    # plt.title('vgg16_bn Feature Extract',fontsize='large',fontweight='bold')
    # plt.title('vgg16_bn Fine-tune',fontsize='large', fontweight='bold')
    # plt.title('ResNet18 Feature Extract',fontsize='large', fontweight='bold')
    # plt.title('ResNet18 Fine-tune',fontsize='middle', fontweight='bold')
    plt.subplot(221)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Total MSE loss",fontsize=10)
    plt.plot(x_list,train_total,"x-",label="train loss")
    plt.plot(x_list,val_total,"+-",label="val loss")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0., fontsize=5)
    plt.subplot(222)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Binary MSE loss",fontsize=10)
    plt.plot(x_list,train_bin,"x-",label="train loss")
    plt.plot(x_list,val_bin,"+-",label="val loss")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0., fontsize=5)
    plt.subplot(223)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Door MSE loss",fontsize=10)
    plt.plot(x_list,train_door,"x-",label="train loss")
    plt.plot(x_list,val_door,"+-",label="val loss")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0., fontsize=5)
    plt.subplot(224)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Position MSE loss",fontsize=10)
    plt.plot(x_list,train_pos,"x-",label="train loss")
    plt.plot(x_list,val_pos,"+-",label="val loss")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0., fontsize=5)

    plt.savefig(plot_dir+"{}_ft_{}_mse.jpg".format(model_name, part_name))
    
    plt.subplot(121)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Door MAE loss",fontsize=10)
    plt.plot(x_list,train_door_mae,"x-",label="train loss")
    plt.plot(x_list,val_door_mae,"+-",label="val loss")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0., fontsize=7)
    plt.subplot(122)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Position MAE loss",fontsize=10)
    plt.plot(x_list,train_pos_mae,"x-",label="train loss")
    plt.plot(x_list,val_pos_mae,"+-",label="val loss")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0., fontsize=7)

    plt.savefig(plot_dir+"{}_ft_{}_mae.jpg".format(model_name, part_name))
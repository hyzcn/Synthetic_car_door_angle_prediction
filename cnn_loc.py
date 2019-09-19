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
from utils.seg_dict_save import read_seg_dict
from model.model import set_parameter_requires_grad, initialize_model
import re
from utils.random_sample import get_samples
from options.train_loc_options import TrainOptions
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

os.environ['QT_QPA_PLATFORM']='offscreen'


def main():
    opt = TrainOptions()
    args = opt.initialize()
    opt.print_options(args)

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = args.model_name
    part_name = args.part_name

    # Number of classes in the dataset
    if part_name == "fl":
        num_factors = 1
    elif part_name == "all":
        num_factors = 5
    if args.add_pre:
        num_classes = (1+3)*num_factors
    else:
        num_classes = (1+2)*num_factors

    # Data range
    data_range = args.data_range

    # Dir settings
    ## Crop
    if args.add_crop == False:
        crop_dir = None
        crop_gt_dir = None
    else:
        crop_dir = args.crop_dir
        crop_gt_dir = args.crop_gt_dir.format(part_name)

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

    # torch.autograd.set_detect_anomaly(True)

    def draw_plot():
        # plot
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

        plt.savefig(args.plot_dir+"{}_ft_{}_mse_0.3_0.6_64.jpg".format(model_name, part_name))
        
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

        plt.savefig(args.plot_dir+"{}_ft_{}_mae_0.3_0.6_64.jpg".format(model_name, part_name))

    def delete_pre_train(labels, outputs):
        x, y = np.where((labels.cpu()[:,::4]==False)==True)
        # x, y = np.where(((labels.cpu()[:,::4]==False) | ((labels.cpu()[:,2::4]==0) & (labels.cpu()[:,3::4]==0)))==True)
        y = y*4+1
        outputs[x, y] = labels[x, y]
        outputs[x, y+1] = labels[x, y+1]
        outputs[x, y+2] = labels[x, y+2]
        labels.cuda()

        return outputs

    def delete_pre_test(labels, outputs):
        x, y = np.where((outputs.cpu()[:,::4]<0.5) & (labels.cpu()[:,::4]==False)==True)
        # x, y = np.where((outputs.cpu()[:,::4]<0.5) & ((labels.cpu()[:,::4]==False) | ((labels.cpu()[:,2::4]==0) & (labels.cpu()[:,3::4]==0)))==True)
        y = y*4+1
        outputs[x, y] = labels[x, y]
        outputs[x, y+1] = labels[x, y+1]
        outputs[x, y+2] = labels[x, y+2]

        return outputs

    def delete_loc_false(labels, outputs):
        x, y = np.where((labels.cpu()[:,1::3]==0) & (labels.cpu()[:,2::3]==0)==True)
        y = y*3
        outputs[x, y] = labels[x, y]
        outputs[x, y+1] = labels[x, y+1]
        outputs[x, y+2] = labels[x, y+2]
        # for i in range(len(labels)):
        #     for j in range(0,num_classes,3):
        #         if labels[i][j+1] == 0 and labels[i][j+2] == 0:
        #             outputs[i,j:j+3] = labels[i,j:j+3]

        return outputs

    def output_test(file, html, names, labels, outputs):
        gt_img = np.load(args.test_gt_dir.format(part_name)).item()
        for i in range(len(names)):
            # visualize txt
            type_gt, fl_gt, fr_gt, bl_gt, br_gt, trunk_gt, az_gt, el_gt, dist_gt = names[i].split('_')
            # content  = "name: {}---gt: [ {}, {}, {}, {}]---predictions: [".format(names[i], bool(labels[i][0]), int(labels[i][1]*data_range), \
            #                                                                     int(labels[i][2]*224), int(labels[i][3]*224))
            # content += '{:.2f}, {}, {}, {}'.format(outputs[i][0], int(round(outputs[i][1]*data_range)), int(round(outputs[i][2]*224)), int(round(labels[i][3]*224)))
            # content += "]\n"
            # file.write(content)
            if args.add_pre:
                html.write("{}:gt {}:pred {}\n".format(names[i], [fl_gt, fr_gt, bl_gt, br_gt, trunk_gt], str(np.array(outputs[i,1::4])*data_range)))
            else:
                html.write("{}:gt {}:pred {}\n".format(names[i], [fl_gt, fr_gt, bl_gt, br_gt, trunk_gt], str(np.array(outputs[i,::3])*data_range)))

        
    def test_model(model, dataloaders, criterion):
        since = time.time()
        file = open(args.output_dir.format(model_name, part_name),'w')
        html = open(args.html_dir.format(model_name, part_name),'w')
        print("Start testing...")
        
        running_loss = {"total_mse": 0.0, "bin_mse": 0.0, "door_mse":0.0, "pos_mse":0.0, "door_mae": 0.0, "pos_mae": 0.0}
        for names, inputs, labels in tqdm(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            model.eval()
            
            outputs = model(inputs)
            if args.add_pre:
                outputs = delete_pre_test(labels, outputs)
                loss_bin = criterion(outputs[:,::4], labels[:,::4])
                loss_door = criterion(outputs[:,1::4], labels[:,1::4])
                loss_pos = (criterion(outputs[:,2::4], labels[:,2::4])+criterion(outputs[:,3::4], labels[:,3::4]))/2
                loss = 0.1*loss_bin + 0.4*loss_door + 0.5*loss_pos
                dist_door = mean_absolute_error(outputs.cpu().detach().numpy()[:,1::4], labels.cpu().detach().numpy()[:,1::4])
                dist_pos = 0.5*mean_absolute_error(outputs.cpu().detach().numpy()[:,2::4]*224, labels.cpu().detach().numpy()[:,2::4]*224)+\
                                        0.5*mean_absolute_error(outputs.cpu().detach().numpy()[:,3::4]*224, labels.cpu().detach().numpy()[:,3::4]*224)
            else:
                outputs = delete_loc_false(labels, outputs)
                loss_door = criterion(outputs[:,::3], labels[:,::3])
                loss_pos = (criterion(outputs[:,1::3], labels[:,1::3])+criterion(outputs[:,2::3], labels[:,2::3]))/2
                loss = 0.5*loss_door + 0.5*loss_pos
                dist_door = mean_absolute_error(outputs.cpu().detach().numpy()[:,::3], labels.cpu().detach().numpy()[:,::3])
                dist_pos = 0.5*mean_absolute_error(outputs.cpu().detach().numpy()[:,1::3]*224, labels.cpu().detach().numpy()[:,1::3]*224)+\
                                        0.5*mean_absolute_error(outputs.cpu().detach().numpy()[:,2::3]*224, labels.cpu().detach().numpy()[:,2::3]*224)
            
            if args.add_pre:
                running_loss["bin_mse"] += loss_bin.item() * inputs.size(0)
            else:
                running_loss["bin_mse"] = 0
            running_loss["door_mse"] += loss_door.item() * inputs.size(0)
            running_loss["pos_mse"] += loss_pos.item() * inputs.size(0)
            running_loss["total_mse"] += loss.item() * inputs.size(0)
            running_loss["door_mae"] += dist_door.item() * inputs.size(0)
            running_loss["pos_mae"] += dist_pos.item() * inputs.size(0)
            
            output_test(file, html, names, labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        
        loss_bin = running_loss["bin_mse"] / len(dataloaders.dataset)
        loss_door = running_loss["door_mse"] / len(dataloaders.dataset)
        loss_pos = running_loss["pos_mse"] / len(dataloaders.dataset)
        loss = running_loss["total_mse"] / len(dataloaders.dataset)
        dist_door = running_loss["door_mae"] / len(dataloaders.dataset)
        dist_pos = running_loss["pos_mae"] / len(dataloaders.dataset)
        file.close()
        html.close()
        
        return loss, loss_door, loss_pos, loss_bin, dist_door, dist_pos
            

    def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        print("Start training...")

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

                running_loss = {"total_mse": 0.0, "bin_mse": 0.0, "door_mse":0.0, "pos_mse":0.0, "door_mae": 0.0, "pos_mae": 0.0}
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

                        if args.add_pre:
                            if phase == "train":
                                outputs = delete_pre_train(labels, outputs)
                            else:
                                outputs = delete_pre_test(labels, outputs)
                            loss_bin = criterion(outputs[:,::4], labels[:,::4])
                            loss_door = criterion(outputs[:,1::4], labels[:,1::4])
                            loss_pos = (criterion(outputs[:,2::4], labels[:,2::4])+criterion(outputs[:,3::4], labels[:,3::4]))/2
                            dist_door = mean_absolute_error(outputs.cpu().detach().numpy()[:,1::4], labels.cpu().detach().numpy()[:,1::4])
                            dist_pos = 0.5*mean_absolute_error(outputs.cpu().detach().numpy()[:,2::4]*224, labels.cpu().detach().numpy()[:,2::4]*224)+\
                                                    0.5*mean_absolute_error(outputs.cpu().detach().numpy()[:,3::4]*224, labels.cpu().detach().numpy()[:,3::4]*224)
                            if epoch < 10:
                                loss = 0.1*loss_bin + 0.3*loss_door + 0.6*loss_pos
                            else:
                                loss = 0.1*loss_bin + 0.6*loss_door + 0.3*loss_pos
                        else:
                            outputs = delete_loc_false(labels, outputs)
                            print(labels)
                            print(outputs)
                            input()
                            loss_door = criterion(outputs[:,::3], labels[:,::3])
                            loss_pos = (criterion(outputs[:,1::3], labels[:,1::3])+criterion(outputs[:,2::3], labels[:,2::3]))/2
                            dist_door = mean_absolute_error(outputs.cpu().detach().numpy()[:,::3], labels.cpu().detach().numpy()[:,::3])
                            dist_pos = 0.5*mean_absolute_error(outputs.cpu().detach().numpy()[:,1::3]*224, labels.cpu().detach().numpy()[:,1::3]*224)+\
                                                    0.5*mean_absolute_error(outputs.cpu().detach().numpy()[:,2::3]*224, labels.cpu().detach().numpy()[:,2::3]*224)
                            if epoch < 10:
                                loss = 0.4*loss_door + 0.6*loss_pos
                            else:
                                loss = 0.6*loss_door + 0.4*loss_pos
                        

                        # print("--step {}: total loss: {}, loss_bin: {}, loss_door: {}, loss_pos: {}".format(i, loss, loss_bin, loss_door, loss_pos))
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    if args.add_pre:
                        running_loss["bin_mse"] += loss_bin.item() * inputs.size(0)
                    else:
                        running_loss["bin_mse"] = 0
                    running_loss["door_mse"] += loss_door.item() * inputs.size(0)
                    running_loss["pos_mse"] += loss_pos.item() * inputs.size(0)
                    running_loss["total_mse"] += loss.item() * inputs.size(0)
                    running_loss["door_mae"] += dist_door * inputs.size(0)
                    running_loss["pos_mae"] += dist_pos * inputs.size(0)

                epoch_loss_bin = running_loss["bin_mse"] / len(dataloaders[phase].dataset)
                epoch_loss_door = running_loss["door_mse"] / len(dataloaders[phase].dataset)
                epoch_loss_pos = running_loss["pos_mse"] / len(dataloaders[phase].dataset)
                epoch_loss = running_loss["total_mse"] / len(dataloaders[phase].dataset)
                epoch_dist_door = running_loss["door_mae"] / len(dataloaders[phase].dataset)
                epoch_dist_pos = running_loss["pos_mae"] / len(dataloaders[phase].dataset)

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
                if phase == 'val' and (epoch == 0 or epoch_dist_door<best_loss):
                    best_loss = epoch_dist_door
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.module.state_dict(), args.model_dir.format(model_name, part_name))
            
            draw_plot()


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
        
    class myDataset(torch.utils.data.Dataset):
        def __init__(self, dataSource, gtSource, mode, cropSource=None, cropGt=None, test_id=None):
            # Just normalization for validation
            self.mode = mode
            self.dir_img = dataSource
            self.gt_img = np.load(gtSource).item()
            if cropSource:
                self.dir_crop = cropSource
                self.gt_crop = np.load(cropGt).item()
            else:
                self.dir_crop = None
                self.gt_crop = None
            self.names = self.load_names(dataSource, mode, test_id)
            print("{} data loaded: {} images".format(mode, len(self.names)))

        def __getitem__(self, index):
            if self.names[index][-3:] == "png":
                return self.names[index], self.load_image(self.dir_crop+self.names[index]), self.load_gt(self.gt_crop, self.names[index][:-4])
            else:
                return self.names[index], self.load_image(self.dir_img+self.names[index]+".png"), self.load_gt(self.gt_img, self.names[index])
            
        def __len__(self):
            return len(self.names)

        def load_crops(self, dir):
            name_data = []
            for file in os.listdir(dir):
                if file[-3:] == "png":
                    name_data.append(file)
            return name_data
        
        def load_names(self, dir, mode, test_id=None):
            name_data = []
            if mode == 'train':
                name_data = open(args.train_name_dir, 'r').read().splitlines() 
                if self.dir_crop:
                    crop_name = self.load_crops(self.dir_crop)
                    name_data += crop_name
            elif mode == 'test':
                for file in os.listdir(dir):
                    if file[-3:] == "png":
                        name_data.append(file[:-4])
            elif mode == 'test_baseline':
                n = 0
                for file in os.listdir(dir):
                    if file[-3:] == "png":
                        if n in test_id:
                            name_data.append(file[:-4])
                    n += 1
            elif mode == 'test_texture':
                name_data = open(args.test_name_dir, 'r').read().splitlines() 

            return name_data

        def load_image(self, dir):
            data_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            img = cv2.imread(dir)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            return data_transforms(img)

        def load_gt(self, dic, name):
            ins, fl, fr, bl, br, trunk, az, el, dist = name.split('_')
            if part_name == "fl":
                bin, x, y = dic[name]
                if args.add_pre:
                    return torch.FloatTensor([bin, abs(int(fl))/data_range if x!= None else 0, x if x!= None else 0, y if x!= None else 0])
                else:
                    return torch.FloatTensor([abs(int(fl))/data_range if x!= None else 0, x if x!= None else 0, y if x!= None else 0])
            elif part_name == "all":
                name = name.replace("Hatchback", "hatchback").replace("Hybrid", "hybrid").replace("Sedan2Door", "sedan2door").replace("Sedan4Door", "sedan4door").replace("Suv", "suv")
                ins, fl, fr, bl, br, trunk, az, el, dist = name.split('_')
                if args.texture:
                    new_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(ins, str(-abs(int(fl))), fr, str(-abs(int(bl))), br, trunk, az, el, dist)
                else:
                    new_name = name
                fl_bin, fl_x, fl_y, fr_bin, fr_x, fr_y, bl_bin, bl_x, bl_y, br_bin, br_x, br_y, trunk_bin, trunk_x, trunk_y =  dic[new_name]
                if args.add_pre:
                    return torch.FloatTensor([(fl_bin and bool(fl_x) and bool(fl_y)), abs(int(fl))/data_range, fl_x, fl_y, \
                                            (fr_bin and bool(fr_x) and bool(fr_y)), abs(int(fr))/data_range, fr_x, fr_y, \
                                            (bl_bin and bool(bl_x) and bool(bl_y)), abs(int(bl))/data_range, bl_x, bl_y, \
                                            (br_bin and bool(br_x) and bool(br_y)), abs(int(br))/data_range, br_x, br_y, \
                                            (trunk_bin and bool(trunk_x) and bool(trunk_y)), abs(int(trunk))/data_range, trunk_x, trunk_y])
                else:
                    return torch.FloatTensor([abs(int(fl))/data_range, fl_x, fl_y, \
                                            abs(int(fr))/data_range, fr_x, fr_y, \
                                            abs(int(bl))/data_range, bl_x, bl_y, \
                                            abs(int(br))/data_range, br_x, br_y, \
                                            abs(int(trunk))/data_range, trunk_x, trunk_y])


    # Detect if we have a GPU available
    device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")

    # Setup the loss fxn
    criterion = nn.MSELoss()

    if args.command == "train":
        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, args.feature_extract, use_pretrained=True)
        model_ft = nn.DataParallel(model_ft)


        # Print the model we just instantiated
        # print(model_ft)

        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        trainsets = myDataset(dataSource=args.train_dir, gtSource=args.train_gt_dir.format(part_name), cropSource=crop_dir, cropGt=crop_gt_dir, mode='train')
        if args.test_texture:
            testsets = myDataset(dataSource=args.train_dir, gtSource=args.train_gt_dir.format(part_name), mode='test_texture')
        else:
            testsets = myDataset(dataSource=args.test_dir.format(part_name), gtSource=args.test_gt_dir.format(part_name), mode='test')

        image_datasets = {'train': trainsets, 'val': testsets}


        # Create training and validation dataloaders
        dataloaders_dict = {x: Data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=8) for x in ['train', 'val']}

        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        # print("Params to learn:")
        if args.feature_extract:
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
        model_ft = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=args.num_epochs, is_inception=(model_name=="inception"))

    # Test
    # Load model
    if args.command == "test":
        model_ft, input_size = initialize_model(model_name, num_classes, args.feature_extract, use_pretrained=False)
        model_ft.load_state_dict(torch.load(args.model_dir.format(model_name, part_name)))
        model_ft = nn.DataParallel(model_ft)
        if isinstance(model_ft,torch.nn.DataParallel):
                model_ft = model_ft.module
        model_ft.to(device)

        model_ft.eval()

        if args.test_baseline:
            # original testset
            random_list = range(args.num_images)
            test_id = random.sample(random_list, 9720)
            testsets = myDataset(dataSource=args.train_dir, gtSource=args.train_gt_dir.format(part_name), mode='test_baseline', test_id=test_id)
        elif args.test_texture:
            testsets = myDataset(dataSource=args.train_dir, gtSource=args.train_gt_dir.format(part_name), mode='test_texture')
        else:
            testsets = myDataset(dataSource=args.test_dir.format(part_name), gtSource=args.test_gt_dir.format(part_name), mode='test')

    # Build testset
    testloader_dict = Data.DataLoader(testsets, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loss, test_door, test_pos, test_bin, dist_door, dist_pos = test_model(model_ft, testloader_dict, criterion)
    print("Total test mse: ", test_loss)
    print("Test binary mse: ", test_bin)
    print("Test door mse: ", test_door)
    print("Test position mse: ", test_pos)
    print("Test door mae: ", dist_door*abs(data_range))
    print("Test position mae: ", dist_pos)

if __name__=="__main__":
    main()

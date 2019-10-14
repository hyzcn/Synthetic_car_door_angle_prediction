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
from sklearn.metrics import mean_absolute_error
import torch.utils.data as Data
from tqdm import tqdm
import random
from model.model_seg import initialize_model
from model.metric import eveluate_iou, vis_seg
from options.train_norm_options import TrainOptions
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

os.environ['QT_QPA_PLATFORM']='offscreen'

def main():
    opt = TrainOptions()
    args = opt.initialize()
    opt.print_options(args)
    train_dict = np.load(args.train_gt_dir).item()
    test_dict = np.load(args.test_gt_dir).item()

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = args.model_name
    part_name = args.part_name

    if part_name == 'all':
        num_classes = 5
    else:
        num_classes = 1

    # Data range
    data_range = args.data_range

    x_list = []
    train_mse = []
    val_mse = []
    train_mae = []
    val_mae = []
    train_seg = []
    val_seg = []
    train_total = []
    val_total = []

    def draw_plot():
        # plot
        plt.subplot(221)
        plt.cla()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title("Total loss")
        plt.plot(x_list,train_total,"x-",label="train loss")
        plt.plot(x_list,val_total,"+-",label="val loss")
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.subplot(222)
        plt.cla()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title("Seg loss")
        plt.plot(x_list,train_seg,"x-",label="train loss")
        plt.plot(x_list,val_seg,"+-",label="val loss")
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.subplot(223)
        plt.cla()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title("MSE loss")
        plt.plot(x_list,train_mse,"x-",label="train loss")
        plt.plot(x_list,val_mse,"+-",label="val loss")
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.subplot(224)
        plt.cla()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title("MAE loss")
        plt.plot(x_list,train_mae,"x-",label="train loss")
        plt.plot(x_list,val_mae,"+-",label="val loss")
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

        plt.savefig(args.plot_dir.format(model_name, part_name))

    def output_test(file, html, names, outputs):
        for i in range(len(names)):
            # visualize txt
            type_gt, fl_gt, fr_gt, bl_gt, br_gt, trunk_gt, az_gt, el_gt, dist_gt = names[i].split('_')
            content  = "gt: [{}]  predictions: [{}]".format(fl_gt,str(int(round(outputs[i][0]*-60))))
            content += "]\n"
            file.write(content)
            html.write("{}:gt {}:pred {}\n".format(names[i], [fl_gt, fr_gt, bl_gt, br_gt, trunk_gt], str(np.array(outputs[i])*data_range)))
            
    def delete_pre_test(names, labels, outputs, ndict):
        num = len(names)*5
        for i in range(len(names)):
            fl_bin, fl_x, fl_y, fr_bin, fr_x, fr_y, bl_bin, bl_x, bl_y, br_bin, br_x, br_y, trunk_bin, trunk_x, trunk_y =  ndict[names[i].lower()]
            if not (fl_bin and bool(fl_x) and bool(fl_y)):
                outputs[i][0] = labels[i][0]
                num -= 1
            if not (fr_bin and bool(fr_x) and bool(fr_y)):
                outputs[i][1] = labels[i][1]
                num -= 1
            if not (bl_bin and bool(bl_x) and bool(bl_y)):
                outputs[i][2] = labels[i][2]
                num -= 1
            if not (br_bin and bool(br_x) and bool(br_y)):
                outputs[i][3] = labels[i][3]
                num -= 1
            if not (trunk_bin and bool(trunk_x) and bool(trunk_y)):
                outputs[i][4] = labels[i][4]
                num -= 1

        return outputs, num

    def test_model(model, dataloaders, criterion):
        since = time.time()
        file = open(args.output_dir.format(model_name, part_name),'w')
        html = open(args.html_dir.format(model_name, part_name),'w')
        
        running_loss = 0
        running_dist = 0
        snum = 0
        IoU = [0 for i in range(args.seg_classes)]
        TP = [0 for i in range(args.seg_classes)]
        FP = [0 for i in range(args.seg_classes)]
        FN = [0 for i in range(args.seg_classes)]
        with torch.no_grad():
            for names, inputs, labels, seglabels in tqdm(dataloaders):
                inputs = inputs.to(device)
                labels = labels.to(device)
                seglabels = seglabels.to(device)
                
                model.eval()
                
                outputs, segputs = model(inputs)
                # segmentation
                segcls = nn.Softmax(1)
                segputs = segcls(segputs)
                segputs = torch.argmax(segputs, dim=1)
                ious, tps, fps, fns = eveluate_iou(seglabels.cpu().numpy(), segputs.cpu().numpy(), args.seg_classes)
                # print(names[0])
                # print(outputs[0]*60)
                # vis_seg(segputs[0].cpu().numpy(), args.seg_classes)
                # input()
                IoU += ious * inputs.size(0)
                TP += tps * inputs.size(0)
                FP += fps * inputs.size(0)
                FN += fns * inputs.size(0)
                
                # regression
                if args.test_spatial:
                    outputs, num = delete_pre_test(names, labels, outputs, test_dict)
                else:
                    outputs, num = delete_pre_test(names, labels, outputs, train_dict)
                loss = criterion(outputs, labels) * inputs.size(0) * num_classes / num
                dist = mean_absolute_error(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())*data_range * inputs.size(0) * num_classes / num
                snum += num

                running_loss += loss.item() * num
                running_dist += dist * num
                
                output_test(file, html, names, outputs.cpu().detach().numpy())
        
        loss = running_loss / snum
        dist = running_dist / snum
        IoU /= len(dataloaders.dataset)
        TP /= len(dataloaders.dataset)
        FP /= len(dataloaders.dataset)
        FN /= len(dataloaders.dataset)
        file.close()
        html.close()
        
        return loss, dist, IoU, TP, FP, FN
            
        

    def train_model(model, dataloaders, criterion, seg_criterion, optimizer, lam, num_epochs=25, is_inception=False):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1000.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            x_list.append(epoch)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_dist = 0.0
                running_seg_loss = 0.0
                running_reg_loss = 0.0
                # Iterate over data.
                snum = 0
                for i, (name, inputs, labels, seglabels) in tqdm(enumerate(dataloaders[phase])):
                    #if epoch == 0:
                    #    print(name)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    seglabels = seglabels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, segputs = model(inputs)
                        # segmentation
                        seg_loss = seg_criterion(segputs, seglabels)

                        # regression
                        if phase == 'train':
                            outputs, num = delete_pre_test(name, labels, outputs, train_dict)
                        else:
                            if args.test_spatial:
                                outputs, num = delete_pre_test(name, labels, outputs, test_dict)
                            else:
                                outputs, num = delete_pre_test(name, labels, outputs, train_dict)
                        reg_loss = criterion(outputs, labels) * inputs.size(0) * num_classes / num
                        dist = mean_absolute_error(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())*data_range* inputs.size(0)* num_classes / num
                        snum += num

                        loss = lam*seg_loss + (1-lam)*reg_loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_dist += dist * num
                    running_reg_loss += reg_loss.item() * num
                    running_seg_loss += seg_loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_dist = running_dist / snum
                epoch_reg_loss = running_reg_loss / snum
                epoch_seg_loss = running_seg_loss / len(dataloaders[phase].dataset)

                print('{} Total Loss: {:.4f}, MSE Loss: {:.4f}, MAE Loss: {:.4f}, Seg Loss: {:.4f}'.format(
                    phase, epoch_loss, epoch_reg_loss, epoch_dist, epoch_seg_loss))
                
                # plot
                if phase == 'train':
                    train_total.append(epoch_loss)
                    train_mse.append(epoch_reg_loss)
                    train_mae.append(epoch_dist)
                    train_seg.append(epoch_seg_loss)
                else:
                    val_total.append(epoch_loss)
                    val_mse.append(epoch_reg_loss)
                    val_mae.append(epoch_dist)
                    val_seg.append(epoch_seg_loss)

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.module.state_dict(), args.model_dir.format(model_name, part_name))

            draw_plot()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
        
    class myDataset(torch.utils.data.Dataset):
        def __init__(self, dataSource, segSource, mode, data_id=None):
            # Just normalization for validation
            self.dir_img = dataSource
            self.seg_dir = segSource
            self.names = self.load_names(dataSource, mode, data_id)
            print("{} data loaded: {} images".format(mode, len(self.names)))

        def __getitem__(self, index):
            return self.names[index], self.load_image(self.dir_img+self.names[index]+".png"), self.load_gt(self.names[index]), self.load_seg_gt(self.seg_dir, self.names[index])
            
        def __len__(self):
            return len(self.names)
        
        def load_names(self, dir, mode, data_id):
            name_data = []
            n = 0
            if mode == "train":
                name_data = open(args.train_name_dir, 'r').read().splitlines() 
            elif mode == "test":
                name_data = open(args.test_name_dir, 'r').read().splitlines() 
            elif mode == "test_spatial":
                for file in os.listdir(dir):
                    if file[-3:] == "png":
                        name_data.append(file[:-4])

            return name_data

        def load_image(self, dir):
            data_transforms = transforms.Compose([
                    # transforms.Resize((224, 224), interpolation=2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            if os.path.isfile(dir):
                img = cv2.imread(dir)
                # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                return data_transforms(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
            else:
                print(dir+" doesn't exist!")
                return []

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
        
        def load_seg_gt(self, dir, name):
            seg_gt = np.load(dir+name+'.npy')
            # seg_gt = cv2.resize(seg_gt, (224, 224), interpolation=cv2.INTER_CUBIC)
            return torch.LongTensor(seg_gt)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup the loss fxn
    criterion = nn.MSELoss()
    seg_criterion = nn.CrossEntropyLoss()

    # Setup train test split
    random_list = range(args.num_images)
    all_id = random.sample(random_list, args.train_num+args.test_num)
    train_id, test_id, _, _ = train_test_split(all_id, [0 for x in range(args.train_num+args.test_num)], test_size=args.test_num/(args.test_num+args.train_num), random_state=42)

    if args.command == "train":
        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, args.seg_classes, args.feature_extract, use_pretrained=True)
        
        model_ft = nn.DataParallel(model_ft)

        # Print the model we just instantiated
        # print(model_ft)

        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        trainsets = myDataset(args.data_dir, args.train_seg_gt_dir, 'train', train_id)
        if args.test_spatial:
            testsets = myDataset(args.test_dir.format(part_name), args.test_seg_gt_dir, 'test_spatial')
        else:
            testsets = myDataset(args.data_dir, args.train_seg_gt_dir, 'test', test_id)

        image_datasets = {'train': trainsets, 'val': testsets}


        # Create training and validation dataloaders
        dataloaders_dict = {x: Data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

        # Send the model to GPU
        model_ft = model_ft.to(device)

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
        model_ft = train_model(model_ft, dataloaders_dict, criterion, seg_criterion, optimizer_ft, args.lam, num_epochs=args.num_epochs, is_inception=(model_name=="inception"))


    # Test
    # Load model
    if args.command == "test":
        model_ft, input_size = initialize_model(model_name, num_classes, args.seg_classes, args.feature_extract, use_pretrained=False)
        model_ft.load_state_dict(torch.load(args.model_dir.format(model_name, part_name)))
        model_ft = nn.DataParallel(model_ft)
        if isinstance(model_ft,torch.nn.DataParallel):
                model_ft = model_ft.module
        model_ft.to(device)

        model_ft.eval()

        if args.test_spatial:
            testsets = myDataset(args.test_dir.format(part_name), args.test_seg_gt_dir, 'test_spatial')
        else:
            testsets = myDataset(args.data_dir, args.train_seg_gt_dir, 'test', test_id)

    # Build testset
    testloader_dict = Data.DataLoader(testsets, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loss, test_dist, IoU, TP, FP, FN = test_model(model_ft, testloader_dict, criterion)
    print("test mse: ", test_loss)
    print("test mae: ", test_dist)
    print("Test mIoU: ", IoU.mean())
    print("Test IoU: ", IoU)
    # print("Test TP: ", TP)
    # print("Test FP: ", FP)
    # print("Test FN: ", FN)

if __name__=='__main__':
    main()
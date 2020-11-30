#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import argparse
import copy
import torch
import torchvision
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

#from torch.optim.lr_scheduler import StepLR


from dataLoader_DeepLab import Dataset

#from engine import train_one_epoch, evaluate

import utils


# In[2]:


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes = num_classes)

    

    return model


# In[3]:


def find_weight_ratio(train_loader):
    pixel_mask = 0
    pixel_bcg = 0
    for data, target in train_loader:
        for t in target:
            t = t.long()
            t = t.squeeze(1)
            mask_r = torch.sum(t[0])
            pixel_mask += mask_r
            pixel_bcg += t[0].numel() - mask_r
    w = float(pixel_mask) / float(pixel_bcg)
    return w


# In[4]:


def i_over_u(inpt,target,threshold):
    intersection = torch.sum(((inpt[0][1] > threshold) & (target[0] > 0)).type(torch.uint8))
    union = torch.sum(((inpt[0][1] > threshold) | (target[0] > 0)).type(torch.uint8))
    iou = (intersection)/(union + 1e-6)
    return iou
    


# In[13]:


def train( model, device, train_loader, optimizer, loss_func, epoch, model_name):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data = torch.stack(data)
        data = data.to(device)
        #target = torch.stack(target)
        target = target.long()
        target = target.squeeze(1)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)['out']
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            with open("DeepLabTrainLoss_" + model_name +".txt","a") as f_train:
                f_train.write(str(loss.item()) + "\n")


# In[6]:


def evaluate(model, device, validation_loader, loss_func, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device)
            target = target.long()
            target = target.squeeze(1)
            target = target.to(device)
            output = model(data)['out']
            #print(loss_func(output, target).item())
            test_loss += loss_func(output, target).item()  # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
            #print("yeet")

    test_loss /= len(validation_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}'.format(
        test_loss))
    with open("DeepLabValidationLoss_" + model_name +".txt","a") as f_eval:
        f_eval.write(str(test_loss) + "\n")
    return test_loss


# In[7]:


def test(model,device,test_loader):
    model.eval()
    ious = []
    thresholds = [0.5,0.6,0.7,0.8,0.9]
    iou = [0,0,0,0,0]
    with torch.no_grad():            
        for data, target in test_loader:
            data = data.to(device)
            target = target.long()
            target = target.squeeze(1)
            target = target.to(device)
            output = model(data)['out']
            for threshold in thresholds:   
                iou[thresholds.index(threshold)] += i_over_u(output,target,threshold)
        iou = np.array(iou) / len(test_loader.dataset)
        for threshold in thresholds:   
            ious.append([threshold,iou[thresholds.index(threshold)].item()])
            #print("Test:" + str([threshold,iou[thresholds.index(threshold)].item()]))
    return ious


# In[8]:


def iou_precision(model,device,test_loader,threshold):
    print("Best threshold: " + str(threshold))
    model.eval()
    precision = [0.5,0.6,0.7,0.8,0.9]
    output_pr = [0,0,0,0,0]
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.long()
            target = target.squeeze(1)
            target = target.to(device)
            output = model(data)['out']
            for pr in precision:
                if i_over_u(output,target,threshold).item() > pr:
                    output_pr[precision.index(pr)] += 1
    output_pr = (np.array(output_pr).astype(float)/float(len(test_loader.dataset))).tolist()
    return list(zip(precision,output_pr))
            


# In[9]:


def precision(model,device,test_loader,threshold):
    model.eval()
    intersection = 0
    output_pixel = 1
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.long()
            target = target.squeeze(1)
            target = target.to(device)
            output = model(data)['out']
            intersection += torch.sum(((output[0][1] > threshold) & (target[0] > 0)).type(torch.uint8))
            output_pixel += torch.sum((output[0][1] > threshold).type(torch.uint8))
    return float(intersection)/float(output_pixel)


# In[10]:


def recall(model,device,test_loader,threshold):
    model.eval()
    intersection = 0
    target_pixel = 1
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.long()
            target = target.squeeze(1)
            target = target.to(device)
            output = model(data)['out']
            intersection += torch.sum(((output[0][1] > threshold) & (target[0] > 0)).type(torch.uint8))
            target_pixel += torch.sum((target[0] > 0).type(torch.uint8))
    return float(intersection)/float(target_pixel)


# In[14]:


def main(load_model = None, dataset = "men", with_weights = False,flip = False, random_crop = False, scaling = False, rotation = False):
    # Training settings
    args = {
        "batch-size" : 1,
        "test-batch-size" : 1, 
        "epochs" : 14, 
        "lr" : 0.001, 
        "gamma" : 0.7, 
        "seed" : 1,
        "log-interval" : 100,
        "save-model" : False
    }

    
    if load_model == None:
        save_name = "model_deeplabv3_8.pt"
    else: 
        save_name = load_model
    
    use_cuda = torch.cuda.is_available()
    #use_cuda = False

    torch.manual_seed(args["seed"])

    # CUDA for PyTorch
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    print(save_name)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # our dataset has two classes only - background and head of person
    num_classes = 2
    
    if with_weights:
        training_set = Dataset('train',dataset = dataset)
        train_loader = data.DataLoader(training_set, batch_size=1, shuffle=True, num_workers=1)
        weights = torch.tensor([find_weight_ratio(train_loader),1.0])
        print(weights)
        weights = weights.to(device)
        loss_func = nn.CrossEntropyLoss(weights)
    else:
        loss_func = nn.CrossEntropyLoss()

    training_set = Dataset('train', dataset = dataset , flip = flip, random_crop = random_crop, scaling=scaling, rotation=rotation)
    train_loader = data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=4, drop_last = True)
    
    validation_set = Dataset('validation', dataset = dataset)
    validation_loader = data.DataLoader(validation_set, batch_size=1, shuffle=False, **kwargs)
    
    test_set = Dataset('test', dataset = dataset)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)
    
    
    # get the model using our helper function
    if load_model == None:
        model = get_model_instance_segmentation(num_classes)
    else:
        model = torch.load(load_model)

    
    # move model to the right device
    model.to(device)

    # construct an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=15,
                                                    gamma=0.1)


    num_epochs = 100

#     images,targets = next(iter(train_loader))
#     images = list(image for image in images)
#     targets = [{k: v.to(device) for k, v in targets.items()}]
    best_val_loss = float("inf")
    best_model = copy.deepcopy(model)
    best_epoch = 0
    
    if load_model == None:
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train(model, device, train_loader, optimizer, loss_func, epoch, save_name)
            average_loss = evaluate(model, device, validation_loader, loss_func, save_name)
            if average_loss < best_val_loss:
                best_val_loss = average_loss
                torch.save(model,save_name)
                best_model = copy.deepcopy(model)
                best_epoch = copy.deepcopy(epoch)
            lr_scheduler.step()
    else:
        best_val_loss = evaluate(best_model, device, validation_loader, loss_func)
    
    print("Best loss: " + str(best_val_loss) + " at epoch " + str(best_epoch))    
    # evaluate on the test dataset
    ious = test(best_model,device,test_loader)
    print("Test (threshold, iou): " + str(ious))
    best_thr = max(ious, key = lambda x : x[1])
    best_thr = best_thr[0]
    p_iou = iou_precision(best_model,device,test_loader,best_thr)
    print("Precision (iou's threshold, %of images above): " + str(p_iou))
    p = precision(best_model,device,test_loader,best_thr)
    print("Precision (intersection/model's activated pixel): " + str(p))
    r = recall(best_model,device,test_loader,best_thr)
    print("Recall (intersection/target's activated pixel): " + str(r))
    with open("DeepLabTestResults.txt","a") as f:
        f.write("=====\n")
        f.write(save_name + "\n")
        f.write("best eval loss: " + str(best_val_loss) + " at epoch " + str(best_epoch) + "\n")
        f.write("ious: " + str(ious)+"\n")
        f.write("precisions iou: " + str(p_iou)+"\n")
        f.write("precisions: " + str(p)+"\n")
        f.write("recall: " + str(r)+"\n")
    print("That's it!")

    #if args["save_model"]:
        #torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main(load_model = None, dataset = "men", with_weights = True, flip = True, random_crop = True, scaling = True, rotation = True)


# In[2]:


# from PIL import Image
# from matplotlib.pyplot import imshow
# import torchvision.transforms as nftf
# model = torch.load("model_deeplabv3_5.pt")
# training_set = Dataset('train')
    
# validation_set = Dataset('validation')
    
# test_set = Dataset('test')


# In[17]:


#  with torch.no_grad():
#     image, mask = test_set.__getitem__(30)
#     im = nftf.ToPILImage()(image)
#     imshow(Image.fromarray(np.uint8(mask[0])))
#     image = [image]
#     image = torch.stack(image)
#     model.to("cuda")
#     model.eval()
#     image = image.to("cuda")
#     output = model(image)['out']
#     print(np.uint8(output[0][0].cpu()))


# In[18]:


#imshow(Image.fromarray(np.uint8(output[0][0].cpu())))


# In[19]:


#imshow(im)


# In[ ]:





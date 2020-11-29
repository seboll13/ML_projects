#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import argparse
import copy

import numpy as np
import math
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import vision.references.detection.transforms as T
import vision.references.detection.utils

#from torch.optim.lr_scheduler import StepLR

from dataLoader_MaskRCNN import Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN

from engine import train_one_epoch, evaluate
import utils


# In[2]:


def find_weight_ratio(train_loader):
    pixel_mask = 0
    pixel_bcg = 0
    for data, target in train_loader:
        for t in target:
            t = t['masks'].long()
            t = t.squeeze(1)
            mask_r = torch.sum(t[0])
            pixel_mask += mask_r
            pixel_bcg += t[0].numel() - mask_r
    w = float(pixel_mask) / float(pixel_bcg)
    return w


# In[3]:


def i_over_u(inpt,target,threshold):
    intersection = torch.sum(((inpt[0] > threshold) & (target[0] > 0)).type(torch.uint8))
    union = torch.sum(((inpt[0] > threshold) | (target[0] > 0)).type(torch.uint8))
    iou = (intersection)/(union + 1e-6)
    return iou
    


# In[4]:


def iou_precision(model,device,test_loader,threshold):
    print("Best threshold: " + str(threshold))
    model.eval()
    precision = [0.5,0.6,0.7,0.8,0.9]
    output_pr = [0,0,0,0,0]
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(img.to(device) for img in images)
            
            new_list = []
            for t in targets:
                new_dict = {}
                for k, v in t.items():
                    if k == 'masks':
                        new_dict[k] = v.long()
                    else:
                        new_dict[k] = v
                new_list.append(new_dict)
            targets = new_list
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            
            for pr in precision:
                for i in range(len(outputs)): # Iterate over all images in batch (should be 1)
                    if outputs[i]['masks'].nelement() > 0:                    
                        output_masks = outputs[i]['masks']
                        for output_mask in output_masks:
                            if i_over_u(output_mask,targets[i]['masks'],threshold).item() > pr:
                                output_pr[precision.index(pr)] += 1
    output_pr = (np.array(output_pr).astype(float)/float(len(test_loader.dataset))).tolist()
    return list(zip(precision,output_pr))
            


# In[5]:


def precision(model,device,test_loader,threshold):
    model.eval()
    intersection = 0
    output_pixel = 1
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(img.to(device) for img in images)
            
            new_list = []
            for t in targets:
                new_dict = {}
                for k, v in t.items():
                    if k == 'masks':
                        new_dict[k] = v.long()
                    else:
                        new_dict[k] = v
                new_list.append(new_dict)
            targets = new_list
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            for i in range(len(outputs)): # Iterate over all images in batch (should be 1)
                if outputs[i]['masks'].nelement() > 0:
                    output_mask = outputs[i]['masks']

                    intersection += torch.sum(((output_mask[0][0] > threshold) & (targets[i]['masks'][0] > 0)).type(torch.uint8))
                    output_pixel += torch.sum((output_mask[0][0] > threshold).type(torch.uint8))        
            
    return float(intersection)/float(output_pixel)


# In[6]:


def recall(model,device,test_loader,threshold):
    model.eval()
    intersection = 0
    target_pixel = 1
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(img.to(device) for img in images)
            
            new_list = []
            for t in targets:
                new_dict = {}
                for k, v in t.items():
                    if k == 'masks':
                        new_dict[k] = v.long()
                    else:
                        new_dict[k] = v
                new_list.append(new_dict)
            targets = new_list
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images) 
            
            for i in range(len(outputs)): # Iterate over all images in batch (should be 1)
                if outputs[i]['masks'].nelement() > 0:
                    output_mask = outputs[i]['masks']

                    intersection += torch.sum(((output_mask[0][0] > threshold) & (targets[i]['masks'][0] > 0)).type(torch.uint8))
                    target_pixel += torch.sum((targets[i]['masks'][0] > 0).type(torch.uint8))
    return float(intersection)/float(target_pixel)


# In[7]:


def validate(model, device, valid_loader, loss_func, model_name):
    model.eval()
    test_loss = 0
    nb_null = 0
    correct = 0
    with torch.no_grad():
        for images, targets in valid_loader:
            images = list(img.to(device) for img in images)
            
            new_list = []
            for t in targets:
                new_dict = {}
                for k, v in t.items():
                    if k == 'masks':
                        new_dict[k] = v.long()
                    else:
                        new_dict[k] = v
                new_list.append(new_dict)

            targets = new_list
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
#             print("target boxes for image 0:")
#             print(targets[0]['boxes'])
#             print(targets[0]['image_id'])
#             print("output boxes :")
            
#             if outputs[0]['boxes'].nelement() == 0 :
#                 print("ERROR: output for image 0 is empty...")    
#             else :
#                 print(outputs[0]['boxes'][0])
                
            tensor_ones = torch.ones([1, 256, 256], dtype = torch.float)
            tensor_ones = tensor_ones.to(device)
            
            for i in range(len(outputs)): # Iterate over all images in batch
                min_loss = 1.0
                
                if outputs[i]['boxes'].nelement() == 0:
                    nb_null += 1
                    tensor_zeros = torch.zeros([1, 256, 256], dtype = torch.float)
                    tensor_zeros = tensor_zeros.to(device)
                    c = torch.cat((tensor_ones, tensor_zeros))
                    c = c.unsqueeze(0)                                            # Add one extra dimension
                    min_loss = loss_func(c, targets[i]['masks']).item()
                    print(min_loss)
                
                else:
                    for j in range(len(outputs[i]['masks'])):                         # Iterate over all masks for image i of the batch
                        output_mask = outputs[i]['masks'][j]
                        output_mask = output_mask.cpu().numpy()
                        output_mask = output_mask > 0.5
                        output_mask = output_mask.astype(int)  
                        output_mask = torch.as_tensor(output_mask, dtype=torch.float32)
                        output_mask = output_mask.to(device)

                        tensor_class_zero = tensor_ones - output_mask 
                        c = torch.cat((tensor_class_zero, output_mask))               # Prepend mask of class zero (background)
                        c = c.unsqueeze(0)                                            # Add one extra dimension
                        loss = loss_func(c, targets[i]['masks']).item()               # Compute CrossEntropy loss, c.size() = [1, 2, 256, 256]

                        if loss < min_loss:
                            min_loss = loss
                        
                test_loss += min_loss
#             test_loss += loss_func(outputs, targets).item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= (len(valid_loader.dataset))
    with open("MaskRCNNValidationLoss_" + model_name +".txt","a") as f_eval:
        f_eval.write(str(test_loss) + " " + str(nb_null) + "\n")
    
    print('\nvalidation set: Average loss: {:.4f}'.format(test_loss))
    print('\nvalidation set: Images that could not be predicted: {} \n'.format(nb_null))
    
    return test_loss


# In[8]:


def test(model,device,test_loader):
    model.eval()
    ious = []
    thresholds = [0.5,0.6,0.7,0.8,0.9]
    iou = [0,0,0,0,0]
    with torch.no_grad():            
        for images, targets in test_loader:
            images = list(img.to(device) for img in images)
            
            new_list = []
            for t in targets:
                new_dict = {}
                for k, v in t.items():
                    if k == 'masks':
                        new_dict[k] = v.long()
                    else:
                        new_dict[k] = v
                new_list.append(new_dict)
            targets = new_list
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            
            for threshold in thresholds:   
                for i in range(len(outputs)): # Iterate over all images in batch (should be 1)
                    if outputs[i]['masks'].nelement() > 0:
                        output_masks = outputs[i]['masks']
                        best_result = 0
                        for output_mask in output_masks:
                            result = i_over_u(output_mask,targets[i]['masks'],threshold)
                            if result > best_result:
                                best_result = result
                        iou[thresholds.index(threshold)] += best_result
                
        iou = np.array(iou) / len(test_loader.dataset)
        for threshold in thresholds:   
            ious.append([threshold,iou[thresholds.index(threshold)].item()])
            #print("Test:" + str([threshold,iou[thresholds.index(threshold)].item()]))
    return ious


# In[9]:


def get_model_instance_segmentation(num_classes):
    
    # Anchor Generator, defined the same way as in MaskRCNN
    anchor_sizes = ((16,), (24,), (32,), (48,), (64,))
    aspect_ratios = ((0.5, 0.75, 1.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )

    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,
                                                               min_size=256, max_size=256,
                                                               rpn_anchor_generator=rpn_anchor_generator)

    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# In[10]:


# def get_model_instance_segmentation(num_classes):

#     # load a pre-trained model for classification and return
#     # only the features
#     backbone = torchvision.models.mobilenet_v2(pretrained=False).features
#     # FasterRCNN needs to know the number of
#     # output channels in a backbone. For mobilenet_v2, it's 1280
#     # so we need to add it here
#     backbone.out_channels = 1280

#     # let's make the RPN generate 5 x 3 anchors per spatial
#     # location, with 5 different sizes and 3 different aspect
#     # ratios. We have a Tuple[Tuple[int]] because each feature
#     # map could potentially have different sizes and
#     # aspect ratios
#     anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                        aspect_ratios=((0.5, 1.0, 2.0),))

#     # let's define what are the feature maps that we will
#     # use to perform the region of interest cropping, as well as
#     # the size of the crop after rescaling.
#     # if your backbone returns a Tensor, featmap_names is expected to
#     # be [0]. More generally, the backbone should return an
#     # OrderedDict[Tensor], and in featmap_names you can choose which
#     # feature maps to use.
#     roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
#                                                     output_size=7,
#                                                     sampling_ratio=2)

#     # put the pieces together inside a FasterRCNN model
#     model = FasterRCNN(backbone,
#                        num_classes=num_classes,
#                        min_size=256, max_size=256,
#                        rpn_anchor_generator=anchor_generator,
#                        box_roi_pool=roi_pooler)
#     return model


# In[11]:


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# In[12]:


def main(load_model = None):
    # Training settings
    args = {
        "batch-size" : 1,
        "test-batch-size" : 1, 
        "epochs" : 14, 
        "lr" : 0.001, 
        "gamma" : 0.1, 
        "seed" : 1,
        "log-interval" : 100,
        "save-model" : False
    }

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args["seed"])

    # CUDA for PyTorch
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # our dataset has two classes only - background and head of person
    num_classes = 2
    
    training_set = Dataset('train', dataset = "women", flip=False, random_crop=False, scaling = False, rotation = False)
    train_loader = data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn, drop_last=True)
    
    valid_set = Dataset('validation', dataset = "women")
    valid_loader = data.DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn, drop_last=True)
    
    test_set = Dataset('test', dataset = "women")
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn, drop_last=True)

    loss_func = nn.CrossEntropyLoss()
    
    # get the model using our helper function
    if load_model == None:
        save_name = "model_maskrcnn_100_001_women.pt"
        model = get_model_instance_segmentation(num_classes)
    else:
        save_name = load_model
        model = torch.load(load_model)
    
    print("model name:")
    print(save_name)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 100 epochs
    num_epochs = 100

    best_val_loss = float("inf")
    best_epoch = 0
    best_model = copy.deepcopy(model)

    if load_model == None:    
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, train_loader, device, epoch, save_name, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            
#             print('\nvalidation set: Average loss: {:.4f}'.format(average_loss))
#             with open("MaskRCNNValidationLoss_" + model_name +".txt","a") as f_eval:
#                 f_eval.write(str(average_loss) + "\n")
            
            # validate on the validation dataset
            average_loss = validate(model, device, valid_loader, loss_func, save_name)
            if epoch > 40 and average_loss < best_val_loss:
                best_val_loss = average_loss
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                torch.save(best_model,save_name)
        print("Arrived at end of loop \n")
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
    
    with open("MaskRCNNTestResults.txt","a") as f:
        f.write("=====\n")
        f.write(save_name + "\n")
        f.write("best eval loss: " + str(best_val_loss) + " at epoch " + str(best_epoch) + "\n")
        f.write("ious: " + str(ious)+"\n")
        f.write("precisions iou: " + str(p_iou)+"\n")
        f.write("precisions: " + str(p)+"\n")
        f.write("recall: " + str(r)+"\n")
    print("That's it!")
    

if __name__ == '__main__':
    main()


# In[130]:


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.pyplot import imshow
# from PIL import Image
# import numpy as np
# import torchvision.transforms as nftf
# import torchvision.transforms.functional as tf


# model = torch.load("model_maskrcnn_100_001_aug.pt")
# training_set = Dataset('train')
    
# validation_set = Dataset('validation')
    
# test_set = Dataset('test')
# loss = nn.CrossEntropyLoss(torch.tensor([0.0087,1.0]))


# In[131]:


# device = torch.device("cuda")
# with torch.no_grad():
#     image, target = test_set.__getitem__(30)
#     im = nftf.ToPILImage()(image)
#     image = [image]
#     image = torch.stack(image)
#     model.to(device)
#     model.eval()
#     image = image.to(device)
#     output = model(image)
    
#     output_mask = output[0]['masks'][0]
#     output_mask = output_mask.cpu().numpy()
#     output_mask = output_mask > 0.5
#     output_mask = output_mask.astype(int)  
#     output_mask = torch.as_tensor(output_mask, dtype=torch.float32)
#     output_mask = output_mask.to(device)
    
#     tensor_ones = torch.ones([1, 256, 256], dtype = torch.float)
#     tensor_ones = tensor_ones.to(device)
#     output_mask = target['masks'].float().to(device)
#     tensor_class_zero = tensor_ones - output_mask 
#     c = torch.cat((tensor_class_zero, output_mask))               # Prepend mask of class zero (background)
#     c = c.unsqueeze(0) 
    
#     print(c.size())
#     print(target['masks'].size())
#     print(output[0]['boxes'][0])
#     print(area(output[0]['boxes'][0]))
#     print(target['boxes'])
#     print(area(target['boxes'][0]))
#     l = loss(c.cpu(), target['masks'].long())
    
# #     l.backward()


# In[132]:


# print(l)


# In[100]:


# def area(box):
#     return (box[3] - box[1]) * (box[2] - box[0])


# In[116]:


# im = np.array(tf.to_pil_image(image[0].cpu()), dtype=np.uint8)

# # Create figure and axes
# fig,ax = plt.subplots(1)

# # Display the image
# ax.imshow(im)

# # Create a Rectangle patch
# boxes = target['boxes'].numpy()[0]
# x = boxes[2] - boxes[0]
# y = boxes[3] - boxes[1]
# rect = patches.Rectangle(boxes[0:2],x,y,linewidth=1,edgecolor='r',facecolor='none')

# # Add the patch to the Axes
# ax.add_patch(rect)

# plt.show()


# In[117]:


# # Create figure and axes
# fig,ax = plt.subplots(1)

# # Display the image
# ax.imshow(im)

# boxes = (output[0]['boxes']).cpu().numpy()
# for box in boxes:
#     # Create a Rectangle patch
#     x = box[2] - box[0]
#     y = box[3] - box[1]
#     rect = patches.Rectangle(box[0:2],x,y,linewidth=1,edgecolor='r',facecolor='none')

#     # Add the patch to the Axes
#     ax.add_patch(rect)

# plt.show()


# In[114]:


# # Create figure and axes
# fig,ax = plt.subplots(1)

# # Display the image
# ax.imshow(im)

# boxes = (output[0]['boxes']).cpu().numpy()
# for box in boxes:
#     # Create a Rectangle patch
#     x = box[2] - box[0]
#     y = box[3] - box[1]
#     rect = patches.Rectangle(box[0:2],x,y,linewidth=1,edgecolor='r',facecolor='none')

#     # Add the patch to the Axes
#     ax.add_patch(rect)

# plt.show()


# In[123]:


# # Create figure and axes
# fig,ax = plt.subplots(1)

# # Display the image
# ax.imshow(im)

# boxes = (output[0]['boxes']).cpu().numpy()
# for box in boxes:
#     # Create a Rectangle patch
#     x = box[2] - box[0]
#     y = box[3] - box[1]
#     rect = patches.Rectangle(box[0:2],x,y,linewidth=1,edgecolor='r',facecolor='none')

#     # Add the patch to the Axes
#     ax.add_patch(rect)

# plt.show()


# In[50]:


# # Create figure and axes
# fig,ax = plt.subplots(1)

# # Display the image
# ax.imshow(im)

# boxes = (output[0]['boxes']).cpu().numpy()
# for box in boxes:
#     # Create a Rectangle patch
#     x = box[2] - box[0]
#     y = box[3] - box[1]
#     rect = patches.Rectangle(box[0:2],x,y,linewidth=1,edgecolor='r',facecolor='none')

#     # Add the patch to the Axes
#     ax.add_patch(rect)

# plt.show()


# In[118]:


# imshow(Image.fromarray(np.uint8(target['masks'][0])))


# In[119]:


# print(output)


# output_mask = output[0]['masks'][0][0]
# output_mask = output_mask.cpu().numpy()
# output_mask = output_mask > 0.5
# output_mask = output_mask.astype(int)  
# output_mask = torch.as_tensor(output_mask, dtype=torch.float32)

# imshow(Image.fromarray(np.uint8(output_mask)))


# In[ ]:





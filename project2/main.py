#!/usr/bin/env python
# coding: utf-8

# In[33]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils import data

from data.dataLoader_synthetic_dataset import Dataset
from model.architectures.resnet import resnet101


# In[64]:


def get_model():
    
    resnet_3d = models.video.r3d_18(pretrained=False, num_classes=12)
    resnet_mixed_conv = models.video.mc3_18(pretrained=False, num_classes=12)
    resnet_2_1d = models.video.r2plus1d_18(pretrained=False, num_classes=12)

    resnet_101 = resnet101(num_classes=12)
    
    net = resnet_101
    return net


# In[65]:


def mse_loss(input, target): 
    return torch.sum((input - target) ** 2) 


# In[66]:


def train(model, device, train_loader, optimizer, loss_func, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data = torch.stack(data)
        data = data.type(torch.float32).to(device)
        batch_size = data.size()[0]
        
        print('Input size:', data.size(), '\n')
        
        #target = torch.stack(target)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        print('Target values:', target)
        print('Output values:', output, '\n')
        
        print('Target size: ', target.size())
        print('Output size: ', output.size(), '\n')
        
        loss = 0
        for i in range(len(target)):
            loss += loss_func(output[i], target[i])
        loss /= len(target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# In[67]:


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
            output = model.predict(data)['out']
            #print(loss_func(output, target).item())
            test_loss += loss_func(output, target) # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(validation_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}'.format(
        test_loss))
    with open("DeepLabValidationLoss_" + model_name +".txt","a") as f_eval:
        f_eval.write(str(test_loss) + "\n")
    return test_loss


# In[68]:


def main():
    # Training settings
    args = {
        "batch-size" : 1,
        "test-batch-size" : 1, 
        "epochs" : 1, 
        "lr" : 0.001, 
        "gamma" : 0.1, 
        "seed" : 1,
        "step_size" : 10,
        "log-interval" : 100,
        "save-model" : False
    }
    
    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    torch.manual_seed(args["seed"])

    # CUDA for PyTorch
    device = torch.device("cpu") # GPU if possible
    print(device)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    training_set = Dataset('train', nb_of_input_images = 10)
    train_loader = data.DataLoader(training_set, batch_size=2, shuffle=True, num_workers=4)
    
#     validation_set = Dataset('validation', dataset = dataset)
#     validation_loader = data.DataLoader(validation_set, batch_size=1, shuffle=False, **kwargs)
    
#     test_set = Dataset('test', dataset = dataset)
#     test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)
    
    
#     # get the model using our helper function
#     if load_model == None:
#         model = get_model_instance_segmentation(num_classes)
#     else:
#         model = torch.load(load_model)
    model = get_model()
    loss_func = mse_loss
    
    # move model to the right device
    model.to(device)

    # construct an optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args['step_size'],
                                                    gamma=args['gamma'])

    num_epochs = args['epochs']

#     best_val_loss = float("inf")
#     best_model = copy.deepcopy(model)
#     best_epoch = 0
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train(model, device, train_loader, optimizer, loss_func, epoch)
        average_loss = evaluate(model, device, validation_loader, loss_func, save_name)
#         if average_loss < best_val_loss:
#             best_val_loss = average_loss
#             torch.save(model,save_name)
#             best_model = copy.deepcopy(model)
#             best_epoch = copy.deepcopy(epoch)
        #lr_scheduler.step()
    


# In[69]:


if __name__ == '__main__':
    main()


# In[45]:


get_ipython().system('jupyter nbconvert --to script main.ipynb')


# In[ ]:





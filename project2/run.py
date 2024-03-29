import sys
import matplotlib.pyplot as plt
import os
import os.path
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils import data

from data.dataLoader_synthetic_dataset import Dataset as SyntheticDataset
from data.dataLoader_real_dataset import Dataset as RealDataset
from data.dataLoader_both_dataset import Dataset as MixedDataset

from architectures.resnets import r3d_18
from architectures.resnets import mc3_18
from architectures.resnets import r2plus1d_18

# Imports all parameters from run_parameters.py
from run_parameters import *
# Import functions to create graphs
import display_losses


def get_model(model_name):
    """Selects the architecture to train and/or test on
       
       # Arguments
           model_name: name of the model
        
       The implementations of the architectures are imported from pytorch, which can be found in architectures/resnets.py or here: 
           https://pytorch.org/docs/stable/torchvision/models.html#video-classification
           https://pytorch.org/docs/stable/_modules/torchvision/models/video/resnet.html
    """
    if model_name == 'resnet_3d':
        net = r3d_18(pretrained=False, num_classes=num_classes)
    elif model_name == 'resnet_mixed_conv':
        net = mc3_18(pretrained=False, num_classes=num_classes)
    elif model_name == 'resnet_2_1d':
        net = r2plus1d_18(pretrained=False, num_classes=num_classes)
    else :
        sys.exit('Error: Incorrect model name')
    return net


def train(model, device, train_loader, optimizer, loss_func, epoch, model_name,path):
    """Trains the model for an epoch on the training set, calculates the loss averaged over the batch and propagates it to the network.
       
       # Arguments
           model: model
           device: the device to run on, CPU or GPU
           train_loader: dataloader for the training set
           optimizer: the optimizer
           loss_func: the loss function, Mean Squared Error
           epoch: the current epoch it is running
           model_name: the name of the model, for results saving purposes
           path: the path to the saving location
       
       The training losses are saved in models/[model_name]_[id]/losses_train.txt
       """
    model.train()
    train_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move tensors to the right device
        data = data.type(torch.float32).to(device)
        target = target.to(device)
        
        # Prepare optimizer, train and propagate the loss
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output.float(), target.float())
        train_loss += loss
        loss.backward()
        optimizer.step()
        
        print('[epoch %d, batch_idx %2d] => average datapoint and batch loss : %.2f' % (epoch+1, batch_idx, loss.item()))

    train_loss /= (len(train_loader.dataset)/ batch_size)
    print('\nTraining set: Average loss: {:.4f}\n'.format(train_loss))
    
    # Save training loss in .txt
    txt_path = os.path.join(path, "losses_train.txt") 
    with open(txt_path, "a") as f_train:
        f_train.write(str(train_loss.item()) + "\n")
    print('Finished training')
    
    
def evaluate(model, device, validation_loader, loss_func, model_name, path):
    """Evaluates the model on the validation set for one epoch
    
       # Arguments
           model: the model
           device: CPU or GPU
           validation_loader: dataloader for the validation set
           loss_func: the loss function, Mean Squared Error
           model_name: the name of the model, for results saving purposes
           path: the path to the saving location    
    
    The validation losses are saved in models/[model_name]/losses_valid.txt
    """
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_loader):
            # Move tensors to the right device
            data = data.type(torch.float32).to(device)
            target = target.to(device)
            
            # Evaluate
            output = model(data)
            loss = loss_func(output.float(), target.float())
            test_loss += loss # sum up batch loss
            
            if batch_idx == 0:
                print('\n One validation output example:')
                print(target)
                print(output)
    
    test_loss /= (len(validation_loader.dataset)/ batch_size)
    print('\nValidation set: Average loss: {:.4f}\n'.format(test_loss))
    
    txt_path = os.path.join(path, "losses_valid.txt") 
    with open(txt_path, "a") as f_valid:
        f_valid.write(str(test_loss.item()) + "\n")
    
    return test_loss


def main():
    # Generates the folder [model_name]_[id]/ to save the model, its settings, the losses and test results during training
    # An ID is generated by checking models/ folder, to avoid overriding previous runs.
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    num = 0
    save_name = model_name + "_" 
    while os.path.exists(os.path.join(models_folder,save_name + str(num))):
        num += 1
    save_name = save_name + str(num)
    path = os.path.join(models_folder,save_name)
    os.mkdir(path)
    f = open(os.path.join(path,"settings.txt"), 'w')
    for t in settings:
        line = ' '.join(str(x) for x in t)
        f.write(line + '\n')
    f.close()
    save_path = os.path.join(path, "model.pth")


    torch.manual_seed(seed)
    
    # Selects the device to train the model on (CPU or GPU)
    device = torch.device("cuda" if use_cuda else "cpu") 
    print('Using device:', device)

    # Loads the synthetic dataset or real dataset
    if select_datalaoder == "synthetic":
        print("Loading synthetic data...")
        print('Number of input images:', nb_of_input_images)
        training_set = SyntheticDataset('train', nb_of_input_images = nb_of_input_images, normalize = normalize)
        validation_set = SyntheticDataset('validation', nb_of_input_images = nb_of_input_images, normalize = normalize)
    elif select_datalaoder == "real":
        print("Loading real data...")
        print('Number of input images:', nb_of_input_images)
        training_set = RealDataset('train', nb_of_input_images = nb_of_input_images, normalize = normalize)
        validation_set = RealDataset('validation', nb_of_input_images = nb_of_input_images, normalize = normalize)
    elif select_datalaoder == "mixed":
        print("Loading mixed data...")
        print('Number of input images:', nb_of_input_images)
        training_set = MixedDataset('train', nb_of_input_images = nb_of_input_images)
        validation_set = MixedDataset('validation', nb_of_input_images = nb_of_input_images)
    else:
        raise ValueError("Wrong dataloader; Only synthetic, real or mixed")
        
    print("Amount of training images: " + str(training_set.__len__()))
    print("Amount of validation images: " + str(validation_set.__len__()))
    
    train_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=False, num_workers=num_train_workers)
    validation_loader = data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_valid_workers)
    
    # Generates a new model to be trained
    model = get_model(model_name)
    print('Using model :', model_name)
    print('Saving model in :', path)
    
    # defines the loss function
    loss_func = nn.MSELoss(reduction='mean')
    
    # move model to the right device
    model.to(device)
    # construct an optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)

    
    # Training loop
    print('Training for :', num_epochs, 'epochs')
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, loss_func, epoch, model_name,path)
        average_loss = evaluate(model, device, validation_loader, loss_func, model_name,path)
        lr_scheduler.step()

    torch.save(model,save_path)
    return save_name


if __name__ == '__main__':
    save_name = main()
    display_losses.main(save_name)
    
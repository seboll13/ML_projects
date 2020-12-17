import sys
import matplotlib.pyplot as plt
import os.path
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils import data

from data.dataLoader_synthetic_dataset import Dataset as SyntheticDataset
from data.dataLoader_real_dataset import Dataset as RealDataset

from architectures.resnets import r3d_18
from architectures.resnets import mc3_18
from architectures.resnets import r2plus1d_18

# Model parameters
num_classes = 13
model_names = ['resnet_3d', 'resnet_mixed_conv', 'resnet_2_1d']
model_name = model_names[0]

# Dataloader parameters
train_on_synthetic_data = True
nb_of_input_images = 2000
num_train_workers = 0
num_valid_workers = 0

# If possible, GPU will be used for computation
use_cuda = True & torch.cuda.is_available()

# Training parameters
batch_size = 8
test_batch_size = 1
num_epochs = 30
gamma = 0.1
lr = 0.005
step_size = 10
seed = 1

settings = (
    ("model_name",model_name),
    ("train_on_synthetic_data",train_on_synthetic_data),
    ("nb_of_input_images",nb_of_input_images),
    ("num_train_workers",num_train_workers),
    ("num_valid_workers",num_valid_workers),
    ("batch_size",batch_size),
    ("test_batch_size",test_batch_size),
    ("num_epochs",num_epochs),
    ("gamma",gamma),
    ("lr",lr),
    ("step_size",step_size),
    ("seed",seed)
)

# Saving parameters
models_folder = "models"
load_model = False
load_folder = ""
#results_folder = "results"
#trained_on = 's' if train_on_synthetic_data else 'r'
#path_parameters = str(num_epochs) + "_" + trained_on + str(nb_of_input_images) + "_" + str(lr) + "_" + str(gamma)
#save_model = True
#save_name = "models/" + model_name + path_parameters + ".pth"

def get_model(model_name):
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
    """Train the model"""
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.type(torch.float32).to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output.float(), target.float())
        train_loss += loss
        loss.backward()
        optimizer.step()
        
        if batch_idx == 0:
            print('\n One training output example:')
            print(target)
            print(output, '\n')      
        print('[epoch %d, batch_idx %2d] => average datapoint and batch loss : %.2f' % (epoch+1, batch_idx, loss.item()))
        
    train_loss /= len(train_loader.dataset)/batch_size
    print('\nTraining set: Average loss: {:.4f}\n'.format(train_loss))
    if not load_model:
        txt_path = os.path.join(path, "losses_train.txt") 
        with open(txt_path, "a") as f_train:
            f_train.write(str(train_loss.item()) + "\n")
    print('Finished training')
    
    
def evaluate(model, device, validation_loader, loss_func, model_name, path):
    """Validate if the model has trained well or not"""
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_loader):
            data = data.type(torch.float32).to(device)
            target = target.to(device)
            
            output = model(data)
            loss = loss_func(output.float(), target.float())
            test_loss += loss # sum up batch loss
            
            if batch_idx == 0:
                print('\n One validation output example:')
                print(target)
                print(output)

    test_loss /= len(validation_loader.dataset)/batch_size
    print('\nValidation set: Average loss: {:.4f}\n'.format(test_loss))
    
    if not load_model:
        txt_path = os.path.join(path, "losses_valid.txt") 
        with open(txt_path, "a") as f_valid:
            f_valid.write(str(test_loss.item()) + "\n")
    
    return test_loss


def test(model, device, test_loader, loss_func, model_name, path):
    """Test the model evaluation"""
    model.eval()
    test_loss = 0
    
    print('Testing...')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.type(torch.float32).to(device)
            target = target.to(device)
            
            output = model(data)
            loss = loss_func(output.float(), target.float())
            test_loss += loss # sum up batch loss
            
            target = target.cpu()
            output = output.cpu()
            target = target.numpy()[0]
            output = output.numpy()[0]
            
            # Write out test loss, labels and outputs 
            txt_path = os.path.join(path, "losses_test.txt") 
            with open(txt_path, "a") as f_test:
                f_test.write(str(loss.item()) + "\n")
                
            txt_path = os.path.join(path, "test_results.txt") 
            with open(txt_path,"a") as f_test:
                f_test.write('[')
                for i in range(len(target)):
                    f_test.write(str(target[i]))
                    if i < len(target) - 1:
                        f_test.write(', ')
                f_test.write('] \n')
                
                f_test.write('[')
                for i in range(len(output)):
                    f_test.write(str(output[i]))
                    if i < len(output) - 1:
                        f_test.write(', ')
                    
                f_test.write('] \n')
                
    print('Finished test prints')
    test_loss /= len(test_loader.dataset)
    
    return test_loss


def main():
    if not load_model :
        if not os.path.exists(models_folder):
            os.mkdir(models_folder)
        num = 0
        save_name = model_name + "_" 
        while os.path.exists(os.path.join(models_folder,save_name + str(num))):
            num += 1
        path = os.path.join(models_folder,save_name + str(num))
        os.mkdir(path)
        f = open(os.path.join(path,"settings.txt"), 'w')
        for t in settings:
            line = ' '.join(str(x) for x in t)
            f.write(line + '\n')
        f.close()
        save_name = os.path.join(path, "model.pth")
    else:
        path = os.path.join(models_folder,load_folder)
        load_name = os.path.join(models_folder,load_folder,"model.pth")

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu") # GPU if possible
    print('Using device:', device)

    
    if train_on_synthetic_data:
        print("Loading synthetic data...")
        print('Number of input images:', nb_of_input_images)
        training_set = SyntheticDataset('train', nb_of_input_images = nb_of_input_images)
        validation_set = SyntheticDataset('validation', nb_of_input_images = nb_of_input_images)
        test_set = SyntheticDataset('test', nb_of_input_images = nb_of_input_images)
    else:
        print("Loading real data...")
        print('Number of input images:', nb_of_input_images)
        training_set = RealDataset('train', nb_of_input_images = nb_of_input_images)
        validation_set = RealDataset('validation', nb_of_input_images = nb_of_input_images)
        test_set = RealDataset('test', nb_of_input_images = nb_of_input_images)
    
    print("Amount of training images: " + str(training_set.__len__()))
    print("Amount of validation images: " + str(validation_set.__len__()))
    print("Amount of testing images: " + str(test_set.__len__()))
    
    train_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=False, num_workers=num_train_workers)
    validation_loader = data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_valid_workers)
    test_loader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    
    if load_model:
        model = torch.load(load_name)
    else:
        model = get_model(model_name)
        print('Using model :', model_name)
        print('Saving model in :', path)
    
    loss_func = nn.MSELoss(reduction='mean')
    
    # move model to the right device
    model.to(device)
    # construct an optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)

    best_model = copy.deepcopy(model)
    
    if not load_model:
        print('Training for :', num_epochs, 'epochs')
        for epoch in range(num_epochs):
            train(model, device, train_loader, optimizer, loss_func, epoch, model_name,path)
            average_loss = evaluate(model, device, validation_loader, loss_func, model_name,path)
            if not load_model:
                torch.save(model,save_name)
                best_model = copy.deepcopy(model)
            lr_scheduler.step()
    
    print(test(best_model, device, test_loader, loss_func, model_name,path))
    sys.exit()


if __name__ == '__main__':
    main()
import sys
import matplotlib.pyplot as plt
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils import data

from data.dataLoader_synthetic_dataset import Dataset as SyntheticDataset
from data.dataLoader_real_dataset import Dataset as RealDataset

from model.architectures.resnets import r3d_18
from model.architectures.resnets import mc3_18
from model.architectures.resnets import r2plus1d_18


num_classes = 13
train_on_synthetic_data = True

results_folder = "results"


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


def train(model, device, train_loader, optimizer, loss_func, epoch, model_name):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.type(torch.float32).to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = loss_func(output.float(), target.float())
        
        loss.backward()
        optimizer.step()
        if batch_idx == 0:
            print('\n One training output example:')
            print(target.int8())
            print(output, '\n')
            
        print('[epoch %d, batch_idx %2d] => average datapoint and batch loss : %.2f' % (epoch+1, batch_idx, loss.item()))
        
        if batch_idx % 5 == 0:
            path = os.path.join(results_folder, model_name + "_train.txt") 
            with open(path, "a") as f_train:
                f_train.write(str(loss.item()) + "\n")
    print('Finished training')

    
    
    
def evaluate(model, device, validation_loader, loss_func, model_name):
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
                print(target.int8())
                print(output)

    test_loss /= len(validation_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}\n'.format(test_loss))
    
    path = os.path.join(results_folder, model_name + "_valid.txt") 
    with open(path, "a") as f_valid:
        f_valid.write(str(test_loss.item()) + "\n")
    
    return test_loss



def test(model, device, test_loader, loss_func, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    print('Testing...')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.type(torch.float32).to(device)
            target = target.to(device)
            
            output = model(data)
            loss = loss_func(output.float(), target.float())
            test_loss += loss # sum up batch loss
            
            target = target.numpy()[0]
            output = output.numpy()[0]
            
            # Write out test labels and outputs 
            
            path = os.path.join(results_folder, model_name + "_test.txt") 
            with open(path, "a") as f_test:
                f_test.write(str(loss.item()) + "\n")
                
            path = os.path.join(results_folder, model_name + "_test_results.txt") 
            with open(path,"a") as f_test:
                f_test.write('[')
                for i in range(len(target)):
                    f_test.write(str(target.uint8()[i]))
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
    # Training settings
    args = {
        "batch_size" : 2,
        "test_batch_size" : 1, 
        "epochs" : 30, 
        "gamma" : 0.1, 
        "log-interval" : 100,
        "lr" : 0.01, 
        "model_name" : "resnet_2_1d",
        "seed" : 1,
        "step_size" : 10,
        "save-model" : False
    }
    
    torch.manual_seed(args["seed"])

    use_cuda = False #torch.cuda.is_available()
    
    # CUDA for PyTorch
    device = torch.device("cuda" if use_cuda else "cpu") # GPU if possible
    print('Using device: ', device)

    
    if train_on_synthetic_data:
        training_set = SyntheticDataset('train', nb_of_input_images = 100)
        validation_set = SyntheticDataset('validation', nb_of_input_images = 100)
        test_set = SyntheticDataset('test', nb_of_input_images = 100)
    else :
        training_set = RealDataset('train')
        validation_set = RealDataset('validation')
        test_set = RealDataset('test')
    
    
    train_loader = data.DataLoader(training_set, batch_size=args['batch_size'], shuffle=True, num_workers=4)
    validation_loader = data.DataLoader(validation_set, batch_size=args['batch_size'], shuffle=True, num_workers=1)
    test_loader = data.DataLoader(test_set, batch_size=args['test_batch_size'], shuffle=False)
    
    
#     # get the model using our helper function
#     if load_model == None:
#         model = get_model_instance_segmentation(num_classes)
#     else:
#         model = torch.load(load_model)

    model_name = args['model_name']
    model = get_model(args['model_name'])
    print('Using model : ', args['model_name'])
    
    loss_func = nn.MSELoss(reduction='mean')
    
    # move model to the right device
    model.to(device)
    # construct an optimizer
    optimizer = optim.SGD(model.parameters(), lr=args['lr'])
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args['step_size'],
                                                    gamma=args['gamma'])

    num_epochs = args['epochs']
    
#     best_val_loss = float("inf")
#     best_model = copy.deepcopy(model)
#     best_epoch = 0
    print('Training for :', num_epochs, 'epochs')
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train(model, device, train_loader, optimizer, loss_func, epoch, model_name)
        average_loss = evaluate(model, device, validation_loader, loss_func, model_name)
#         if average_loss < best_val_loss:
#             best_val_loss = average_loss
#             torch.save(model,save_name)
#             best_model = copy.deepcopy(model)
#             best_epoch = copy.deepcopy(epoch)
        lr_scheduler.step()
    
    
    test(model, device, test_loader, loss_func, model_name)
    
    
if __name__ == '__main__':
    main()
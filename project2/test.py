import sys
import os
import os.path

import torch
import torch.nn as nn
from torch.utils import data

from data.dataLoader_real_dataset import Dataset as RealDataset
from test_parameters import *
import display_test_results

def test(model, device, test_loader, loss_func, path):
    """Test the final model after selecting the most performing one
    
       # Arguments
           model: model
           device: CPU or GPU
           test_loader: data loader for the testing set
           loss_func: MSE
           path: the path to the saving location
    
    The test losses are saved in models/[model_name_id]/losses_test.txt, and the speedplots (predicted outputs and labels) are saved in models/[model_name_id]/test_results.txt
    """
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
                
    print('Saved test results')
    test_loss /= len(test_loader.dataset)
    
    return test_loss


def main():
    # Retrieve the folder containing the model to be tested
    path = os.path.join(models_folder,load_folder)
    load_path = os.path.join(models_folder,load_folder,"model.pth")

    torch.manual_seed(seed)
    
    # Selects the device to train the model on (CPU or GPU)
    device = torch.device("cuda" if use_cuda else "cpu") 
    print('Using device:', device)

    # Loads the real testing dataset
    test_set = RealDataset('test', nb_of_input_images = nb_of_input_images, normalize = normalize)
    print("Amount of testing images: " + str(test_set.__len__()))
    test_loader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    
    # Loads a pretrained model to be tested
    print('Loading model from folder:', load_folder)
    model = torch.load(load_path)
    
    # defines the loss function
    loss_func = nn.MSELoss(reduction='mean')
    
    # move model to the right device
    model.to(device)

    # Test
    print(test(model, device, test_loader, loss_func, path))

    
if __name__ == '__main__':
    main()
    display_test_results.main(load_folder)
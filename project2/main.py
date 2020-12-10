import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils import data

from data.dataLoader_real_dataset import Dataset
from model.architectures.resnet import resnet101


num_classes = 13

def get_model(model_name):
    if model_name == 'resnet_3d':
        net = models.video.r3d_18(pretrained=False, num_classes=num_classes)
    elif model_name == 'resnet_mixed_conv':
        net = models.video.mc3_18(pretrained=False, num_classes=num_classes)
    elif model_name == 'resnet_2_1d':
        net = models.video.r2plus1d_18(pretrained=False, num_classes=num_classes)
    elif model_name == 'resnet_101':
        net = resnet101(num_classes=num_classes)
    else:
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
        loss = torch.sqrt(2 * loss)
        
        loss.backward()
        optimizer.step()
        if batch_idx == 0:
            print('\n One training output example:')
            print(target)
            print(output, '\n')
            
        print('[epoch %d, batch_idx %2d] => average batch loss: %.2f' % (epoch+1, batch_idx, loss.item()))
        
        if batch_idx % 5 == 0:
            with open(model_name + "_train.txt","a") as f_train:
                f_train.write(str(loss.item()) + "\n")
    print('Finished training')


def evaluate(model, device, validation_loader, loss_func, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_loader):
            data = data.type(torch.float32).to(device)
            target = target.to(device)
            
            output = model(data)
            loss = loss_func(output.float(), target.float())
            loss = torch.sqrt(2 * loss)
            test_loss += loss # sum up batch loss
            
            if batch_idx == 0:
                print('\n One validation output example:')
                print(target)
                print(output)

    test_loss /= len(validation_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}\n'.format(test_loss))
    
    with open(model_name + "_valid.txt","a") as f_eval:
        f_eval.write(str(test_loss.item()) + "\n")
    
    return test_loss


def main():
    # Training settings
    args = {
        "batch_size" : 2,
        "test_batch_size" : 1, 
        "frames_per_datapoint": 1000,
        "epochs" : 20, 
        "gamma" : 0.07, 
        "log-interval" : 100,
        "lr" : 0.01, 
        "model_name" : "resnet_2_1d",
        "seed" : 1,
        "step_size" : 10,
        "save-model" : False
    }
    
    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    torch.manual_seed(args["seed"])

    # CUDA for PyTorch
    device = torch.device("cuda" if use_cuda else "cpu") # GPU if possible
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    training_set = Dataset('train')
    train_loader = data.DataLoader(training_set, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    
    validation_set = Dataset('validation')
    validation_loader = data.DataLoader(
        validation_set, batch_size=args['batch_size'], shuffle=True, **kwargs
    )
    
#     test_set = Dataset('test', dataset = dataset)
#     test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)
    
    
#     # get the model using our helper function
#     if load_model == None:
#         model = get_model_instance_segmentation(num_classes)
#     else:
#         model = torch.load(load_model)

    model_name = args['model_name']
    model = get_model(args['model_name'])
    print('Using model : ', args['model_name'])
    
    loss_func = nn.MSELoss(reduction='sum')
    
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
    

if __name__ == '__main__':
    main()
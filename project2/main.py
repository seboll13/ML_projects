import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils import data
from data.dataLoader_synthetic_dataset import Dataset
from model.architectures.resnet import resnet101

num_classes = 12

def mse_loss(input, target): 
    return torch.sum((input - target) ** 2) 


def get_model():
    resnet_3d = models.video.r3d_18(pretrained=False, num_classes=num_classes)
    resnet_mixed_conv = models.video.mc3_18(pretrained=False, num_classes=num_classes)
    resnet_2_1d = models.video.r2plus1d_18(pretrained=False, num_classes=num_classes)

    resnet_101 = resnet101(num_classes=num_classes)
    
    return resnet_101


def train(model, device, train_loader, optimizer, loss_func, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.type(torch.float32).to(device)
        batch_size = data.size()[0]
        
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        # print('Target values: ', target)
        # print('Output values: ', output, '\n')
        # print('Target size: ', target.size())
        # print('Output size: ', output.size(), '\n')
        
        loss = 0
        for i in range(len(target)):
            loss += loss_func(output[i], target[i])
        loss /= len(target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('[epoch %d, batch_idx %2d] => loss: %.2f' % (epoch+1, batch_idx, loss.item()))
    print('Finished training')


def evaluate(model, device, validation_loader, loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.type(torch.float32).to(device)
            target = target.to(device)
            output = model(data)
            #print(loss_func(output, target).item())
            test_loss += loss_func(output, target) # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(validation_loader.dataset)

    return test_loss


def main():
    # Training settings
    args = {
        "batch_size" : 2,
        "test_batch_size" : 1, 
        "epochs" : 2, 
        "lr" : 0.001, 
        "gamma" : 0.1, 
        "seed" : 2,
        "step_size" : 10,
        "log_interval" : 100,
        "save_model" : False
    }
    
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args["seed"])

    # CUDA for PyTorch
    device = torch.device("cpu")
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    training_set = Dataset('train', nb_of_input_images = 10)
    train_loader = data.DataLoader(
        training_set, batch_size=args['batch_size'], shuffle=True, num_workers=4
    )
    
    validation_set = Dataset('validation', nb_of_input_images = 10)
    validation_loader = data.DataLoader(
        validation_set, batch_size=args['batch_size'], shuffle=False, **kwargs
    )
    
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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args['step_size'], gamma=args['gamma']
    )

    num_epochs = args['epochs']

#     best_val_loss = float("inf")
#     best_model = copy.deepcopy(model)
#     best_epoch = 0
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train(model, device, train_loader, optimizer, loss_func, epoch)
        average_loss = evaluate(model, device, validation_loader, loss_func)
#         if average_loss < best_val_loss:
#             best_val_loss = average_loss
#             torch.save(model,save_name)
#             best_model = copy.deepcopy(model)
#             best_epoch = copy.deepcopy(epoch)
        #lr_scheduler.step()
    

if __name__ == '__main__':
    main()

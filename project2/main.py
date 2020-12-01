import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils import data
from model.architectures.resnet import resnet101
from data.dataLoader_synthetic_dataset import Dataset

num_classes = 12


def mse_loss(input, target): 
    return torch.sum((input - target) ** 2) 


def get_model():
    resnet_3d = models.video.r3d_18(pretrained=False, num_classes=num_classes)
    resnet_mixed_conv = models.video.mc3_18(pretrained=False, num_classes=num_classes)
    resnet_2_1d = models.video.r2plus1d_18(pretrained=False, num_classes=num_classes)

    resnet_101 = resnet101(num_classes=num_classes)
    
    return resnet_3d


def train(model, device, train_loader, optimizer, loss_func, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.type(torch.float32).to(device)
        batch_size = data.size()[0]
        
        print('Input size: ', data.size())
        
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)

        print('Target values: ', target)
        print('Output values: ', output)
        
        print('Target size: ', target.size())
        print('Output size: ', output.size())
        
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
            test_loss += loss_func(output, target).item()

    test_loss /= len(validation_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}'.format(
        test_loss))
    with open("DeepLabValidationLoss_" + model_name +".txt","a") as f_eval:
        f_eval.write(str(test_loss) + "\n")
    return test_loss


def main():
    # Training settings
    args = {
        "batch_size" : 2,
        "test_batch_size" : 1, 
        "epochs" : 1, 
        "lr" : 1e-4, 
        "gamma" : 0.1, 
        "seed" : 1,
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
        training_set, 
        batch_size=args['batch_size'], 
        shuffle=True, 
        num_workers=4
    )

    validation_set = Dataset('validation', dataset = dataset)
    validation_loader = data.DataLoader(
        validation_set, 
        batch_size=args['batch_size'], 
        shuffle=False, 
        **kwargs
    )
    
    model = get_model()
    loss_func = mse_loss
    
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args['step_size'],
        gamma=args['gamma']
    )

    num_epochs = args['epochs']

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train(model, device, train_loader, optimizer, loss_func, epoch)
        average_loss = evaluate(model, device, validation_loader, loss_func, save_name)


if __name__ == '__main__':
    main()
import torch


## Main parameters


########## Model initialisation parameters ##########
# number of values the model has to predict.
num_classes = 13

# names of the three architectures, to simplify switching one architecture to another.
model_names = ['resnet_3d', 'resnet_mixed_conv', 'resnet_2_1d']
# name of the architecture that will be trained and/or tested
model_name = model_names[0]


########## Dataloader parameters ####################
# Selects the dataset to be used (synthetic, real or mixed)
select_datalaoder = "synthetic"
# number of frames to be considered as a single datapoint. We have chosen 2000 images.
nb_of_input_images = 2000
#Normalizes labels (= speeds) to be between -10 and 10
normalize = True

# number of workers for the data loader of the training set.
num_train_workers = 4
# number of workers for the data loader of the validation set.
num_valid_workers = 1


########## Hardware parameters ######################
# train the model on GPU if True (and if cuda is available on the hardware), otherwise train on CPU.
use_cuda = True & torch.cuda.is_available()


########## Training parameters ######################
# number of inputs to be trained simultaneously. 
batch_size = 8
# total number of training epochs.
num_epochs = 30
# learning rate
lr = 0.02
# ratio at which the learning rate decreases
gamma = 0.1
# number of epochs after which the learning decreases 
step_size = 10
# seed for random functions in PyTorch
seed = 1


########## Saving parameters ######################
# format to save all parameters in settings.txt for future analysis
settings = (("model_name",model_name),
            ("select_datalaoder",select_datalaoder),
            ("nb_of_input_images",nb_of_input_images),
            ("num_train_workers",num_train_workers),
            ("num_valid_workers",num_valid_workers),
            ("batch_size",batch_size),
            ("num_epochs",num_epochs),
            ("gamma",gamma),
            ("lr",lr),
            ("step_size",step_size))

# folder containing the models and their results
models_folder = "models"
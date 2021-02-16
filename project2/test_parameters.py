import torch


## Test parameters

########## Dataloader parameters ####################
#Normalizes labels (= speeds) te be between -10 and 10
normalize = True
# number of frames to be considered as a single datapoint. We have chosen 2000 images.
nb_of_input_images = 2000


########## Hardware parameters ######################
# train the model on GPU if True (and if cuda is available on the hardware), otherwise train on CPU.
use_cuda = True & torch.cuda.is_available()


########## Testing parameters ######################
# number of inputs to be tested simultaneously. 
test_batch_size = 1
# seed for random functions in PyTorch
seed = 1

# folder containing the models and their results
models_folder = "models"
# folder to load the model from
load_folder = "resnet_2_1d_0"
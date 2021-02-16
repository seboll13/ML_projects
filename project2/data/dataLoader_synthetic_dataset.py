import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image
import math
import random

#Creates a custom torch dataloader to be used with torch-based architecture, which loads synthetic data.
#Note: by design, it will not load data from disk until they are requested by the main script.
class Dataset(data.Dataset):
    #Initialisation function; this is where the amount of images per datapoint is specified, 
    #as well as the set from which it should give datapoints; train, validation, testing or all
    def __init__(self, usage, nb_of_input_images = 2000, normalize = False):
        #Setting for the behavior of the dataloader: 
        ratio_train_test = 0.8 #first ratio that will be applied to all data
        ratio_train_validation = 0.8 #second ratio; it applied to datapoints that aren't in the testing set
        random.seed(1) #seed to make reproducible randomness
        
        self.input_nb = nb_of_input_images
        self.training = False   #this might have been used for augmenting the data during training
        self.root = 'data/data' #this is the folder inside which the synthetic set's folders should be placed in order for the dataloader to see them
        
        speedplot_rescale_min = -10
        speedplot_rescale_max = 10
        #It scans the root folder for synthetic sets counting up
        folder_num = 0
        self.folders = []
        self.images_by_folder = []
        while os.path.exists(os.path.join(self.root, "synthetic_data_" + str(folder_num))):
            self.folders.append("synthetic_data_" + str(folder_num))
            folder_num += 1
        
        #From the list of folders, it gathers the link to the frames in a list.
        for folder in self.folders:
            image_num = 0
            frames = []
            while os.path.exists(os.path.join(self.root,folder,"frame_" + str(image_num) + ".png")):
                frames.append(os.path.join(self.root, folder, "frame_" + str(image_num) + ".png"))
                image_num += 1
            self.images_by_folder.append(frames.copy())
        
        #It then organizes the list of frames names in datapoints; folder that have not enough frames for one datapoints are ignored. 
        #It also means it will ignore frames if they aren't divisible by the number of frames per datapoints; 
        #for instance you have 2 frames per datapoints, and this folder has 3 frames; the first two will make a datapoint and the third will be ignored.
        self.img_sets = []
        for image_set in self.images_by_folder:
            if len(image_set)/self.input_nb < 1:
                print("Error: not enough frames in folder " + self.folders[self.images_by_folder.index(image_set)] + " for at least one input(" + str(self.input_nb) + ")")
            else:
                for s in range(0,math.floor(len(image_set)/self.input_nb)):
                    self.img_sets.append(image_set[s*self.input_nb:(s+1)*self.input_nb])
        
        #shuffles the datapoints to homogenize the different scenarios
        random.shuffle(self.img_sets)
        
        #This is where it slices the list of datapoints depending on the utilisation
        if usage == 'test':
            self.img_sets = self.img_sets[math.floor(len(self.img_sets)*ratio_train_test):]
        elif usage == 'validation':
            self.img_sets = self.img_sets[math.floor(len(self.img_sets)*ratio_train_test*ratio_train_validation)
            :math.floor(len(self.img_sets)*ratio_train_test)]
        elif usage == 'train':
            self.training = True
            self.img_sets = self.img_sets[:math.floor(len(self.img_sets)*ratio_train_test*ratio_train_validation)]
        elif usage == 'all':
            self.training = True
        else:
            raise ValueError("Wrong usage specified; Only train,validation,test or all")
        
        #After we have our final list of datapoints for this dataloader, it will look into the folders of each datapoints to find the labels list.
        self.speedplots = [np.array([])]*len(self.img_sets)
        for set_nb in range(0,len(self.img_sets)):
            (head, tail) = os.path.split(self.img_sets[set_nb][0])
            speedplot = np.genfromtxt(os.path.join(head, "speedplot.csv") , delimiter=',')
            if normalize:
                speedplot = ((speedplot - speedplot.min()) / (speedplot.max() - speedplot.min())) * (speedplot_rescale_max - speedplot_rescale_min) + speedplot_rescale_min     #scales the speedplots to a common range for both datasets
            self.speedplots[set_nb] = speedplot            
    
    #Returns the amount of datapoints for the current utilisation
    def __len__(self):
        return len(self.img_sets)
    
    #Returns the datapoint at the specified index in the list of datapoints
    def __getitem__(self, index):
        #Concatenates the list of frames name into one numpy array. 
        (height,width) = np.asarray(Image.open(self.img_sets[0][0])).shape
        np_images = np.zeros((self.input_nb,height,width))
        for img in range(0,len(self.img_sets[index])):
            np_images[img,:,:] = np.asarray(Image.open(self.img_sets[index][img]))
        speedplot = self.speedplots[index]
        #Once it has fetched the datapoints, it applies any extra transformation befor handing it to the main script.
        return self.transform(np_images,speedplot)
    
    #Utility function to format the datapoints in a compatible way to the architecture; 
    #this is also where we would put augmentations for training if we had done any.
    def transform(self,images,speedplot):
        #This is where we convert it from 1-channel to 3-channel
        return np.repeat(images[np.newaxis, :, :, :], 3, axis = 0), speedplot
import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image
import math
import pickle
import random

#Creates a custom torch dataloader to be used with torch-based architecture, which loads the real data.
#Note: contrary to the synthetic dataloader, it has to load all the dataset in the RAM; 
#this is because the files we were provided were pickles files, and we had to unpickle them to work with the sets.
class Dataset(data.Dataset):
    #Initialisation function; this is where the amount of images per datapoint is specified, 
    #as well as the set from which it should give datapoints; train, validation, testing or all
    def __init__(self, usage, nb_of_input_images = 17800, normalize = False):
        #Setting for the behavior of the dataloader: 
        ratio_train_test = 0.8  #first ratio that will be applied to all data
        ratio_train_validation = 0.8  #second ratio; it applied to datapoints that aren't in the testing set
        random.seed(1) #seed to make reproducible randomness
        
        self.input_nb = nb_of_input_images
        self.training = False  #this might have been used for augmenting the data during training
        self.root = 'data/data'  #this is the folder inside which the real sets .pickle files should be placed in order for the dataloader to see them
        
        speedplot_rescale_min = -10
        speedplot_rescale_max = 10
        #Scans the root folder to get all the .pickle files
        self.filelist=os.listdir(self.root)
        for file in self.filelist[:]: 
            if not(file.endswith(".pickle")):
                self.filelist.remove(file)
        
        #Here it unpickles the files and generates the datapoints, as well as the labels
        self.img_sets = []
        self.speedplots = []
        for file in self.filelist[:]:
            with open(os.path.join(self.root,file),'rb') as fo:
                u = pickle._Unpickler(fo)
                u.encoding = 'latin1'
                shot, t_window, brt_arr, r_arr, z_arr, vz_from_wk, vz_err_from_wk, col_r = u.load()
                if np.isnan(np.sum(np.asarray(vz_from_wk))):        #This is where it refuses sets which have NaN in labels
                    self.filelist.remove(file)
                    print(file + " had NaN in labels and was not included in the dataset")
                elif len(t_window)/self.input_nb < 1:               
                        print("Error: not enough frames in file " + file 
                                 + " for at least one input(" + str(self.input_nb) + " frames)")
                else:
                    #Scales the array to be between 0 and 255
                    np_brt_arr = np.asarray(brt_arr)
                    np_brt_arr = np.nan_to_num(np_brt_arr)
                    np_brt_arr *= (255.0/np_brt_arr.max())
                    np_brt_arr = np_brt_arr.astype(np.uint8)
                    
                    #In the labels, the speeds of the columns with the shear might not be ordered correctly; 
                    #For instance, it might be something like 15,12,10,5,-1,2,-3,-6,ect
                    #So we need to reorder it to be: 15,12,10,5,2,-1,-3,-6,ect in order to be consistent with the labelling.
                    col_r = np.sort(col_r)
                    indices = np.argsort(col_r)
                    speedplot = np.asarray(vz_from_wk)[indices]
                    shear = list(i if col_r[i]==col_r[i+1] else -1 for i in range(0,len(col_r)-1))
                    shear = list(value for value in shear if value != -1)
                    if not shear:
                        raise ValueError("Error: the column speed listing in " + file 
                                 + " did not have correct position information")
                    shear = shear[0]
                    if speedplot[shear] > 0 and speedplot[shear+1] < 0:
                        speedplot[shear], speedplot[shear+1] = speedplot[shear+1],speedplot[shear]
                    for s in range(0,math.floor(len(t_window)/self.input_nb)):
                        self.img_sets.append(np_brt_arr[:,:,s*self.input_nb:(s+1)*self.input_nb])
                        if normalize:
                            speedplot = ((speedplot - speedplot.min()) / (speedplot.max() - speedplot.min())) * (speedplot_rescale_max - speedplot_rescale_min) + speedplot_rescale_min     #scales the speedplots to a common range for both datasets
                        self.speedplots.append(speedplot)
        
        #shuffles the datapoints to homogenize the different scenarios
        zipped = list(zip(self.img_sets,self.speedplots))
        random.shuffle(zipped)
        self.img_sets,self.speedplots = zip(*zipped)
        
        #This is where it slices the list of datapoints and labels depending on the utilisation
        if usage == 'test':
            self.img_sets = self.img_sets[math.floor(len(self.img_sets)*ratio_train_test):]
            self.speedplots = self.speedplots[math.floor(len(self.speedplots)*ratio_train_test):]
        elif usage == 'validation':
            self.img_sets = self.img_sets[math.floor(len(self.img_sets)*ratio_train_test*ratio_train_validation)
            :math.floor(len(self.img_sets)*ratio_train_test)]
            self.speedplots = self.speedplots[math.floor(len(self.speedplots)*ratio_train_test*ratio_train_validation)
            :math.floor(len(self.speedplots)*ratio_train_test)]
        elif usage == 'train':
            self.training = True
            self.img_sets = self.img_sets[:math.floor(len(self.img_sets)*ratio_train_test*ratio_train_validation)]
            self.speedplots = self.speedplots[:math.floor(len(self.speedplots)*ratio_train_test*ratio_train_validation)]
        elif usage == 'all':
            self.training = True
        else:
            raise ValueError("Wrong usage specified; Only train,validation,test or all")
        
    #Returns the amount of datapoints for the current utilisation 
    def __len__(self):
        return len(self.img_sets)
    
    #Returns the datapoint at the specified index in the list of datapoints
    def __getitem__(self, index):
        #Concatenates the list of frames name into one numpy array. 
        np_images = np.zeros((self.input_nb, self.img_sets[index].shape[1], self.img_sets[index].shape[0]))
        for img in range(0,self.img_sets[index].shape[2]):
            np_images[img,:,:] = self.img_sets[index][:, :, img].T
        speedplot = self.speedplots[index]
        #Once it has fetched the datapoints, it applies any extra transformation befor handing it to the main script.
        return self.transform(np_images,speedplot)
    
    #Utility function to format the datapoints in a compatible way to the architecture; 
    #this is also where we would put augmentations for training if we had done any.
    def transform(self,images,speedplot):
        #This is where we convert it from 1-channel to 3-channel
        return np.repeat(images[np.newaxis, :, :, :], 3, axis = 0), speedplot

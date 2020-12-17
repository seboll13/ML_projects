import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image
import math
import pickle
import random

#Creates a custom torch dataloader to be used with torch-based architecture, which loads the real and synthetic data.
#It will load a mix of both for training, but only images from the real dataset for validation and testing.
#This datalaoder was not used in this project, due to both time constraints and the report which would have been too large; still we include it in case the lab is interested in it.
#Note: due to the fact that it has to mix both datasets, it will load all datasets in RAM.
class Dataset(data.Dataset):
    def __init__(self, usage, nb_of_input_images = 17800):
        #Setting for the behavior of the dataloader: 
        ratio_train_test = 0.8 #first ratio that will be applied to all data
        ratio_train_validation = 0.8 #second ratio; it applied to datapoints that aren't in the testing set
        random.seed(1) #seed to make reproducible randomness
        
        self.input_nb = nb_of_input_images
        self.training = False   #this might have been used for augmenting the data during training
        self.root = 'data/data' #this is the folder inside which the synthetic set's folders and the real data's pickle files should be placed in order for the dataloader to see them
        
        ############################################
        #Here it loads the real data just like the specific dataloader
        
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
                        speedplot *= (10.0/speedplot.max())        #scales the speedplots to match both datasets
                        self.speedplots.append(speedplot)
        
        
        #shuffles the datapoints to homogenize the different scenarios
        zipped = list(zip(self.img_sets,self.speedplots))
        random.shuffle(zipped)
        self.img_sets,self.speedplots = zip(*zipped)
        
        ##################################################################################
        #Here it loads the synthetic data, but unlike the synthetic dataloader it has to load it in memory to mix it with the real datapoints
        
        #It scans the root folder for synthetic sets counting up
        folder_num = 0
        self.folders = []
        while os.path.exists(os.path.join(self.root, "synthetic_data_" + str(folder_num))):
            self.folders.append("synthetic_data_" + str(folder_num))
            folder_num += 1
        
        #From the list of folders, it gathers the link to the frames in a list.      
        self.images_by_folder = []    
        self.images_by_folder_np = []        
        for folder in self.folders:
            image_num = 0
            frames = []
            frames_np = []
            while os.path.exists(os.path.join(self.root,folder,"frame_" + str(image_num) + ".png")):
                frames_np.append(np.asarray(Image.open(os.path.join(self.root, folder, "frame_" + str(image_num) + ".png"))).T)
                frames.append(os.path.join(self.root, folder, "frame_" + str(image_num) + ".png"))
                image_num += 1
            self.images_by_folder.append(frames.copy())
            self.images_by_folder_np.append(np.array(frames_np.copy()))
        
        #It then organizes the list of frames names in datapoints; folder that have not enough frames for one datapoints are ignored. 
        #It also means it will ignore frames if they aren't divisible by the number of frames per datapoints; 
        #for instance you have 2 frames per datapoints, and this folder has 3 frames; the first two will make a datapoint and the third will be ignored.
        #Here it also loads the synthetic data's label early, as they need to be mixed with the real ones
        self.img_sets_synth = []
        self.img_sets_synth_name = []
        self.speedplots_synth = []
        for image_set in range(0,len(self.images_by_folder_np)):
            if len(self.images_by_folder[image_set])/self.input_nb < 1:
                print("Error: not enough frames in folder " + self.folders[image_set] 
                                 + " for at least one input(" + str(self.input_nb) + ")")
            else:
                for s in range(0,math.floor(len(self.images_by_folder_np[image_set])/self.input_nb)):
                    np_set = np.zeros((12,10,self.input_nb))
                    for img in range(0, self.input_nb):
                        np_set[:,:,img] = self.images_by_folder_np[image_set][img,:,:]
                    self.img_sets_synth.append(np_set.copy())
                    (head, tail) = os.path.split(self.images_by_folder[image_set][0])
                    speedplot = np.genfromtxt(os.path.join(head, "speedplot.csv") , delimiter=',')
                    speedplot *= (10.0/speedplot.max())        #scales the speedplots to match both datasets
                    self.speedplots_synth.append(speedplot)
        
        
        ##################################################################################
            
        
        #This is where it slices the list of datapoints depending on the utilisation
        if usage == 'test':
            self.img_sets = self.img_sets[math.floor(len(self.img_sets)*ratio_train_test):]
            self.speedplots = self.speedplots[math.floor(len(self.speedplots)*ratio_train_test):]
        elif usage == 'validation':
            self.img_sets = self.img_sets[math.floor(len(self.img_sets)*ratio_train_test*ratio_train_validation)
            :math.floor(len(self.img_sets)*ratio_train_test)]
            self.speedplots = self.speedplots[math.floor(len(self.speedplots)*ratio_train_test*ratio_train_validation)
            :math.floor(len(self.speedplots)*ratio_train_test)]
        elif usage == 'train' or usage == 'all':
            #Removes the test and validation sets for training
            if usage == 'train':
                self.img_sets = self.img_sets[:math.floor(len(self.img_sets)*ratio_train_test*ratio_train_validation)]
                self.speedplots = self.speedplots[:math.floor(len(self.speedplots)*ratio_train_test*ratio_train_validation)]
            self.img_sets = list(self.img_sets)
            self.speedplots = list(self.speedplots)
            #If it's not for testing or validation, then we can add the synthetic data to the mix and shuffle it again
            for i in range(0,len(self.img_sets_synth)):
                self.img_sets.append(self.img_sets_synth[i])
                self.speedplots.append(self.speedplots_synth[i])
            zipped = list(zip(self.img_sets,self.speedplots))
            random.shuffle(zipped)
            self.img_sets,self.speedplots = zip(*zipped)
            
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

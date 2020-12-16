import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image
import math
import random

class Dataset(data.Dataset):
    def __init__(self, usage, nb_of_input_images = 2000, flip = False):
        ratio_train_test = 0.8
        ratio_train_validation = 0.8
        random.seed(1)
        
        self.input_nb = nb_of_input_images
        self.flip = flip
        self.training = False
        self.root = 'data/data'
        folder_num = 0
        self.folders = []
        self.images_by_folder = []
        while os.path.exists(os.path.join(self.root, "synthetic_data_" + str(folder_num))):
            self.folders.append("synthetic_data_" + str(folder_num))
            folder_num += 1

        for folder in self.folders:
            image_num = 0
            frames = []
            while os.path.exists(os.path.join(self.root,folder,"frame_" + str(image_num) + ".png")):
                frames.append(os.path.join(self.root, folder, "frame_" + str(image_num) + ".png"))
                image_num += 1
            self.images_by_folder.append(frames.copy())
        
        self.img_sets = []
        for image_set in self.images_by_folder:
            if len(image_set)/self.input_nb < 1:
                print("Error: not enough frames in folder " + self.folders[self.images_by_folder.index(image_set)] + " for at least one input(" + str(self.input_nb) + ")")
            else:
                for s in range(0,math.floor(len(image_set)/self.input_nb)):
                    self.img_sets.append(image_set[s*self.input_nb:(s+1)*self.input_nb])
        
        random.shuffle(self.img_sets)
        
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
        
        self.speedplots = [np.array([])]*len(self.img_sets)
        for set_nb in range(0,len(self.img_sets)):
            (head, tail) = os.path.split(self.img_sets[set_nb][0])
            self.speedplots[set_nb] = np.genfromtxt(os.path.join(head, "speedplot.csv") , delimiter=',')
    
    def __len__(self):
        return len(self.img_sets)
    
    def __getitem__(self, index):
        (height,width) = np.asarray(Image.open(self.img_sets[0][0])).shape
        np_images = np.zeros((self.input_nb,height,width))
        for img in range(0,len(self.img_sets[index])):
            np_images[img,:,:] = np.asarray(Image.open(self.img_sets[index][img]))
        speedplot = self.speedplots[index]
        return self.transform(np_images,speedplot)
        
    def transform(self,images,speedplot):
        return np.repeat(images[np.newaxis, :, :, :], 3, axis = 0), speedplot
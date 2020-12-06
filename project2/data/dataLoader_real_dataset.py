import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image
import math
import pickle

class Dataset(data.Dataset):
    def __init__(self, usage, nb_of_input_images = 17800):
        self.input_nb = nb_of_input_images
        self.training = False
        self.root = 'realdata'
        
        self.filelist=os.listdir(self.root)
        for file in self.filelist[:]: # filelist[:] makes a copy of filelist.
            if not(file.endswith(".pickle")):
                self.filelist.remove(file)
        
        self.img_sets = []
        self.speedplots = []
        for file in self.filelist[:]:
            with open(os.path.join(self.root,file),'rb') as fo:
                u = pickle._Unpickler(fo)
                u.encoding = 'latin1'
                shot, t_window, brt_arr, r_arr, z_arr, vz_from_wk, vz_err_from_wk, col_r = u.load()
                if np.isnan(np.sum(np.asarray(vz_from_wk))):
                    self.filelist.remove(file)
                    print(file + " had NaN in labels and was not included in the dataset")
                else:
                    if shot/self.input_nb < 1:
                        raise ValueError("Error: not enough frames in file " + file 
                                 + " for at least one input(" + str(self.input_nb) + " frames)")
                    np_brt_arr = np.asarray(brt_arr)
                    np_brt_arr = np.nan_to_num(np_brt_arr)
                    np_brt_arr *= (255.0/np_brt_arr.max())
                    for s in range(0,math.floor(shot/self.input_nb)):
                        self.img_sets.append(np_brt_arr[:,:,s*self.input_nb:(s+1)*self.input_nb])
                        self.speedplots.append(np.asarray(vz_from_wk))
    
        if usage == 'test':
            self.img_sets = self.img_sets[math.floor(len(self.img_sets)*0.8):]
        elif usage == 'validation':
            self.img_sets = self.img_sets[math.floor(len(self.img_sets)*0.8*0.8):math.floor(len(img_sets)*0.8)]
        elif usage == 'train':
            self.training = True
            self.img_sets = self.img_sets[:math.floor(len(self.img_sets)*0.8*0.8)]
        elif usage == 'all':
            self.training = True
        else:
            raise ValueError("Wrong usage specified; Only train,validation,test or all")
    
    def __len__(self):
        return len(self.img_sets)
    
    def __getitem__(self, index):
        np_images = np.zeros((self.input_nb, self.img_sets[index].shape[1], self.img_sets[index].shape[0]))
        for img in range(0,self.img_sets[index].shape[2]):
            np_images[img,:,:] = self.img_sets[index][:, :, img].T
        speedplot = self.speedplots[index]
        return self.transform(np_images,speedplot)
        
    def transform(self,images,speedplot):
        return np.repeat(images[np.newaxis, :, :, :], 3, axis = 0), speedplot

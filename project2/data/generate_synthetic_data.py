import numpy as np
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import sys

from generate_synthetic_data_parameters import *

#Function that assign the speed in Z given the coordinate in R, based on a tanh.
def assign_speed(x):
    adjusted = (x-shift)/(max_x/2)-1
    return magnitude * np.tanh(adjusted*compression)

#plots a speed curve in mathlab for each pixel in canvas, and returns the average speed over the 13 column of the window.
def plot_speed_curve():
    #plotting over the canvas
    x = np.linspace(0,max_x,max_x)
    
    y = assign_speed(x)

    fig = plt.figure()
    fig.suptitle('Speed in Z depending on R', fontsize=20)
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel('Z in pixel/s', loc='top')
    ax.set_xlabel('R in pixel', loc='right')

    plt.plot(x,y, 'r')
    
    
    #calculating the average over 13 columns inside the window to be like the real data
    begining = math.floor((max_x/2) - (window/2))
    end = math.floor((max_x/2) + (window/2))
    x = np.linspace(0,max_x,max_x)[begining:end]
    
    y = assign_speed(x)
    
    interval = math.floor(window/final_size_x)
    
    columns = []
    for i in range(0,13):
        columns.append(i*interval)
    zero_crossing = np.where(np.diff(np.sign(y)))[0][0] + 1
    columns.append(zero_crossing)
    columns.sort()
    
    speedplot = []
    for i in range(0,len(columns)-1):
        b = columns[i+1]-columns[i]
        speedplot.append(np.divide(y[columns[i]:columns[i+1]].sum(), b, out=np.zeros(1), where=b!=0)[0])
        
    return speedplot

#Generates a square gaussian array of specified size
def create_gaussian_array(size):
    x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size)) 
    dst = np.sqrt(x*x+y*y) 

    # Calculating Gaussian array 
    gauss = intensity * np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) ) 
    return gauss

#Creates a specific amount of gaussian structures, and assign them random position, and speed according to the assign_speed funciton
#Returns a list of tuple contianing position as numpy array, the gaussian numpy array, the speed as a numpy array
def create_structures(amount):
    gauss = []
    for s in range(0,amount):
        size = np.random.randint(min_size,max_size)
        struc = create_gaussian_array(size)
        
        if s > amount*(1-negative_ratio):
            struc = struc*-1
        
        x = np.random.randint(0,max_x)
        y = np.random.randint(0,max_y)
        
        gauss.append([np.array([x,y]),struc,np.array([0,assign_speed(x)])])
    return gauss

#Function that takes a list of tuple containing (position, structure, speed), and a time multiplier;
#it will create a new list of tuple like the one it was given, but with positions updated according to the speed of each structures
#with a time factor that represent the time elapsed since last update
def update_structs(structs, time):
    new_structs = structs.copy()
    for p in range(0,len(structs)):
        new_pos = structs[p][0] + structs[p][2]*time
        new_pos[0] = new_pos[0] % max_x
        new_pos[1] = new_pos[1] % max_y
        new_structs[p][0] = new_pos
    return new_structs
    

#This function takes the list of tuples of (position, structure, speed), and add all the structures in the canvas to draw them.
#It returns said canvas.
def draw_structs(structs):
    canvas = np.zeros((max_y,max_x))
    for s in structs:
        pad_x = math.floor((max_x - s[1].shape[0])/2) + 1
        pad_y = math.floor((max_y - s[1].shape[1])/2) + 1
        padded = np.pad(s[1],((pad_x,pad_x),(pad_y,pad_y)), 'constant')
        padded = padded[0:max_x,0:max_y]
        
        roll_x =  math.floor(s[0][0] - (max_x/2))
        roll_y =  math.floor(s[0][1] - (max_y/2))
        if roll_x > 0:
            rolled = np.pad(padded,((0,0),(roll_x,0)), mode='constant')[:, :-roll_x]
        else:
            rolled = np.pad(padded,((0,0),(0,-roll_x)), mode='constant')[:, -roll_x:]
        rolled = np.roll(rolled,-roll_y, axis = 0)
        canvas += rolled
    return canvas

#This function takes an array, and smaller dimensions; it then creates a new array with the dimensions, 
#and averages the given array to fit within these new dimensions
def downsize(array,newdim_x,newdim_y):
    olddim = array.shape[0]
    return array.reshape([newdim_x, olddim//newdim_x, newdim_y, olddim//newdim_y]).mean(3).mean(1)

#This function takes the canvas numpy array, makes sure it is 8-bit and convert it to PIL image
def draw(canvas):
    canvas = canvas * 255
    canvas[canvas > 255] = 255
    canvas[canvas < 0] = 0
    return Image.fromarray(np.array(canvas,dtype=np.uint8))

#This function extract the window ffrom the center of the canvas
def extract_middle(array):
    size = window
    oldsize = array.shape[0]
    begining = math.floor((oldsize/2) - (size/2))
    end = math.floor((oldsize/2) + (size/2))
    return array[begining:end, begining:end]

#This function saves the image array as a 10 fps video with the matching dimensions under the specified name
def save_video(img_array, name, size_x, size_y):
    video = cv2.VideoWriter(name, 0, 10, (size_x,size_y))

    for image in img_array:
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        video.write(open_cv_image)

    cv2.destroyAllWindows()
    video.release()
    
#This function saves the canvas or final window at the path under the specified name
def save_to_folder(array,path,name):
    if name == "full_canvas.avi":
        save_video(array,os.path.join(path , name),max_x,max_y)
    if name == "resized_window.avi":
        save_video(array,os.path.join(path , name),final_size_x,final_size_y)


def main():
    print("Creating gaussian structures...")
    structs = create_structures(amount)
    
    root = "data"
    if not os.path.exists(root):
        os.mkdir(root)
    #Finds a spot to save the data
    num = 0
    name = "synthetic_data_"
    while os.path.exists(os.path.join(root,name + str(num))):
        num += 1
    print("Saving in folder:" + name + str(num))
    path = os.path.join(root,name + str(num))
    os.mkdir(path)
    #Saves the speedplot and the labels
    speedplot = plot_speed_curve()
    plt.savefig(os.path.join(path,"speed_plot.png"))
    np.savetxt(os.path.join(path,"speedplot.csv"), speedplot, delimiter=",")
    #Saves the settings
    f = open(os.path.join(path,"settings.txt"), 'w')
    for t in settings:
        line = ' '.join(str(x) for x in t)
        f.write(line + '\n')
    f.close()
    
    #Now it wil iterate and generate the data
    
    canvas = draw_structs(structs)
    image = draw(canvas)
    resized = draw(downsize(extract_middle(canvas),final_size_y,final_size_x))
    

    if save_video_canvas:
        images = [image]
    if save_video_output:
        resizeds = [resized]
        
    print("Using seed : ", seed)
    print("Using shift : ", shift)
    
    print("Calculating all iterations...")
    for i in range(0,iterations):
        if i % 100 == 0:
            print("Iteration: " + str(i))
        resized.save(os.path.join(path, "frame_" + str(i) + ".png"))
        structs = update_structs(structs, 1)
        canvas = draw_structs(structs)
        resized = draw(downsize(extract_middle(canvas),final_size_y,final_size_x))
        if save_video_canvas:
            images.append(draw(canvas))
        if save_video_output:
            resizeds.append(resized)
    if save_video_canvas:
        save_to_folder(images,path,"full_canvas.avi")
    if save_video_output:
        save_to_folder(resizeds,path,"resized_window.avi")
    
    
if __name__ == '__main__':
    print("Initialization...")
    main()

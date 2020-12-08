import numpy as np
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import sys


#Settings 
#Save full size video (can be a lot of data)
save_video_canvas = False
#Seed for blob size and position
seed = 0
np.random.seed(seed)
#size of canvas
max_x = 480    
max_y = 480
#window which is captured from main canvas to do the images
window = max_x / 2
final_size_x = 12
final_size_y = 10
#amount of gaussian structures to create
amount = 66
#size of the gaussian structures
min_size = 200
max_size = min(max_x,max_y)
#how many negative gaussian array, as a ratio of total arrays created
negative_ratio = 0.5
#parameters for gaussian structures
sigma = 0.2
muu = 0.000
#speedplot parameters for tanh
shift = 0    #shift the center of the tanh, in pixels
magnitude = 10     #multiplies tanh; tanh originally goes from -1 to 1, so now from -magnitude to magnitude
compression = 3    #dictates the shape of tanh; higher number means it goes more quickly to 1 or -1 (it's more compressed at the center)
#number of iterations
iterations = 200

settings = (("seed",seed),("max_x",max_x),("max_y",max_y),("window",window),("final_size_x",final_size_x),("final_size_y",final_size_y),
("amount",amount),("min_size",min_size),("max_size",max_size),("negative_ratio",negative_ratio),("sigma",sigma),("muu",muu),
("shift",shift),("magnitude",magnitude),("compression",compression),("iterations",iterations))

def assign_speed(x, max_x):
    adjusted = (x-shift)/(max_x/2)-1
    return magnitude * np.tanh(adjusted*compression)

def plot_speed_curve():
    x = np.linspace(0,max_x,max_x)
    
    y = assign_speed(x,max_x)

    # setting the axes at the centre
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

    # plot the function
    plt.plot(x,y, 'r')
    interval = math.floor(max_x/final_size_x)
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
    
def create_gaussian_array(size):
    x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size)) 
    dst = np.sqrt(x*x+y*y) 

    # Calculating Gaussian array 
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) ) 
    return gauss

def create_structures(max_x, max_y, amount):
    gauss = []
    for s in range(0,amount):
        size = np.random.randint(min_size,max_size)
        struc = create_gaussian_array(size)
        
        if s > amount*negative_ratio:
            struc = struc*-1
        
        x = np.random.randint(0,max_x)
        y = np.random.randint(0,max_y)
        
        gauss.append([np.array([x,y]),struc,np.array([0,assign_speed(x,max_x)])])
    return gauss

def update_structs(structs, time, max_x, max_y):
    new_structs = structs.copy()
    for p in range(0,len(structs)):
        new_pos = structs[p][0] + structs[p][2]*time
        new_pos[0] = new_pos[0] % max_x
        new_pos[1] = new_pos[1] % max_y
        new_structs[p][0] = new_pos
    return new_structs

def draw_structs(structs,max_x,max_y):
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
        #rolled = np.roll(padded,roll_x, axis = 1)
        rolled = np.roll(rolled,-roll_y, axis = 0)
        canvas += rolled
    return canvas

def downsize(array,newdim_x,newdim_y):
    olddim = array.shape[0]
    return array.reshape([newdim_x, olddim//newdim_x, newdim_y, olddim//newdim_y]).mean(3).mean(1)

def draw(canvas):
    canvas = canvas * 255
    canvas[canvas > 255] = 255
    canvas[canvas < 0] = 0
    return Image.fromarray(np.array(canvas,dtype=np.uint8))

def extract_middle(array, size):
    oldsize = array.shape[0]
    begining = math.floor((oldsize/2) - (size/2))
    end = math.floor((oldsize/2) + (size/2))
    return array[begining:end, begining:end]

def save_video(img_array, name, size_x, size_y):
    video = cv2.VideoWriter(name, 0, 10, (size_x,size_y))

    for image in img_array:
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        video.write(open_cv_image)

    cv2.destroyAllWindows()
    video.release()
    
def save_to_folder(images,resized):
    root = "data"
    if not os.path.exists(root):
        os.mkdir(root)
    num = 0
    name = "synthetic_data_"
    while os.path.exists(os.path.join(root,name + str(num))):
        num += 1
    print("Saving in folder:" + name + str(num))
    path = os.path.join(root,name + str(num))
    os.mkdir(path)
    speedplot = plot_speed_curve()
    plt.savefig(os.path.join(path,"speed_plot.png"))
    np.savetxt(os.path.join(path,"speedplot.csv"), speedplot, delimiter=",")
    f = open(os.path.join(path,"settings.txt"), 'w')
    for t in settings:
        line = ' '.join(str(x) for x in t)
        f.write(line + '\n')
    f.close()
    if save_video_canvas:
        save_video(images,os.path.join(path , "full_canvas.avi"),max_x,max_y)
    save_video(resized,os.path.join(path , "resized_window.avi"),final_size_x,final_size_y)
    for i in range(0,len(resized)):
        resized[i].save(os.path.join(path, "frame_" + str(i) + ".png"))

def main():
    print("Creating gaussian structures...")
    structs = create_structures(max_x, max_y, amount)
    canvas = draw_structs(structs,max_x,max_y)
    
    arrays = [canvas]
    images = [draw(canvas)]
    resized = [draw(downsize(extract_middle(canvas,window),final_size_y,final_size_x))]
    print("Calculating all iterations...")
    for i in range(0,iterations-1):
        structs = update_structs(structs, 1, max_x, max_y)
        canvas = draw_structs(structs,max_x,max_y)
        arrays.append(canvas)
        images.append(draw(canvas))
        resized.append(draw(downsize(extract_middle(canvas,window),final_size_y,final_size_x)))
    print("Saving to folder...")
    save_to_folder(images, resized)
    
if __name__ == '__main__':
    print("Initialization...")
    main()

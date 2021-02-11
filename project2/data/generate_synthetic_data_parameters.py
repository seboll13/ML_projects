import numpy as np

# Data generation parameters

#Settings 
#Save full size video (can be a lot of data)
save_video_canvas = True
#Save video of output
save_video_output = True
#Seed for blob size and position
seed = np.random.randint(100000)
np.random.seed(seed)
#size of canvas
max_x = 480    
max_y = 480
#window which is captured from main canvas to do the images
window = max_x / 2
final_size_x = 12
final_size_y = 10
#amount of gaussian structures to create
amount = 66*4
#size of the gaussian structures
min_size = 200/2
max_size = min(max_x,max_y)/2
#how many negative gaussian array, as a ratio of total arrays created
negative_ratio = 0.3    #between 0 and 1
#parameters for gaussian structures
sigma = 0.2
muu = 0.000
intensity = 0.5 #dims (<1) or augment (>1) the gaussians structures
#speedplot parameters for tanh
shift = np.random.randint(-50,50)    #shift the center of the tanh, in pixels
magnitude = 7     #multiplies tanh; tanh originally goes from -1 to 1, so now from -magnitude to magnitude
compression = np.random.randint(3,6)    #dictates the shape of tanh; higher number means it goes more quickly to 1 or -1 (it's more compressed at the center)
#number of iterations, or frames
iterations = 2000

#list of settings to print to a file
settings = (("seed",seed),("max_x",max_x),("max_y",max_y),("window",window),("final_size_x",final_size_x),("final_size_y",final_size_y),
("amount",amount),("min_size",min_size),("max_size",max_size),("negative_ratio",negative_ratio),("sigma",sigma),("muu",muu),
("shift",shift),("magnitude",magnitude),("compression",compression),("iterations",iterations))
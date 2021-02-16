# Dataloader parameters

#Settings for the behavior of the dataloader: 
#first ratio that will be applied to all data
ratio_train_test = 0.8 
#second ratio; it applied to datapoints that aren't in the testing set
ratio_train_validation = 0.8 
#seed to make reproducible randomness
seed = 1 
#this is the folder inside which the synthetic set's folders should be placed in order for the dataloader to see them
root = 'data/data'

speedplot_rescale_min = -10
speedplot_rescale_max = 10
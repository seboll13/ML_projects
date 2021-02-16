# 2020 EPFL Machine Learning - Project 2 structure

## Structure description
- The current folder contains the following sub-folders:
  - ```architectures```, which contains all the source code of the architectures the we have imported and used for the project.
  - ```data``` in which all scripts required to generate data are, with the data loaders and another sub-folder in which all images generated will be stored.
  - ```report``` in which our report tex and pdf files are stored along with the bib file and the images used in the report.
  - ```models``` in which the models used will be stored when running the main python code.
- It also contains the main python programs ```run.py``` and ```test.py```, along with two other scripts ```display_losses.py``` and ```display_test_results.py``` that we created to generate plots of the data we trained for an easier visual representation.

## Parameters description
- The main process is executed by the command `python run.py`, and the parameters used are stored in `run_parameters.py`. It is used to train a new model. It will generate a folder `[architecture]_[id]` in ```models/``` containing the model, the parameters used and the results (losses). Here are the descriptions of the settings:
  ### Model initialisation parameters 
  - ```num_classes```: number of values the model has to predict. It matches with the number of columns to predict.
  - ```model_names```: contains the names of the three architectures, to simplify switching one architecture to another.
  - ```model_name```: the name of the architecture that will be trained and/or tested.
  
  ### Dataloader parameters
  - ```select_datalaoder```: Selects the dataset to be used (synthetic, real or mixed) to train or test.
  - ```nb_of_input_images```: number of frames to be considered as a single datapoint. We have chosen 2000 images.
  - ```num_train_workers```: number of workers for the data loader of the training set.
  - ```num_valid_workers```: number of workers for the data loader of the validation set.
  - ```normalize```: normalizes labels (= speeds) to be between -10 and 10
   
  ### Hardware parameter
  - ```use_cuda```: train the model on GPU if True (and if cuda is available on the hardware), otherwise train on CPU.
  
  ### Training parameters
  - ```batch_size```: number of inputs to be trained simultaneously. 
  - ```num_epochs```: total number of training iterations. We have chosen 30 epochs.
  - ```lr```: learning rate
  - ```gamma```: ratio at which the learning rate decreases
  - ```step_size```: number of epochs after which the learning decreases 
  - ```seed```: seed for random functions in PyTorch
  
  ### Saving parameters
  - ```settings```: save all parameters in settings.txt for future analysis
  - ```models_folder```: folder containing the models and their results
  
  
- The script `test.py` is used to test a pretrained model on unseen data. The command is `python test.py`, and the parameters used are stored in `test_parameters.py`. It will generate the testing results in the folder of the model, as well as graphs of some predictions on the testing set (of the real dataset). Here are the descriptions of the settings:
  ### Dataloader parameters
  - ```nb_of_input_images```: number of frames to be considered as a single datapoint. We have chosen 2000 images.
  - ```normalize```: normalizes labels (= speeds) to be between -10 and 10
   
  ### Hardware parameter
  - ```use_cuda```: train the model on GPU if True (and if cuda is available on the hardware), otherwise train on CPU.
  
  ### Training parameters
  - ```test_batch_size```: number of inputs to be tested simultaneously. 
  - ```seed```: seed for random functions in PyTorch
  
  ### Saving parameters
  - ```models_folder```: folder containing the models and their results
  - ```load_folder```: folder to load the model from

## Data related scripts parameters
- To generate synthetic data, run the script `generate_synthetic_data.py`, in `data/`. It takes its parameters from `generate_synthetic_data_parameters.py` It will generate a new folder `synthetic_data_[id]` in `data/data/`, inside of which will be: All the frames generated as per the settings, a speedplot of the whole canvas, and a list of settings for reproductibility purposes. Additionnaly, it can save a video of the whole canvas through all the iteration, and a video of all the frames, if you set the corresponding options.

- Here is a description of all the settings at the beginning of the file:
  - `save_video_canvas` : Save full size canvas video if set to true
  - `save_video_output` : Save the frames in a video if set to true
  - `seed` : Seed for blob size and position
  - `max_x` : Horizontal size of the canvas  
  - `max_x` : Vertical size of the canvas 
  - `window` : size of the square window which is captured from main canvas to do the images
  - `final_size_x` : Horizontal final size of the output frames, to match with G.P.I data
  - `final_size_y` : Vertical final size of the output frames, to match with G.P.I data
  - `amount` : Amount of gaussian structures to create
  - `min_size` : Minimum size of the gaussian structures, in pixels
  - `max_size` : Maximum size of the gaussian structures, in pixels
  - `iterations` : Number of frames the script will generate

  ### Parameters for the gaussian structures
  - `sigma` : sets the sigma of the gaussian
  - `muu` : sets the muu of the gaussian
  - `intensity` : dims (<1) or augment (>1) the gaussians structure's brightness
  - `negative_ratio` : Ratio of negative structures, between 0 an 1

  ### Speedplot parameters for tanh
  - `shift` : Shift the center of the tanh, in pixels
  - `magnitude` : Multiplies the tanh; tanh originally goes from -1 to 1, so now from -magnitude to magnitude
  - `compression` : Dictates the shape of tanh; higher number means it goes more quickly to 1 or -1 (it's more compressed at the center)
  
If a model was trained, a new folder with the model name is generated in `models/` containing the results and the model `.pth`.<br>
If a model was tested, the results are saved in the folder the tested models was in.

## Data visualization scripts
- Once the model has been trained, the display scripts can be used as follows:
  - The script accepts one parameter, the name of the folder where the model parameters have been added. To launch it, just run: ```python display_losses.py model_name``` or ```python display_test_results.py model_name```, where ```model_name``` is the name of the folder containing the results of the model we want to visualize.
  - ```display_losses.py``` displays the training and validations plots over the epochs.
  - ```display_test_results``` outputs the mean and standard deviation test losses, as well as a plot for each prediction and its label.
  - The png image will then appear in the selected models corresponding folder.
  sed will be stored when running the main python code.
- It also contains the main python program ```run.py``` along with two other scripts ```display_losses.py``` and ```display_test_results.py``` that we created to generate plots of the data we trained for an easier visual representation.

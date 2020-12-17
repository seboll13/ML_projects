# 2020 EPFL Machine Learning - Project 2 structure
- The current folder contains the following sub-folders:
  - ```architecture``` in which all source code of the architectures have been imported and implemented.
  - ```data``` in which all scripts required to generate data are and another sub-folder in which all images generated will be stored.
  - ```report``` in which our report tex and pdf files are stored along with the bib file and the images used in the report.
  - ```models``` in which the models used will be stored when running the main python code.
- It also contains the main python program ```run.py``` along with two other scripts ```display_losses.py``` and ```display_test_results.py``` that we created to generate plots of the data we trained for an easier visual representation.

- [...] dataloader



- The main process is executed by the command `python run.py`. Here are the settings to take into account:
  #### Model parameters
  - num_classes: number of values the model has to predict. It matches with the number of columns to predict.
  - model_names: contains the names of the three architectures, to simplify switching one architecture to another.
  - model_name: the name of the architecture that will be trained and/or tested.
  
  #### Dataloader parameters
  - train_on_synthetic_data: train the model on the synthetic or real dataset.
  
  
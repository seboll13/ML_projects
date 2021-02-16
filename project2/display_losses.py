import matplotlib.pyplot as plt
import time
import sys
import os

# parameters
folder = 'models'
sets = ['train', 'valid']

nb_input_images, batch_size, nb_epochs, gamma, lr = 0,0,0,'',''

def main(model):
    trains, valids = [], []
    for i in range(len(sets)):
        # get path for current file
        path = os.path.join(folder, model, "losses_" + sets[i] + ".txt")
        if not os.path.exists(path):
            sys.exit('Sorry, this folder does not exist.')
        with open(path, 'r') as f:
            for line in f:
                if line == 'nan':
                    break
                # fill in the tables
                if i == 0:
                    trains.append(float(line[:-1]))
                else:
                    valids.append(float(line[:-1]))
    
    # Handle settings file
    print("Generating training and validation graphs for model: " + model + "...")
    settings_path = os.path.join(folder, model, 'settings.txt')
    with open(settings_path, 'r') as f:
        lines = [line.rstrip() for line in f]
        nb_input_images = int(lines[2].split(' ')[1])
        batch_size = int(lines[5].split(' ')[1])
        nb_epochs = int(lines[6].split(' ')[1])
        gamma = lines[7].split(' ')[1]
        lr = lines[8].split(' ')[1]

    params = model + 's' + str(nb_input_images) + '_' + lr + '_' + gamma

    # plot the two arrays of losses
    fig = plt.figure()
    plt.plot(trains, label='training_losses')
    plt.plot(valids, label='validation_losses')
    plt.legend()
    plt.ylim(0, 20)

    plt.savefig(os.path.join(folder,model,'plot_losses_train_valid.png'))
    plt.close(fig)
    
if __name__ == '__main__':
    main(sys.argv[1])
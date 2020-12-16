import matplotlib.pyplot as plt
import time
import os

folder = 'models'
model = 'resnet_3d_11'
sets = ['train', 'valid']
epochs = 30
nb_input_images = 100
lrs = [0.005, 0.01, 0.05, 0.1]
gammas = [0.05, 0.1, 0.2, 0.5]
if __name__ == '__main__':
    for lr in lrs:
        for gamma in gammas:
            trains, valids = [], []
            for i in range(len(sets)):
                path = os.path.join(folder,model,"losses_" + sets[i] + ".txt")
                with open(path, 'r') as f:
                    for line in f:
                        if line == 'nan':
                            break
                        if i == 0:
                            trains.append(float(line[:-1]))
                        else:
                            valids.append(float(line[:-1]))
            if len(trains) < 30 or len(valids) < 30:
                break
            params = model + 's' + str(nb_input_images) + '_' + str(lr) + '_' + str(gamma)
            plt.figure()
            plt.plot(trains, label='training_losses')
            plt.plot(valids, label='validation_losses')
            plt.legend()
            plt.ylim(0, 20)

            
            plt.savefig(os.path.join(folder,model,'plot_losses_train_valid.png'))
            

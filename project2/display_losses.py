import matplotlib.pyplot as plt
import time
import os

folder = 'results/'
model = 'resnet_2_1d_'
sets = ['train_', 'valid_']
epochs = 30
nb_input_images = 100
lrs = [0.005, 0.01, 0.05, 0.1]
gammas = [0.05, 0.1, 0.2, 0.5]
if __name__ == '__main__':
    for lr in lrs:
        for gamma in gammas:
            trains, valids = [], []
            for i in range(len(sets)):
                path = folder + model + sets[i] + str(epochs) + '_s' + str(nb_input_images) + '_'\
                + str(lr) + '_' + str(gamma) + '.txt'
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
            plt.ylim(0, 10)

            base_dir = folder + 'plots'
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            os.chdir(base_dir)
            plt.savefig('plot_' + params + '.png')
            os.chdir('../..')

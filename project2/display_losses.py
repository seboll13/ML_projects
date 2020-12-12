import matplotlib.pyplot as plt
import time

folder = 'results/'
model = 'resnet_2_1d_'
sets = ['train_', 'valid_']
epochs = 30
lr = 0.01

if __name__ == '__main__':
    trains, valids = [], []
    for i in range(len(sets)):
        path = folder + model + sets[i] + str(epochs) + '_' + str(lr) + '.txt'
        with open(path, 'r') as f:
            for line in f:
                if i == 0:
                    trains.append(float(line[:-1]))
                else:
                    valids.append(float(line[:-1]))

    plt.figure()
    plt.plot(trains, label='training_losses')
    plt.plot(valids, label='validation_losses')
    plt.legend()
    plt.ylim(0, 10)
    plt.show()

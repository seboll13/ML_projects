import matplotlib.pyplot as plt
import os.path
import numpy as np

# parameters
folder = 'models'
model = 'resnet_3d_43'
path = os.path.join(folder,model,"test_results.txt") 
path_loss = os.path.join(folder,model,"losses_test.txt") 
lines = []
labels = []
outputs = []
loss = []

if __name__ == '__main__':
    # go through each array of results
    with open(path,"r") as f:
        for line in f:
            l = [float(x) for x in line[1:-3].split(", ")]
            lines.append(l)

    # add losses to its respective array
    with open(path_loss,"r") as f_l:
        for line in f_l:
            loss.append(float(line))

    loss = np.array(loss)
    labels, outputs = lines[::2], lines[1::2]

    # compute and display useful calculations
    for i in range(len(loss)):
        print('Average loss for this prediction :', loss[i])

    print('Mean :', np.mean(loss))
    print('Std :', np.std(loss))

    # handle plotting part
    for i in range(len(loss)):
        plt.figure()
        line_t, = plt.plot(labels[i], label="target")
        line_o, = plt.plot(outputs[i], label="output")
        axes = plt.gca()
        axes.set_xlim([0,12])
        plt.ylabel('Velocity')
        plt.xlabel('Column')
        plt.legend(handles=[line_t, line_o])

        plt.savefig(os.path.join(folder,model,'plot_test_results_' + str(i) + '.png'))
    
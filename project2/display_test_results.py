import matplotlib.pyplot as plt
import os.path
import sys
import numpy as np


def main(model):
    # parameters
    folder = 'models'

    lines = []
    labels = []
    outputs = []
    loss = []
    # Display result every [step] predictions (displaying all results might generate too many .png than needed)
    step = 50
    
    if not model :
        sys.exit('Please input the folder name containing the test results to be plotted')

    if not os.path.exists(os.path.join(folder,model)):
        sys.exit('This folder does not exist in ' + folder)

    print("Generating testing graphs for model : " + model + "...")
    path = os.path.join(folder,model,"test_results.txt") 
    path_loss = os.path.join(folder,model,"losses_test.txt") 
    
    
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

    print('Mean :', np.mean(loss))
    print('Std :', np.std(loss))

    # handle plotting part
    print("Plotting some predictions...")
    plt.ioff()
    for i in range(len(loss)):
        if i%step == 0:
            fig = plt.figure()
            line_t, = plt.plot(labels[i], label="target")
            line_o, = plt.plot(outputs[i], label="output")
            axes = plt.gca()
            axes.set_xlim([0,12])
            plt.ylabel('Velocity')
            plt.xlabel('Column')
            plt.legend(handles=[line_t, line_o])

            plt.savefig(os.path.join(folder,model,'plot_test_results_' + str(i) + '.png'))
            plt.close(fig)
    
if __name__ == '__main__':
    main(str(sys.argv[1]))
    
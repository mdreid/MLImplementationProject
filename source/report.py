import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataSize = np.array([100, 500, 1000, 5000, 10000])
    prePruningAccuracy = np.array([0.3384, 0.3996, 0.4326, 0.3597, 0.295])
    postPruningAccuracy = np.array([0.3643, 0.404, 0.4324, 0.3606, 0.2951])
    postPruningWithKdAccuracy = [0.4362, 0.6368, 0.7135, 0.9016, 0.9416]
    chanceAccuracy = [0.1 for i in range(len(dataSize))] 
    # chanceAccuracy = [0.1, 0.1, 0.1, 0.1, 0.1]
    fig = plt.figure(1)
    plt.title("Accuracy for different training size")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Size")
    plt.plot(dataSize, postPruningAccuracy, 'ro--', label='Without KD tree')
    plt.plot(dataSize, postPruningWithKdAccuracy, 'g^--', label='With KD tree')
    plt.plot(dataSize, chanceAccuracy, 'bs--', label='Chance')
    # plt.plot(dataSize, prePruningAccuracy, 'bs--', label='Pre-pruning')
    plt.ylim(ymin=0)
    plt.legend(loc="best")
    plt.show()
    
    # save image to file
    fig.savefig('learning_curve.png')

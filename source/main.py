import struct
import os
import numpy as np
import cPickle as pickle
from NearestNeighborKD import NearestNeighborKD as nn

# INPUT FILES
DATASET_FOLDERS = '.' + os.sep + 'datasets'
TRAIN_IMAGES_FILE = DATASET_FOLDERS + os.sep + 'train-images-idx3-ubyte'
TRAIN_LABEL_FILE = DATASET_FOLDERS + os.sep + 'train-labels-idx1-ubyte'

SMALL_SIZE = 1000
TRAIN_IMAGES_FILE_SMALL = TRAIN_IMAGES_FILE + '_small_' + str(SMALL_SIZE)
TRAIN_LABEL_FILE_SMALL = TRAIN_LABEL_FILE + '_small_' + str(SMALL_SIZE)

# Set to 1 to use a smaller data set
USE_SMALL_SIZE = 1

# Set to 1 if you change the SMALL_SIZE to create a smaller set of data, you only have to do this once
NEW_SMALL_SIZE = 0


# Read the data matrix whose dimensions are in "dimensions" array
# Recursion!!!
def readDimension(f, dimensions, currentDimension, dataTypeSize, dataTypeFormat):
    data = []
    dataLength = dimensions[currentDimension]
    # print("Data length: " + str(dataLength))
    if currentDimension == len(dimensions) - 1:
        dataRaw = f.read(dataTypeSize * dataLength)
        offSet = 0
        nextOffSet = dataTypeSize
        for i in range(dataLength):
            data.append(struct.unpack(dataTypeFormat, dataRaw[offSet:nextOffSet])[0])
            offSet = nextOffSet
            nextOffSet += dataTypeSize
        return data
    else:
        for i in range(dataLength):
            data.append(readDimension(f, dimensions, currentDimension + 1, dataTypeSize, dataTypeFormat))
        return data


def readIDX(fileName):
    f = None
    try:
        f = open(fileName, 'rb')
        # first 2 bytes are always 0, ignore them
        f.read(2)

        # data type
        dataType = f.read(1)

        # number of bytes / struct format of data type, assuming MSB format
        # more at https://docs.python.org/3/library/struct.html
        options = {
            b'\x08': [1, 'B'],  # unsigned byte
            b'\x09': [1, 'b'],  # signed byte
            b'\x0B': [2, '>h'],  # short (2 bytes)
            b'\x0C': [4, '>i'],  # int (4 bytes)
            b'\x0D': [4, '>f'],  # float (4 bytes)
            b'\x0E': [8, '>d'],  # double (8bytes)
        }
        dataTypeSize = options[dataType][0]
        dataTypeFormat = options[dataType][1]

        # number of dimensions
        d = struct.unpack('B', f.read(1))[0]
        dimensions = []
        for i in range(d):
            dimensions.append(struct.unpack('>i', f.read(4))[0])

        data = readDimension(f, dimensions, 0, dataTypeSize, dataTypeFormat)
        matrix = np.array(data)
        if len(dimensions) > 1:  # these are training features
            matrix = matrix.reshape(dimensions[0], dimensions[1] * dimensions[2])
    finally:
        if f is not None:
            f.close()

    return matrix


def loadData():
    if USE_SMALL_SIZE == 1:
        if NEW_SMALL_SIZE == 1:
            bigTrain = readIDX(TRAIN_IMAGES_FILE)
            bigLabel = readIDX(TRAIN_LABEL_FILE)

            smallTrain = bigTrain[0:SMALL_SIZE]
            smallLabel = bigLabel[0:SMALL_SIZE]
            # dump to files for faster read later
            pickle.dump(smallTrain, open(TRAIN_IMAGES_FILE_SMALL, "wb"))
            pickle.dump(smallLabel, open(TRAIN_LABEL_FILE_SMALL, "wb"))
        else:
            smallTrain = pickle.load(open(TRAIN_IMAGES_FILE_SMALL, "rb"))
            smallLabel = pickle.load(open(TRAIN_LABEL_FILE_SMALL, "rb"))
        return smallTrain, smallLabel
    else:
        bigTrain = readIDX(TRAIN_IMAGES_FILE)
        bigLabel = readIDX(TRAIN_LABEL_FILE)
        return bigTrain, bigLabel


if __name__ == "__main__":
    train, label = loadData()
    knn = nn()
    knn.fit(train, label, None)

    print("Test on the training data")
    for i in range(100):
        point, predictlabel, squared_distance = knn.predict(train[i])
        print("Training data {0} - Predict label {1} - Correct label {2}".format(i, predictlabel, label[i]))


    print("Test Exhaustive search on the training data")
    for i in range(100):
        point, predictlabel, squared_distance = knn.predict(train[i], False)
        print("Training data {0} - Predict label {1} - Correct label {2}".format(i, predictlabel, label[i]))

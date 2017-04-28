import timeit
import struct
import os
import numpy as np
import sys
import cPickle as pickle
from sklearn.decomposition import IncrementalPCA

# INPUT FILES
DATASET_FOLDERS = '.' + os.sep + 'datasets'
TRAIN_IMAGES_FILE = DATASET_FOLDERS + os.sep + 'train-images-idx3-ubyte'
TRAIN_LABEL_FILE = DATASET_FOLDERS + os.sep + 'train-labels-idx1-ubyte'
TEST_IMAGES_FILE = DATASET_FOLDERS + os.sep + 't10k-images.idx3-ubyte'
TEST_LABEL_FILE = DATASET_FOLDERS + os.sep + 't10k-labels.idx1-ubyte'

TRAIN_PCA_FILE = DATASET_FOLDERS + os.sep + 't10k-labels.idx1-ubyte_PCA'

# Read the data matrix whose dimensions are in "dimensions" array
# Recursion!!!
def readDimension(f, dimensions, currentDimension, dataTypeSize, dataTypeFormat):
    data = []
    dataLength = dimensions[currentDimension]
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


def readIDX(fileName, num_examples=0):
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

        if num_examples > 0:
            dimensions[0] = num_examples

        print(dimensions)
        data = readDimension(f, dimensions, 0, dataTypeSize, dataTypeFormat)
        matrix = np.array(data)
        if (len(dimensions) > 1): # these are training features
            matrix = matrix.reshape(dimensions[0], dimensions[1]*dimensions[2])
            ipca = IncrementalPCA(n_components=100)
            print(matrix.shape)
            ipca.fit(matrix)
            matrix = ipca.transform(matrix)
            print(matrix.shape)
            print(ipca)
    finally:
        if f is not None:
            f.close()

    return matrix

def readPCA(num_examples):
    matrix = pickle.load(open(TRAIN_PCA_FILE, "rb"))
    return matrix[:num_examples]

if __name__ == "__main__":
    start = timeit.default_timer()

    REDUCED_PCA = int(sys.argv[1])
    if REDUCED_PCA == 1:
        trainData = readIDX(TRAIN_IMAGES_FILE)
        pickle.dump(trainData, open(TRAIN_PCA_FILE, "wb"), protocol=2)
        stop = timeit.default_timer()
        print("Read and PCA reduce " + TRAIN_IMAGES_FILE + " time: " + str(stop - start))
    else:
        trainData = pickle.load(open(TRAIN_PCA_FILE, "rb"))
        stop = timeit.default_timer()
        print("Read the reduced version " + TRAIN_IMAGES_FILE + " time: " + str(stop - start))
    
    start = timeit.default_timer()
    labelData = readIDX(TRAIN_LABEL_FILE)
    stop = timeit.default_timer()
    print("Read " + TRAIN_LABEL_FILE + " time: " + str(stop - start))
   

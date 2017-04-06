import timeit
import struct
import os

# INPUT FILES
DATASET_FOLDERS = '.' + os.sep + 'datasets'
TRAIN_IMAGES_FILE = DATASET_FOLDERS + os.sep + 'train-images-idx3-ubyte'
TRAIN_LABEL_FILE = DATASET_FOLDERS + os.sep + 'train-labels-idx1-ubyte'


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


def readIDX(fileName):
    f = None
    try:
        f = open(fileName, 'rb')
        # first 2 bytes are always 0, ignore them
        f.read(2)

        # data type
        dataType = f.read(1)[0]

        # number of bytes / struct format of data type, assuming MSB format
        # more at https://docs.python.org/3/library/struct.html
        options = {
            '\x08': [1, 'B'],  # unsigned byte
            '\x09': [1, 'b'],  # signed byte
            '\x0B': [2, '>h'],  # short (2 bytes)
            '\x0C': [4, '>i'],  # int (4 bytes)
            '\x0D': [4, '>f'],  # float (4 bytes)
            '\x0E': [8, '>d'],  # double (8bytes)
        }
        dataTypeSize = options[dataType][0]
        dataTypeFormat = options[dataType][1]

        # number of dimensions
        d = struct.unpack('B', f.read(1))[0]
        dimensions = []
        for i in range(d):
            dimensions.append(struct.unpack('>i', f.read(4))[0])

        data = readDimension(f, dimensions, 0, dataTypeSize, dataTypeFormat)
    finally:
        if f is not None:
            f.close()

    return data


start = timeit.default_timer()
trainData = readIDX(TRAIN_IMAGES_FILE)
stop = timeit.default_timer()
print("Read " + TRAIN_IMAGES_FILE + " time: " + str(stop - start))

start = timeit.default_timer()
labelData = readIDX(TRAIN_LABEL_FILE)
stop = timeit.default_timer()
print("Read " + TRAIN_LABEL_FILE + " time: " + str(stop - start))


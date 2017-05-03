"""
    Test the speed of NearestNeighbor using KD-Tree versus Exhaustive Search
"""
import sys
import read_input
import timeit
from NearestNeighborKD import NearestNeighborKD

def printTime(message, s): 
    milliseconds = (s*1000) % 1000
    s = int(s)
    seconds = s % 60
    s = s/60
    minutes = s % 60
    hours = s/60
    print("{0}: {1}:{2}:{3}'{4}\"".format(message, hours, minutes, seconds, milliseconds))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Invalid parameters. Should be: speed_test <num_examples> <num_testing>")

    num_examples = int(sys.argv[1])
    num_testing = int(sys.argv[2])
    # print("Number of training examples: {0}".format(num_examples))
    # print("Number of testing examples: {0}".format(num_testing))
    train_data, test_data = read_input.readInputData(read_input.TRAIN_IMAGES_FILE, read_input.TEST_IMAGES_FILE, num_examples, 0, num_testing)
    train_label= read_input.readIDX(read_input.TRAIN_LABEL_FILE, num_examples)
    test_label = read_input.readIDX(read_input.TEST_LABEL_FILE, num_testing)

    nn = NearestNeighborKD()

    print("**********USING KD-TREE**********")
    start = timeit.default_timer()
    nn.fit(train_data, train_label, None)
    end = timeit.default_timer()
    printTime("Training time for {0} instances".format(num_examples), end-start)

    start = timeit.default_timer()
    correct = 0
    for i in range(len(test_data)):
        point, predict_label, distance = nn.predict(test_data[i])
        # print("Test data {0} - Predict label {1} - Correct label {2}".format(i, predict_label, test_label[i]))
        if predict_label == test_label[i]:
            correct += 1
    end = timeit.default_timer()
    print("Accuracy: {0}".format(float(correct) / len(test_data)))
    printTime("Testing time for {0} instances".format(len(test_data)), end-start)

    print("")
    print("**********USING EXHAUSTIVE-SEARCH**********")
    start = timeit.default_timer()
    nn.fit(train_data, train_label, None, False)
    end = timeit.default_timer()
    printTime("Training time for {0} instances".format(num_examples), end - start)
    correct = 0
    start = timeit.default_timer()
    for i in range(len(test_data)):
        point, predict_label, distance = nn.predict(test_data[i], False)
        # print("Test data {0} - Predict label {1} - Correct label {2}".format(i, predict_label, test_label[i]))
        if predict_label == test_label[i]:
            correct += 1
    end = timeit.default_timer()
    print("Accuracy: {0}".format(float(correct) / len(test_data)))
    printTime("Testing time for {0} instances".format(len(test_data)), end-start)


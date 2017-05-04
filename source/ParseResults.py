import sys
import string

def throwAway(f):
    for line in f:
        if "Prediction:" not in line:
            continue
        else:
            break

def computePrecisionRecall(f, header):
    print(header)
    true_positive = dict()
    false_positive = dict()

    total_predicted = dict()
    total_actual = dict()

    for i in range(0, 9+1):
        true_positive[i] = 0
        false_positive[i] = 0
        total_predicted[i] = 0
        total_actual[i] = 0

    for line in f:
        pieces = line.rstrip('\n').split(' ')
        if "Prediction:" == pieces[0]:
            pred = int(pieces[1])
            actual = int(pieces[3])
            total_predicted[pred] += 1
            total_actual[actual] += 1
            if pred == actual:
                true_positive[pred] += 1
            else:
                false_positive[pred] += 1
        else:
            break

    for i in range(0, 9+1):
        recall = true_positive[i] / float(total_actual[i])
        precision = true_positive[i] / float(total_predicted[i])
        print("Digit: " + str(i) + " Recall: " + str(recall) + " Precision: " + str(precision))


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        throwAway(f)
        computePrecisionRecall(f, "FIRST")
        throwAway(f)
        computePrecisionRecall(f, "SECOND")


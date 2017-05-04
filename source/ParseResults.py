import sys
import string


    
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

    found_prediction = False
    for line in f:
        pieces = line.rstrip('\n').split(' ')
        if "Prediction:" == pieces[0]:
            found_prediction = True
            pred = int(pieces[1])
            actual = int(pieces[3])
            total_predicted[pred] += 1
            total_actual[actual] += 1
            if pred == actual:
                true_positive[pred] += 1
            else:
                false_positive[pred] += 1
        else:
            if found_prediction:
                break

    for i in range(0, 9+1):
        recall = true_positive[i] / float(total_actual[i])
        precision = true_positive[i] / float(total_predicted[i])
        print("Digit: " + str(i) + " Recall: " + str(recall) + " Precision: " + str(precision))


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        computePrecisionRecall(f, "FIRST")
        computePrecisionRecall(f, "SECOND")


import sys
import string
import numpy as np

from tabulate import tabulate

    
def computePrecisionRecall(f, header):

    results = np.zeros((10, 10))
    table_header = ['\'' + str(x) + '\'' for x in range(0, 9+1)]
    table_header.insert(0, ' ')
    #print(table_header)
    #print(results)
    # old stuff
    print(header)
    #print(tabulate(results, tablefmt="latex", floatfmt=".2f"))
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

            # needed to compute precision and recall
            found_prediction = True
            pred = int(pieces[1])
            actual = int(pieces[3])
            total_predicted[pred] += 1
            total_actual[actual] += 1
            if pred == actual:
                true_positive[pred] += 1
            else:
                false_positive[pred] += 1
            #new stuff
            results[pred][actual] += 1
        else:
            if found_prediction:
                break

    pr_header = ["Digit", "Precision", "Recall"]
    pr_data = []
    for i in range(0, 9+1):
        # compute precision and recall for one digit at a time
        recall = true_positive[i] / float(total_actual[i])
        precision = true_positive[i] / float(total_predicted[i])

        entry = [str(i), precision, recall]
        pr_data.append(entry)
        #print("Digit: " + str(i) + " Recall: " + str(recall) + " Precision: " + str(precision))
    print("Result Counts")
    proc_results = results.tolist()
    for i in range(0, 9+1):
        proc_results[i].insert(0, '\'' + str(i) + '\'')
    print(tabulate(proc_results, headers=table_header, tablefmt="latex"))
    print("Precision and Recall Values")
    print(tabulate(pr_data, headers=pr_header, tablefmt="latex"))


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        computePrecisionRecall(f, "Without k-d tree")
        computePrecisionRecall(f, "With k-d tree")


from decimal import *
import pdb
import read_input
import numpy as np
import timeit
import sys

LABEL_VALUES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

class Node:

    def __init__(self, left, right, ft, thresh, isLeaf,level,prediction, parent):
        self.left = left
        self.right = right
        self.feature = ft
        self.threshold = thresh
        self.kd = isLeaf
        self.level = level
        self.prediction = prediction
        self.parent = parent
    
    def setParent(self, parent):
        self.parent = parent

    def __repr__(self):
        if self.kd:
            return "Leaf\n"
        else:
            string = ""
            string += ("Split at " + str(self.threshold) + " on feature " + str(self.feature) + " Level: " + str(self.level) + "\n")
            string += ("==Left==\n")
            if self.left is None:
                string += "None\n"
            else:
                string += str(self.left)
            string += ("==Right==\n")
            if self.right is None:
                string += ("None\n")
            else:
                string += str(self.right)
            return string

def dbg_non_zero(vec):
    for v in vec:
        if int(v) != 0:
            return True
    return False

def findThresholdPoints(ft_vector, label_vector, level):
    if dbg_non_zero(ft_vector) and level >= 6:
        pass
        #pdb.set_trace()
    prev_label = int(label_vector[0])
    thresholds = dict()
    pts = []
    ft_vector_len = len(ft_vector)
    for i in range(1, ft_vector_len):
        if ((int(label_vector[i]) != prev_label) and not (ft_vector[i-1] == ft_vector[i])):
            t = Decimal(ft_vector[i-1] + ft_vector[i]) / Decimal(2)
            thresholds[t] = i-1
            if (i == ft_vector_len-1 and ft_vector[i] == ft_vector[i-1]):
                thresholds[t] = i
            prev_label = int(label_vector[i])


    for k in sorted(thresholds.keys()):
        pts.append((k, thresholds[k]))

    return pts

def log2(p):
    if (p == Decimal('0')):
        return Decimal('0')
    return p.log10()/Decimal("2").log10()

# Info(D)
def calcEntropy(label_vector):
    counts = dict() 
    for label in label_vector:
        str_label = str(label)
        if str_label in counts:
            counts[str_label] += 1
        else:
            counts[str_label] = 1

    total = len(label_vector)

    H = Decimal("0")
    for label in counts:
        prob = (Decimal(counts[label]) / total)
        H += prob * log2(prob)
    
    return H * Decimal("-1")

def dbg_is_sorted(v):
    if len(v) == 0:
        return True
    i = v[0]
    for e in v:
        if e < i:
            return False
        i = e
    return True
        

# Calculates normalized info gain for a feature
def calcInfoGainThreshold(info_D, ft_vector, label_vector, level):
    ft_vector = np.matrix.copy(ft_vector)
    label_vector = np.matrix.copy(label_vector)
    idx = np.argsort(ft_vector, kind='quicksort')
    ft_vector = ft_vector[idx]
    label_vector = label_vector[idx]
    assert(dbg_is_sorted(ft_vector))# and dbg_is_sorted(label_vector))

    thresholds = findThresholdPoints(ft_vector, label_vector, level)

    info_gain = Decimal('-1.0')
    #gain_compare = Decimal('Infinity')
    #split_compare = Decimal('1.0')
    gain_ratio = Decimal('-Infinity')
    length = Decimal(len(label_vector))
    best_threshold = 0
    for t in thresholds:
        t_val = t[0]
        idx = t[1]
        d_i = idx + 1
        info_Di_left = calcEntropy(label_vector[0:d_i])
        info_Di_right = calcEntropy(label_vector[d_i:])
        gain_sum = (d_i * info_Di_left + (length - d_i)*info_Di_right) / length

        gain_dt = info_D - gain_sum

        split_left = (d_i / length) * log2(d_i/length) 
        split_right = ((length - d_i)/length) * log2((length - d_i)/length)
        split_dt = Decimal('-1') * safe_add(split_left + split_right)

        info_gain_tmp = gain_dt / split_dt

        if info_gain_tmp > info_gain:
            if info_gain_tmp == Decimal('1'):
                pass
            info_gain = info_gain_tmp
            best_threshold = t_val

    return info_gain, best_threshold

def safe_add(total):
    if total == Decimal('0'):
        return Decimal('1')
    return total

def calcInfoGain(data, label_vector, level):
    gain_compare = Decimal('0.0')
    info_D = calcEntropy(label_vector)
    if info_D > Decimal('1'):
        pass
    best_threshold = 0
    best_feature = 0
    for i in range(28*28-1, 0, -1):
        ft_vector = data[:, i]
        feat_gain, threshold = calcInfoGainThreshold(info_D, ft_vector, label_vector, level)
        if feat_gain > gain_compare:
            gain_compare = feat_gain
            best_threshold = threshold
            best_feature = i
    return best_feature, best_threshold
 
def is_single_class(labels):
    length = len(labels)
    if (length == 0):
        return True
    label = labels[0]
    for i in range(1, length):
        if label != labels[i]:
            return False
    return True

# checks for base case in which for each feature
# all values are same
def all_features_same(data):
    for i in range(len(data[0])):
        col = data[:,i]
        val = col[0]
        for v in range(1, len(col)):
            if val != v:
                return False
    return True

def splitIndices(data, feat, threshold, length):
    left_indices = []
    right_indices = []
    for i in range(length):
        if Decimal(int(data[i, feat])) <= threshold:
            left_indices.append(i)
        else:
            right_indices.append(i)

    return left_indices, right_indices

def makeSubTree(data, label_vector, level):
    if is_single_class(label_vector) or all_features_same(data):
        counts = np.bincount(label_vector)
        prediction = np.argmax(counts) 
        return Node(None, None, -1, 0, True,level,prediction,None)

    if (level > 7):
        pass
        #sys.exit("Still failing") 
    
    # choose a split
    feat, threshold = calcInfoGain(data, label_vector, level)
    left_indices, right_indices = splitIndices(data, feat, threshold, len(label_vector))    
    data_left = data[left_indices]
    data_right = data[right_indices]
    label_left = label_vector[left_indices]
    label_right = label_vector[right_indices]
 
    left_node = makeSubTree(data_left, label_left, level+1)
    right_node = makeSubTree(data_right, label_right, level+1)

    counts = np.bincount(label_vector)
    prediction = np.argmax(counts)
    curr_node = Node(left_node, right_node, feat, threshold, False, level, prediction, None)
    left_node.setParent(curr_node)
    right_node.setParent(curr_node)
    return curr_node

def makeDT(data, label_vector):
    root = makeSubTree(data, label_vector, 1)
    return root

def isLeaf(node):
    return node.kd

def calcErrorCount(node, label_vector):
    count = 0
    for val in label_vector:
        if val != node.prediction:
            count = count + 1

    return count

def pruneDT(root, data, label_vector):
    # prunes the tree
    # filter data
    # TODO: Return error counts
    if root.kd:
        return calcErrorCount(root, label_vector)
    if len(data) == 0:
        return 0
    feat = root.feature
    threshold = root.threshold
    left_indices, right_indices = splitIndices(data, feat, threshold, len(label_vector))

    if len(left_indices) == 0 or len(right_indices) == 0:
        pass
        #pdb.set_trace()

    data_left = data[left_indices]
    data_right = data[right_indices]

    error_left = pruneDT(root.left, data_left, label_vector[left_indices])
    error_right = pruneDT(root.right, data_right, label_vector[right_indices])

    error_root = calcErrorCount(root, label_vector)
    if isLeaf(root.left) and isLeaf(root.right):
        prediction = root.prediction
        if error_root <= error_left + error_right:
            root.kd = True
            root.left = None
            root.right = None
            return error_root

    return error_left + error_right
        
    # go left
    # go right
    # check this root 

def makePrediction(root, example):
    node = root
    while not node.kd:
        threshold = node.threshold
        feature = node.feature
        val = example[feature]
        if val <= threshold:
            node = node.left
        else:
            node = node.right
    return node.prediction

def calcAccuracy(root, data, labels):
    count = 0
    for line in zip(data, labels):
        ex, label = line
        if makePrediction(root, ex) == label:
            count += 1
    
    return Decimal(count)/len(labels)
        

if __name__ == "__main__":
    getcontext().prec = 15
    np.set_printoptions(threshold=np.inf)
    train_matrix = read_input.readIDX(read_input.TRAIN_IMAGES_FILE)
    train_label = read_input.readIDX(read_input.TRAIN_LABEL_FILE)
    test_matrix = read_input.readIDX(read_input.TEST_IMAGES_FILE)
    test_label = read_input.readIDX(read_input.TEST_LABEL_FILE)
    root = makeDT(train_matrix, train_label)
    print(root)
    print("Pre-accuracy: " + str(calcAccuracy(root, test_matrix, test_label)))
    pruneDT(root, test_matrix, test_label)
    print(root)
    print("Post-accuracy: " + str(calcAccuracy(root, test_matrix, test_label)))

from collections import namedtuple
import numpy as np

# based on https://en.wikipedia.org/wiki/K-d_tree
# and http://code.activestate.com/recipes/577497-kd-tree-for-nearest-neighbor-search-in-a-k-dimensi/
Node = namedtuple("Node", 'point axis label left right')


def squared_distance(trainData, testData):
    d = 0
    n = len(testData)  # trainData has the last column as the label, so use testData's len instead
    for i in range(n):
        tmp = trainData[i] - testData[i]
        d += tmp*tmp
    return d


class NearestNeighborKD:
    # The number of features after shrank, can be dynamically set based on the number of training data,
    # but use 28 for now, which is the height and width of an image
    ShrinkSize = 28

    def __init__(self):
        self.treeRoot = None
        self.features = None
        self.shrink_features = 0
        self.trainData = None
        self.labelData = None

    def __shrink(self, instance):
        """
        Create a new smaller set of features by summing continuous features
        :param instance: single instance
        :return: train data with shrank features
        """
        oldFeatureSize = len(instance)
        newFeatureSize = NearestNeighborKD.ShrinkSize
        groupSize = oldFeatureSize / newFeatureSize
        shrankInstance = []
        for j in range(newFeatureSize):
            startIndex = j * groupSize
            val = 0
            for k in range(groupSize):
                if startIndex + k < oldFeatureSize:
                    val += instance[startIndex + k]
            shrankInstance.append(val)
        return np.array(shrankInstance)

    def __shrinkDataSet(self, trainDataSet):
        """
        Create a new smaller set of features by summing continuous features
        :param train: train data
        :return: train data with shrank features
        """
        data = []
        for i in range(len(trainDataSet)):
            data.append(self.__shrink(trainDataSet[i]))
        return np.array(data)

    def build_tree(self, instances, axis=0):
        if instances is None or len(instances) == 0:
            return None

        # http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
        actualFeature = self.features[axis]
        instances = instances[instances[:, actualFeature].argsort()]

        median_idx = len(instances) // 2
        median_point = instances[median_idx]
        median_label = median_point[-1]

        next_axis = (axis + 1) % len(self.features)
        return Node(median_point, axis, median_label,
                    self.build_tree(instances[:median_idx], next_axis),
                    self.build_tree(instances[median_idx + 1:], next_axis))

    def fit(self, trainData, labelData, features, shrink_features):
        """
        :param trainData: train data 
        :param labelData: label data
        :param features: indexes of train data features to use for training/predicting; if not set, use all features
        :param shrink_features: because the data has 784 features, 
        which is inefficient for k-d tree https://en.wikipedia.org/wiki/K-d_tree (High-dimensional data), turn this on 
        to shrink the data features to a smaller dimensional
        If shrink_features is turned on, prioritize it over the passed in "features"
        """
        if shrink_features == 1:
            trainData = self.__shrinkDataSet(trainData)
            features = list(range(len(trainData[0])))
            self.shrink_features = 1
        elif features is None:
            features = list(range(len(trainData[0])))

        self.shrink_features = shrink_features
        self.features = features

        # append the label as the last column of train data
        trainData = np.append(trainData, labelData.reshape((-1, 1)), 1)

        self.treeRoot = self.build_tree(trainData)

    def __nearest_neighbor(self, destination):
        """
        Find nearest neighbor
        :param destination: 
        :return: 
        """
        best = [None, None, float('inf')]

        # state of search: best point found, its label,
        # lowest squared distance

        def recursive_search(here):

            if here is None:
                return
            point, axis, label, left, right = here

            here_sd = squared_distance(point, destination)
            if here_sd < best[2]:
                best[:] = point, label, here_sd

            diff = destination[self.features[axis]] - point[self.features[axis]]
            close, away = (left, right) if diff <= 0 else (right, left)

            recursive_search(close)
            if diff ** 2 < best[2]:
                recursive_search(away)

        recursive_search(self.treeRoot)
        return best[0], best[1], best[2]

    def predict(self, instance):
        """
        Find nearest neighbor
        :param instance: 
        :return: 
        """
        if self.shrink_features == 1:
            instance = self.__shrink(instance)

        return self.__nearest_neighbor(instance)

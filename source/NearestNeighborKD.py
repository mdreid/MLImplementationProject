from collections import namedtuple
import numpy as np

# based on https://en.wikipedia.org/wiki/K-d_tree
# and http://code.activestate.com/recipes/577497-kd-tree-for-nearest-neighbor-search-in-a-k-dimensi/
Node = namedtuple("Node", 'point axis label left right')




class NearestNeighborKD:
    # The number of features after shrank, can be dynamically set based on the number of training data,
    # but use 28 for now, which is the height and width of an image
    ShrinkSize = 28
    def squared_distance(self, trainData, testData):
        d = 0
        n = len(self.features)  # trainData has the last column as the label, so use testData's len instead
        for i in range(n):
            tmp = trainData[i] - testData[i]
            d += tmp*tmp
        return d

    def __init__(self):
        self.treeRoot = None
        self.features = None
        self.shrink_features = 0
        self.trainData = None
        self.labelData = None

    def build_tree(self, instances, axis=0):
        if instances is None or len(instances) == 0:
            return None

        # http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
        actualFeature = self.features[axis]
        instances = instances[instances[:, actualFeature].argsort()]

        median_idx = len(instances) // 2
        median_point = instances[median_idx]
        median_label = median_point[-1]
        #print("Point: " + str(median_point))
        #print("Label: " + str(median_label))
        #print("Type of label: " + str(type(median_label)))

        next_axis = (axis + 1) % len(self.features)
        return Node(median_point, axis, median_label,
                    self.build_tree(instances[:median_idx], next_axis),
                    self.build_tree(instances[median_idx + 1:], next_axis))

    def fit(self, trainData, labelData, features, useKdTree = True):
        """
        :param trainData: train data
        :param labelData: label data
        :param features: indexes of train data features to use for training/predicting; if not set, use all features
        :param useKdTree: whether or not to build kd tree
        """
        if features is None:
            features = list(range(len(trainData[0])))

        #print("Label data: " + str(labelData))

        #self.shrink_features = shrink_features
        self.features = features
        #print("Features: " + str(features))

        # append the label as the last column of train data
        self.trainData = np.append(trainData, labelData.reshape((-1, 1)), 1)
        #print("Train Data: " + str(self.trainData[0]))

        if useKdTree:
            self.treeRoot = self.build_tree(self.trainData)
        else:
            self.treeRoot = None

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

            here_sd = self.squared_distance(point, destination)
            if here_sd < best[2]:
                best[:] = point, label, here_sd

            diff = destination[axis] - point[axis]
            close, away = (left, right) if diff <= 0 else (right, left)

            recursive_search(close)
            if diff ** 2 < best[2]:
                recursive_search(away)

        recursive_search(self.treeRoot)
        return best[0], best[1], best[2]

    def __exhaustive_search(self, instance):
        best = [None, None, float('inf')]

        for trainInstance in self.trainData:
          point = trainInstance[:-1]
          label = trainInstance[-1:][0]
          distance = self.squared_distance(point, instance)

          if distance < best[2]:
            best[:] = point, label, distance

        return best[0], best[1], best[2]

    def predict(self, instance, useKdTree = True):
        """
        Find nearest neighbor
        :param instance:
        :return:
        if self.shrink_features == 1:
            instance = self.__shrink(instance)
        """

        if useKdTree:
            result = self.__nearest_neighbor(instance)
        else:
            result = self.__exhaustive_search(instance)
        return result

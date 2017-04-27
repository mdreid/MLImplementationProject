from collections import namedtuple
import numpy as np
from NearestNeighborKD import NearestNeighborKD, squared_distance

class ExhaustiveSearch:
  ShrinkSize = NearestNeighborKD.ShrinkSize

  def __init__(self):
    self.features = None
    self.shrink_features = 0
    self.trainData = None
    self.labelData = None

  ## TODO: Same as NearestNeighborKD#__shrink
  def __shrink(self, instance):
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

  ## TODO: Same as NearestNeighborKD#__shrink
  def __shrinkDataSet(self, trainDataSet):
    data = []
    for i in range(len(trainDataSet)):
      data.append(self.__shrink(trainDataSet[i]))
    return np.array(data)

  def fit(self, trainData, labelData, features, shrink_features):
    self.labelData = labelData
    if shrink_features == 1:
      self.trainData = self.__shrinkDataSet(trainData)
      self.features = list(range(len(trainData[0])))
      self.shrink_features = 1
    elif features is None:
      self.features = list(range(len(trainData[0])))

    self.shrink_features = shrink_features
    self.features = features

    # append the label as the last column of train data
    self.trainData = np.append(self.trainData, labelData.reshape((-1, 1)), 1)

  def __exhaustive_search(self, instance):
    best = [None, None, float('inf')]

    for trainInstance in self.trainData:
      point = trainInstance[:-1]
      label = trainInstance[-1:][0]
      distance = squared_distance(point, instance)

      if distance < best[2]:
        best[:] = point, label, distance

    return best[0], best[1], best[2]

  def predict(self, instance):
    if self.shrink_features == 1:
      instance = self.__shrink(instance)

    return self.__exhaustive_search(instance)

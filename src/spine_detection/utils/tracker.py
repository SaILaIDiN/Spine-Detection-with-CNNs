from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

from spine_detection.utils.data_utils import calc_metric_xy


class CentroidTracker:
    """Control everything for tracking the spines"""

    def __init__(
        self,
        maxDisappeared: int = 50,
        minAppeared: int = 50,
        maxDiff: int = 30,
        maxVol: int = 80 * 80,
        iomThresh: float = 0.7,
        metric: str = "iom",
    ) -> None:
        """Initialize parameters and the tracker

        Args:
            maxDisappeared (int, optional): how many frames an object isn't seen before counting as disappeared
                Defaults to 50.
            minAppeared (int, optional): how many frames are needed to detect an object as real. Defaults to 50.
            maxDiff (int, optional): max pixel difference for being identified as same object. Defaults to 30.
            maxVol (int, optional): max volume of spines allowed to count as object. Defaults to 80*80.
            iomThresh (float, optional): min IoM necessary to identify as the same spine. Defaults to 0.7.
            metric (str, optional): Which metric should be used, currently 'iom' or 'iou' available. Defaults to 'iom'.
        """
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.beforeObjects = OrderedDict()
        self.disappeared = OrderedDict()
        self.appeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        self.minAppeared = minAppeared
        self.maxDiff = maxDiff
        self.maxVol = maxVol
        self.iomThresh = iomThresh

        if type(metric) == str:
            self.metric = lambda centroid1, centroid2: calc_metric_xy(centroid1, centroid2, metric=metric)
        else:
            self.metric = metric

    def register(self, centroid: np.ndarray) -> None:
        """register a new centroid

        Args:
            centroid (np.ndarray): Centroid in [cX, cY, w, h] format
        """
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.appeared[self.nextObjectID] = 1
        self.nextObjectID += 1

    def getObjects(self) -> OrderedDict:
        """Get all objects that are not hidden

        Returns:
            OrderedDict: dict with id, centroid pairs
        """
        correctObjects = OrderedDict()
        for objectID in self.objects.keys():
            if self.appeared[objectID] > self.minAppeared and self.disappeared[objectID] <= self.maxDisappeared:
                correctObjects[objectID] = self.objects[objectID]
        return correctObjects

    def preprocess(self, inputCentroids: np.ndarray) -> List[List]:
        """Preprocess np array of centroids to only interesting centroids

        Args:
            inputCentroids (np.ndarray): centroids of detection

        Returns:
            List[List]: List of centroids in (cX, cY, w, h) format
        """
        # input must be a np array
        # delete all centroids which are in each other and have the lower probability
        deleted_index = []
        added_centroid = []
        for i in range(len(inputCentroids)):
            # find the box with highest iom
            max_iom = self.iomThresh
            max_index = -1
            for j in range(i + 1, len(inputCentroids)):
                if j in deleted_index:
                    continue
                iom = self.metric(inputCentroids[i], inputCentroids[j])
                if iom >= max_iom:
                    max_iom = iom
                    max_index = j
            if max_index == -1:
                continue
            else:
                if inputCentroids[i][4] >= inputCentroids[max_index][4]:
                    deleted_index.append(max_index)
                else:
                    deleted_index.append(i)

        if len(added_centroid) != 0:
            inputCentroids = np.concatenate((inputCentroids, added_centroid), axis=0)
        ret_list = [inputCentroids[i] for i in range(len(inputCentroids)) if i not in deleted_index]
        return ret_list

    def update(self, rects: List[List]) -> OrderedDict:
        """Update tracker with detection rects

        Args:
            rects (List[List]): List of rects in format (x1, y1, x2, y2, conf)

        Returns:
            OrderedDict: dict with id, centroid pairs
        """
        # check to see if the list of input bounding box rectangles is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them as disappeared
            for objectID in self.objects.keys():
                self.disappeared[objectID] += 1

            # return early as there are no centroids or tracking info to update
            return self.getObjects()

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 5), dtype="float")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY, conf)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            w = int(endX - startX)
            h = int(endY - startY)
            if w * h >= self.maxVol:
                continue
            inputCentroids[i] = (cX, cY, w, h, conf)

        inputCentroids = self.preprocess(inputCentroids)

        # if we are currently not tracking any objects take the input centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, try to match the input centroids to existing object centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object centroids and input centroids, respectively
            # don't use distance, but IoM as comparison -> min is replaced by max!
            D = dist.cdist(np.array(objectCentroids), inputCentroids, metric=self.metric)

            # row -> original tracked objects, cols -> new input bboxes
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            rows = D.max(axis=1).argsort()[::-1]

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmax(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # is the distance too big, so we have to register a new object!
                if D[row, col] <= self.maxDiff:  # > self.maxDiff:
                    objectID = objectIDs[row]
                    self.register(inputCentroids[col])
                    self.disappeared[objectID] += 1
                else:
                    # otherwise, grab the object ID for the current row,
                    # set its new centroid, and reset the disappeared
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.appeared[objectID] += 1
                    self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # compare the number of inputCentroids and the number of real
            # existing centroids (not this ones with appeared <= minAppeared)
            object_len = len([1 for key in self.appeared.keys() if self.appeared[key] > self.minAppeared])
            # print('object len', object_len, D.shape)
            if object_len >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

            # register the input centroids as new objects in any case
            for col in unusedCols:
                self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.getObjects()

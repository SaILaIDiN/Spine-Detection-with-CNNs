from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

from typing import Optional, List

# in most cases np.ndarray can be used instead of List


def calc_metric_xy(centroid1: Optional[List] = None, centroid2: Optional[List] = None, rect1: Optional[List] = None,
                   rect2: Optional[List] = None, metric: str = 'iom') -> float:
    """calculates metric (usually Intersection over Minimum) between two centroids or two rects

    Args:
        centroid1 (Optional[List]): first centroid in (cX, cY, w, h, ...) format. Defaults to None.
        centroid2 (Optional[List]): second centroid in (cX, cY, w, h, ...) format. Defaults to None.
        rect1 (Optional[List]): first rect in (x1, y1, x2, y2) format. Defaults to None.
        rect2 (Optional[List]): second rect in (x1, y1, x2, y2) format. Defaults to None.
        metric (str): Metric to use, either iou or iom. Defaults to iom

    Raises:
        AttributeError: If neither two centroids nor two boxes are given
        NotImplementedError: metric need to be 'iom' or 'iou', other values are not implemented yet

    Returns:
        float: value of IoM
    """
    # if centroids are given -> calc box coordinates first before calculating everything
    if centroid1 is not None and centroid2 is not None:
        cX1, cY1, w1, h1 = centroid1[:4]
        cX2, cY2, w2, h2 = centroid2[:4]

        x11, x12, y11, y12 = cX1-w1/2, cX1+w1/2, cY1-h1/2, cY1+h1/2
        x21, x22, y21, y22 = cX2-w2/2, cX2+w2/2, cY2-h2/2, cY2+h2/2
    elif rect1 is not None and rect2 is not None:
        x11, y11, x12, y12 = rect1
        x21, y21, x22, y22 = rect2
    else:
        raise AttributeError(
            "You have to provide either two centroids or two boxes but neither is given.")

    area1 = (x12-x11)*(y12-y11)
    area2 = (x22-x21)*(y22-y21)

    x1, x2, y1, y2 = max(x11, x21), min(x12, x22), max(y11, y21), min(y12, y22)
    if x1 >= x2 or y1 >= y2:
        return 0

    intersection = (x2-x1)*(y2-y1)
    union = area1 + area2 - intersection

    if metric == 'iom':
        return intersection/min(area1, area2)
    elif metric == 'iou':
        return intersection/union
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented")


def calc_metric_z(centroid1: List, centroid2: List) -> float:
    """calculate IoM only for z-direction

    Args:
        centroid1 (List): First centroid of format (..., z1, z2)
        centroid2 (List): Second centroid of format (..., z1, z2)

    Returns:
        float: IoM in z-direction using start and end z-frames z1, z2
    """
    # look how many of centroid1 and centroid2 z-axis overlap
    # using intersection/union, not intersection/minimum
    min_z1, max_z1 = centroid1[-2:]
    min_z2, max_z2 = centroid2[-2:]

    if max_z1 < min_z2 or max_z2 < min_z1:
        return 0

    # +1 has to be added because of how we count with both ends including!
    # if GT is visible in z-layers 5 - 8 (inclusive) and detection is in layer 8 - 9
    # they have one overlap (8), but 8 - 8 = 0 which is wrong!
    intersection = min(max_z1, max_z2) - max(min_z1, min_z2) + 1
    min_val = min(max_z1-min_z1, max_z2-min_z2) + 1

    if min_val == 0:
        return 0

    # gt has saved each spine with only one img -04.png
    # should be no problem any more
    return intersection/min_val


def calc_metric(centroid1: List, centroid2: List, metric: str = 'iom') -> float:
    """Combine IoM in xy and in z-direction

    Args:
        centroid1 (List): First centroid (cX, cY, w, h, z1, z2)
        centroid2 (List): Second centroid same format
        metric (str): Metric to use, either iou or iom. Defaults to iom

    Returns:
        float: overall F_1-3D-score of both centroids
    """
    # how to combine both metrics
    iom = calc_metric_xy(centroid1, centroid2, metric=metric)
    z_iom = calc_metric_z(centroid1, centroid2)

    # use similar formula to fscore, but replace precision and recall with iom and z_iom
    # beta=low because z_iom should not count that much
    beta = 0.5
    if iom == 0 or z_iom == 0:
        if iom != 0 and z_iom == 0:
            print(f"z-Problem: iom is {iom} while z_iom is {z_iom}")
        return 0
    final_score = (1 + beta**2) * (iom * z_iom)/(beta**2 * iom + z_iom)
    return final_score


class CentroidTracker():
    """Control everything for tracking the spines
    """

    def __init__(self, maxDisappeared: int = 50, minAppeared: int = 50, maxDiff: int = 30, maxVol: int = 80*80,
                 iomThresh: float = 0.7, metric: str = 'iom') -> None:
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
            self.metric = lambda centroid1, centroid2: calc_metric_xy(
                centroid1, centroid2, metric=metric)
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
            for j in range(i+1, len(inputCentroids)):
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
            inputCentroids = np.concatenate(
                (inputCentroids, added_centroid), axis=0)
        ret_list = [inputCentroids[i]
                    for i in range(len(inputCentroids)) if i not in deleted_index]
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
            w = int(endX-startX)
            h = int(endY-startY)
            if w*h >= self.maxVol:
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
            D = dist.cdist(np.array(objectCentroids),
                           inputCentroids, metric=self.metric)

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
            object_len = len([1 for key in self.appeared.keys()
                              if self.appeared[key] > self.minAppeared])
            #print('object len', object_len, D.shape)
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

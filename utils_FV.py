from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

def calc_IoM(centroid1=None, centroid2=None, rect1=None, rect2=None):
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
        raise AttributeError("You have to provide either two centroids or two boxes but neither is given.")
    
    area1 = (x12-x11)*(y12-y11); area2 = (x22-x21)*(y22-y21)
    
    x1, x2, y1, y2 = max(x11,x21), min(x12,x22), max(y11,y21), min(y12,y22)
    if x1 >= x2 or y1 >= y2:# or area1 >= 100*100 or area2 >= 100*100:
        return 0

    intersection = (x2-x1)*(y2-y1)

    return intersection/min(area1, area2)
    
def calc_IoU(centroid1, centroid2):
    cX1, cY1, w1, h1 = centroid1[:4]
    cX2, cY2, w2, h2 = centroid2[:4]
    area1 = w1*h1; area2 = w2*h2

    x11, x12, y11, y12 = cX1-w1/2, cX1+w1/2, cY1-h1/2, cY1+h1/2
    x21, x22, y21, y22 = cX2-w2/2, cX2+w2/2, cY2-h2/2, cY2+h2/2

    x1, x2, y1, y2 = max(x11,x21), min(x12,x22), max(y11,y21), min(y12,y22)
    if x1 >= x2 or y1 >= y2:# or area1 >= 100*100 or area2 >= 100*100:
        return 0

    intersection = (x2-x1)*(y2-y1)
    
    union = area1 + area2 - intersection

    return intersection/union

class CentroidTracker():
    # the diagonal of a 20x20 square is around 30 (28.x)
    def __init__(self, maxDisappeared=50, minAppeared=50, maxDiff=30, maxVol=80*80, iomThresh=0.7, metric='iom'):
        # maxDisappeared -> how many frames an object isn't seen for allowing disappearance
        # minAppeared -> how many frames are needed to detect an object as real
        # maxHold -> how many images can be between two appearances?
        # maxDiff -> max pixel difference for being identified as same object
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
        #self.maxHold = maxHold
        self.maxDiff = maxDiff
        self.maxVol = maxVol
        self.iomThresh = iomThresh
        
        if metric == 'iom':
            self.metric = calc_IoM
        elif metric == 'iou':
            self.metric = calc_IoU
        else:
            self.metric = metric

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        #self.beforeObjects[self.nextObjectID] = centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.appeared[self.nextObjectID] = 1#self.minAppeared+1
        self.nextObjectID += 1

    def deregister(self, objectID):
        # DO NOT DELETE OBJECT, JUST TAKE IT TO THE BACKGROUND
        # Reason: it can be possible that spines from another stack are coming back
        # therefore save old position and just return all correct objects
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        #del self.objects[objectID]
        #del self.disappeared[objectID]
        #del self.appeared[objectID]
        #self.disappeared[objectID] = self.maxDisappeared + 1
        pass

    def getObjects(self):
        correctObjects = OrderedDict()
        for objectID in self.objects.keys():
            if self.appeared[objectID] > self.minAppeared and self.disappeared[objectID] <= self.maxDisappeared:
                correctObjects[objectID] = self.objects[objectID]
        return correctObjects

    def average_centroids(self, centroid1, centroid2):
        cX1, cY1, w1, h1, conf1 = centroid1
        cX2, cY2, w2, h2, conf2 = centroid2

        # weights of weighted arithmetic sum
        prob1 = conf1 / (conf1+conf2)
        prob2 = conf2 / (conf1+conf2)

        cX = prob1*cX1 + prob2*cX2
        cY = prob1*cY1 + prob2*cY2
        w = prob1*w1 + prob2*w2
        h = prob1*h1 + prob2*h2
        conf = prob1*conf1 + prob2*conf2

        return (cX, cY, w, h, conf)

    def preprocess(self, inputCentroids):
        # input must be a np array
        # delete all centroids which are in each other and have the lower probability
        #return inputCentroids
        #if True:
        #    return inputCentroids
        #print('before', inputCentroids)
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
                #print('iom', iom, inputCentroids[i][4], inputCentroids[j][4])
                if iom >= max_iom:
                    max_iom = iom
                    max_index = j
                    # if inputCentroids[i][4] >= inputCentroids[j][4]:
                    #     deleted_index.append(j)
                    # else:
                    #     deleted_index.append(i)
            if max_index == -1:
                continue
            else:
                if inputCentroids[i][4] >= inputCentroids[max_index][4]:
                    deleted_index.append(max_index)
                else:
                    deleted_index.append(i)

            #     if iom >= max_iom:
            #         max_iom = iom
            #         max_index = j

            # # compare confidence and average depending on confidence
            # if max_index == -1:
            #     continue
            # deleted_index.append(i)
            # deleted_index.append(max_index)
            # added_centroid.append(self.average_centroids(inputCentroids[i], inputCentroids[j]))

        if len(added_centroid)!=0:
            inputCentroids = np.concatenate((inputCentroids, added_centroid), axis=0)
        ret_list = [inputCentroids[i] for i in range(len(inputCentroids)) if i not in deleted_index]
        #print('ret:', ret_list)
        return ret_list

    def update(self, rects):
        #print("Rects: ", rects, "Objects: ", self.objects.keys(), "DA: ", self.disappeared, "A: ",self.appeared)
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared

            # use extra list so that there is no confusion with a OrderedDict-Mutation during for loop
            deregister_keys = []
            for objectID in self.objects.keys():
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    deregister_keys.append(objectID)
                    #self.deregister(objectID)

            for objectID in deregister_keys:
                self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.getObjects()

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 5), dtype="float")

        # loop over the bounding box rectangles
        # c for confidence
        for (i, (startX, startY, endX, endY, conf)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            w = int(endX-startX)
            h = int(endY-startY)
            if w*h >= self.maxVol:
                continue
            inputCentroids[i] = (cX, cY, w, h, conf)

        #print('before', inputCentroids)
        inputCentroids = self.preprocess(inputCentroids)
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            #objectCentroids = self.preprocess(np.array(objectCentroids))

            # don't use distance, but IoM as comparison -> min is replaced by max!
            #D = dist.cdist(np.array(objectCentroids), inputCentroids)
            D = dist.cdist(np.array(objectCentroids), inputCentroids, metric=self.metric)

            # row -> original tracked objects, cols -> new input bboxes
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            #rows = D.min(axis=1).argsort()
            #print(D.max(axis=1))
            # normal argsort sorts ascending, we want descending!
            rows = D.max(axis=1).argsort()[::-1]

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            #cols = D.argmin(axis=1)[rows]
            cols = D.argmax(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            #print('before:', self.objects)
            #print('input', inputCentroids)
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # is the distance too big, so we have to register a new object!
                #print('row, col, dist:', row, col, D[row, col], objectIDs[row])
                if D[row, col] <= self.maxDiff: #> self.maxDiff:
                    objectID = objectIDs[row]
                    self.register(inputCentroids[col])
                    #self.appeared[objectID] += 1
                    self.disappeared[objectID] += 1
                else:
                    # otherwise, grab the object ID for the current row,
                    # set its new centroid, and reset the disappeared
                    # counter
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.appeared[objectID] += 1
                    self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            #if D.shape[0] >= D.shape[1]:
            
            # compare the number of inputCentroids and the number of real
            # existing centroids (not this ones with appeared <= minAppeared)
            object_len = len([1 for key in self.appeared.keys() if self.appeared[key] > self.minAppeared])
            #print('object len', object_len, D.shape)
            if object_len >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            #else:
            #    for col in unusedCols:
            #        self.register(inputCentroids[col])
            
            # register the input centroids as new objects in any case
            for col in unusedCols:
                self.register(inputCentroids[col])

        # return the set of trackable objects
        #print('objectID:', self.nextObjectID)
        return self.getObjects()

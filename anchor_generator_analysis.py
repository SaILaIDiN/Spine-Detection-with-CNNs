""" This script contains the analysis of anchor boxes via the mmdetection anchor generator used in Cascade-RCNN. """

from mmdet.core import AnchorGenerator
import torch


# number of tuples in the list for featmap_sizes in grid_priors() must be the number of strides in
# the AnchorGenerator, because the strides define the different feature levels due to sampling through
# an input image with different stride sizes

# number of strides should be the same as base sizes if base_sizes is used

# number of anchor types at each sample point is the product of len(scales) and len(ratios)

# a feature map in grid_priors() defines how the anchor boxes would be shaped if you had sampled
# an input image with given strides and got a feature map of given shape (w, h)

# for fixed a scale, the area of bounding boxes over different aspect ratios remains constant!

# # # Comparison 1
# default_vers = AnchorGenerator(scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64])
# all_anchors = default_vers.grid_priors([(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)], device='cuda')
# print(all_anchors)
# print(all_anchors[0].size())
#
# default_vers = AnchorGenerator(scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64])
#
# other_vers = AnchorGenerator(base_sizes=[4, 8], scales=[1],
#                              ratios=[1.0], strides=[4, 6])
# all_anchors = other_vers.grid_priors([(2, 2), (2, 2)], device='cuda')
# print(all_anchors)
#
# test_vers = AnchorGenerator(base_sizes=[10], scales=[1], ratios=[1.0], strides=[5])
# all_anchors = test_vers.grid_priors([(5, 1)], device='cuda')
# print(all_anchors)

# # # Comparison 2
# our_anchors = AnchorGenerator(scales=[2],
#                               ratios=[1],
#                               strides=[2, 4])
# all_our_anchors = our_anchors.grid_priors([(2, 2), (2, 2)])
# print(all_our_anchors)
# print(all_our_anchors[0].size())
#
# our_anchors = AnchorGenerator(scales=[1],
#                               ratios=[1],
#                               strides=[4, 8])
# all_our_anchors = our_anchors.grid_priors([(2, 2), (2, 2)])
# print(all_our_anchors)
# print(all_our_anchors[0].size())
#
# our_anchors = AnchorGenerator(base_sizes=[1, 1], scales=[1],
#                               ratios=[1],
#                               strides=[4, 8])
# all_our_anchors = our_anchors.grid_priors([(2, 2), (2, 2)])
# print(all_our_anchors)
# print(all_our_anchors[0].size())


# # # Evaluate best Parameters for our Cascade-RCNN for the given Dataset

# Test set data:
# mean height: 18.3+-3.94, mean width: 18.33+-5.14
# total range height: 9 to 33, total range width: 7 to 41
# mean aspect ratio: 1.04+-0.22
# total range aspect ratio: 0.4 to 2.1
# mean area: 347+-163

# Val set data:
# mean height: 20.18+-6.47, mean width: 20.68+-5.43
# total range height: 8 to 45, total range width: 9 to 42
# mean aspect ratio: 1.00+-0.29
# total range aspect ratio: 0.5 to 2.6
# mean area: 435+-233

# Train set data:
# mean height: 20.38+-5.99, mean width: 20.37+-6.04
# total range height: 6 to 48, total range width: 7 to 52
# mean aspect ratio: 1.04+-0.3
# total range aspect ratio: 0.25 to 3.1
# mean area: 434+-234

# First observation: mean height and width and aspect ratios of all subsets are very similar
# Second observation: the total range of height and width of train is significantly larger as for test
# Third observation: total range of aspect ratio for train is also much wider as for test
# Fourth observation: almost all test boxes are clustered around the mean of area and aspect ratio
# Fifth observation: test boxes have an 80 pixel smaller area on average
# Sixth observation: the training set has much more variation in size and aspect ratio than test

# => Conclusion: Even though the anchor boxes are not required to be larger than the test set ranges,
# the model can still benefit from the additional features of larger ground truths in the train set
# without taking into account that this model has to be robust for different scales of spines, we currently
# focus on optimising this network on this given test set, thus we make our decision on the anchor boxes
# only based on the given properties of train, val and test

# We have to adjust the chosen values for the Anchor-Boxes based on the possible spatial Data Augmentation
# Spatial DA impacts dist of height and width and dist of aspect ratios
# We have to choose the lower and upper boundary based on the min(x_min, 1/x_max) for lowest aspect ratio
# and max(x_max, 1/x_min) for highest aspect ratio
# For min height and min width you take min(height_min, width_min) and max(height_max, width_max) for
# max height and max width

# we choose as a first try:
# base_sizes=[5, 15, 20, 25, 30, 35, 50, 60], scales=[1],
# ratios=[0.25, 0.5, 0.75, 1.0, 1.33, 2.0, 4.0]
# strides=[4, 10, 15, 20, 25, 30, 20, 30]
# OR with other strides of 1/5 overlap for aspect ratio = 1.0
# strides=[1, 3, 4, 5, 6, 7, 10, 12]

our_anchors = AnchorGenerator(base_sizes=[5, 15, 20, 25, 30, 35, 50, 60], scales=[1],
                              ratios=[0.25, 0.5, 0.75, 1.0, 1.33, 2.0, 4.0],
                              strides=[1, 3, 4, 5, 6, 7, 10, 12])
# all_our_anchors = our_anchors.grid_priors([(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)])
all_our_anchors = our_anchors.grid_priors([(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
print(all_our_anchors[0])
print(len(all_our_anchors))
print(all_our_anchors[0].size())

our_anchors = AnchorGenerator(scales=[2],
                              ratios=[0.25, 0.5, 0.75, 1.0, 1.33, 2.0, 4.0],
                              strides=[2, 8, 10, 13, 15, 18, 25])
# all_our_anchors = our_anchors.grid_priors([(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)])
all_our_anchors = our_anchors.grid_priors([(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
print(all_our_anchors[0])
print(len(all_our_anchors))
print(all_our_anchors[0].size())

our_anchors = AnchorGenerator(scales=[1],
                              ratios=[0.25, 0.5, 0.75, 1.0, 1.33, 2.0, 4.0],
                              strides=[5, 15, 20, 25, 30, 35, 50])
# all_our_anchors = our_anchors.grid_priors([(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)])
all_our_anchors = our_anchors.grid_priors([(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
print(all_our_anchors[0])
print(len(all_our_anchors))
print(all_our_anchors[0].size())

default_vers = AnchorGenerator(scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64])
all_anchors = default_vers.grid_priors([(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)], device='cuda')
# This is the default Anchor-Generator and tells you that an anchor box with ratio=1 of scales * strides[i] will
# overlap with its next sample by (scales-1)*strides[i]
print(all_anchors[0:2])
print(len(all_anchors))
print(all_anchors[0].size())

# # # From the default setting, we try to adjust it in favor of our data set ground truth properties

default_vers = AnchorGenerator(scales=[8], ratios=[0.5, 1.0, 2.0], strides=[1, 2, 4, 8, 16])
all_anchors = default_vers.grid_priors([(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)], device='cuda')
# Here we, shifted the list of strides towards smaller ones because the boxes in the data sets are smaller,
# 32*8 or 64*8 are far too large for our purpose
print(all_anchors[0:2])
print(len(all_anchors))
print(all_anchors[0].size())


default_vers = AnchorGenerator(scales=[8], ratios=[0.25, 0.33, 0.5, 1.0, 2.0, 3.0, 4.0], strides=[1, 2, 4, 8, 16])
all_anchors = default_vers.grid_priors([(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)], device='cuda')
# Here we keep the strides and scale, but expand the set of aspect ratio more in alignment with our distribution of
# aspect ratios, see the plots
print(all_anchors[0:2])
print(len(all_anchors))
print(all_anchors[0].size())


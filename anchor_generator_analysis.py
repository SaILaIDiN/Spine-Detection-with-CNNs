""" This script contains the analysis of anchor boxes via the mmdetection anchor generator used in Cascade-RCNN. """

from mmdet.core import AnchorGenerator

default_vers = AnchorGenerator(scales=[8], ratios=[1.0], strides=[4, 8, 16, 32, 64])
all_anchors = default_vers.grid_priors([(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)], device='cuda')
print(all_anchors)
# default_vers = AnchorGenerator(scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64])

# other_vers = AnchorGenerator(base_sizes=[4, 8, 16, 32, 64], scales=[1],
#                              ratios=[0.75, 1.0, 1.25], strides=[4, 6, 8, 10, 12])
# other_vers_2 = AnchorGenerator(base_sizes=[16], scales=[1, 2, 4],
#                                ratios=[0.75, 1.0, 1.25], strides=[8])
test_vers = AnchorGenerator(base_sizes=[10], scales=[1], ratios=[1.0], strides=[5])
all_anchors = test_vers.grid_priors([(5, 1)], device='cuda')
print(all_anchors)

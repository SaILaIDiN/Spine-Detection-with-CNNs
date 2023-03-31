import pandas as pd
import glob

# df = pd.read_csv("output/tracking/GT/data_tracking_gt_min.csv")
# df = pd.read_csv("output/tracking/GT/data_tracking_gt_maj.csv")
# df = pd.read_csv("output/tracking/GT/data_tracking_gt_max.csv")
df = pd.read_csv("output/tracking/GT/data_tracking.csv")
counter = 0
file_set_test = set()
for filename, spine_data in df.groupby("filename"):
    counter += 1
    # print("Counter: ", counter)
    file_set_test.add(filename.split('/')[-1])
    # print(file_set_test)

df2 = pd.read_csv("data/default_annotations/train.csv")
file_set_train = set()
counter_2 = 0
for filename, spine_data in df2.groupby("filename"):
    counter_2 += 1
    # print("Counter 2: ", counter_2)
    file_set_train.add(filename.split('/')[-1])
    # print(file_set_train)

print("Test size: ", len(file_set_test))
print("Train size: ", len(file_set_train))
file_set_diff = file_set_train.difference(file_set_test)
print("Diff size: ", len(file_set_diff))

# # # Get filenames with matching expression of test-data
filenames_for_tracking = glob.glob("data/raw/person1/SR052N1D1day1stack*.png")
print("Size of tracking set: ", len(filenames_for_tracking))
print("Type tracking set: ", type(filenames_for_tracking))
print("Tracking set: ", filenames_for_tracking[0])
filenames_for_tracking = [x.split('/')[-1] for x in filenames_for_tracking]
print("Tracking set: ", filenames_for_tracking[0])
print("Test set: ", list(file_set_test)[0])
file_set_tracking = set(filenames_for_tracking)
file_set_diff2 = file_set_tracking.difference(file_set_test)
print("Diff size: ", len(file_set_diff2))
# => This shows that all the 61 Filenames given in GT-Min are inside the 100 input Filenames for tracking/evaluation

print("Set of remaining Input Filenames where no Spine was labeled by experts: ", file_set_diff2)
print("Sorted List of remaining Input Filenames where no Spine was labeled by experts: ", sorted(list(file_set_diff2)))

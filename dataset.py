import os
import numpy as np
import torch
from torchvision.transforms import functional as F
from torch.utils.data.dataset import Dataset
from PIL import Image
import pandas as pd
import references.detection.transforms as T


class SpineDataset(Dataset):
    def __init__(self, csv_name, transforms):
        super().__init__()
        self.csv_name = csv_name
        self.transforms = transforms

        self.df = pd.read_csv(self.csv_name)
        self.imgs = list(sorted(self.df["filename"].values.tolist()))
        # print("Self.imgs size: ", np.shape(self.imgs))
        self.imgs = np.unique(self.imgs)
        # print("Self.imgs: ", self.imgs)
        # print("Self.imgs size: ", np.shape(self.imgs))
        self.boxes = list(sorted(self.df[["filename", "xmin", "ymin", "xmax", "ymax"]].values.tolist()))
        # print("Self.boxes: ", self.boxes)
        # print("Self.boxes size: ", np.shape(self.boxes))

        all_boxes = []
        for filename in self.imgs:
            joint_boxes = []  # for one img
            filename_pos = [i for i, sub_list in enumerate(self.boxes) if sub_list[0] == filename]
            for pos in filename_pos:
                joint_boxes.append(self.boxes[pos][1:])  # only store the four coordinates of a box
            all_boxes.append(joint_boxes)
        self.boxes = all_boxes
        # print("Self.boxes: ", self.boxes)
        # print("Self.boxes size: ", np.shape(self.boxes))

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        boxes = self.boxes[idx]
        num_objs = len(boxes)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader

    csv_name = "data/default_annotations/data.csv"
    dataset = SpineDataset(csv_name, transforms=get_transform(False))
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    img, target = next(iter(train_loader))
    # print("Img: ", img)
    # print("Img size: ", img.size())
    # print("Img type: ", type(img))

    counter = 0
    for i, batch in enumerate(train_loader):
        img, target = batch[0], batch[1]
        counter += 1
        print("Img: ", img)
        print("Img size: ", img.size())
        print("Img type: ", type(img))

        print("Target: ", target)
        print("Target size: ", len(target))
        print("Target type: ", type(target))
        if counter >= 10:
           break

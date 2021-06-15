import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import SpineDataset, get_transform
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from references.detection.engine import train_one_epoch, evaluate
import references.detection.utils as utils


def main():
    # # # Create the model
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # # # Create dataset and dataloader
    csv_name = "data/default_annotations/data.csv"
    dataset = SpineDataset(csv_name, transforms=get_transform(False))

    train_set, test_set = random_split(dataset, [int(len(dataset) * 0.7), len(dataset) - int(len(dataset) * 0.7)])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, test_loader, device=device)

    print("Training and validation complete!")

    torch.save(model.state_dict(), "own_models2/default_model/faster_rcnn_model.pth")
    print("Training model stored!")


if __name__ == "__main__":
    main()

    # # # # Try simple inference
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #
    # # replace the classifier with a new one, that has
    # # num_classes which is user-defined
    # num_classes = 2  # 1 class (person) + background
    # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #
    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)  # Returns predictions
    # print(predictions)

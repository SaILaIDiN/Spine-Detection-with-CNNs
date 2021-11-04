from tracking_mmdet import tracking_main
from tracking_mmdet import parser as parser_tracking


def get_dict(model_type, use_aug, epoch, use_offsets):
    dict_tmp = {"model": model_type + '_aug_' + use_aug,
                "images": "data/raw/person1/SR052N1D1day1stack*.png",
                "use_offsets": use_offsets,
                "model_type": model_type,
                "use_aug": use_aug,
                "model_epoch": epoch}
    return dict_tmp


args_tracking = parser_tracking.parse_args()
argparse_dict = vars(args_tracking)

list_model_type = ["Cascade_RCNN", "GFL", "VFNet", "Def_DETR"]
list_use_aug = ["True", "False"]
list_epochs = ["epoch_" + str(x) for x in range(1, 16)]
use_offsets = "True"

for model_type in list_model_type:
    for use_aug in list_use_aug:
        if model_type == "Def_DETR":
            use_aug = "False"  # because this model has no data augmentation
        for epoch in list_epochs:
            dict_tmp = get_dict(model_type, use_aug, epoch, use_offsets)
            argparse_dict.update(dict_tmp)
            try:
                tracking_main(args_tracking)
            except:
                print("Some file or path is not existent!")


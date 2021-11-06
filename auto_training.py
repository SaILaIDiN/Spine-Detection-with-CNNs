from train_mmdet import train_main
from train_mmdet import parser as parser_train


def get_training_dict(model_type, use_aug, lr,
                      max_epochs=None, warm_up=None, steps_decay=None, momentum=None, weight_decay=None, dropout=None):
    dict_tmp = {'model_type': model_type,
                'use_aug': use_aug,
                'learning_rate': lr,
                'max_epochs': max_epochs,
                'warm_up': warm_up,
                'steps_decay': steps_decay,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'dropout': dropout
                }
    return dict_tmp


def get_data_aug_dict(random_brightness=None, random_contrast=None, vertical_flip=None,
                      horizontal_flip=None, rotate=None):
    dict_tmp = {'random_brightness': random_brightness,
                'random_contrast': random_contrast,
                'vertical_flip': vertical_flip,
                'horizontal_flip': horizontal_flip,
                'rotate': rotate
                }
    return dict_tmp


args_train = parser_train.parse_args()
argparse_train_dict = vars(args_train)

# # # Hardcoded values for basic training setup
list_model_type = ["VFNet"]
list_use_aug = ["True", "False"]
val_max_epochs = 10
list_learning_rate = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
list_warm_up = ["linear"]  # can use 'constant', 'linear', 'exp' or None
val_steps_decay = [5, 7]  # format [step_1, step_2, ..]

# # # Hardcoded values for data augmentation
val_vertical_flip = 0.5
val_horizontal_flip = 0.5
val_rotate = 0.5

for model_type in list_model_type:
    for use_aug in list_use_aug:
        if model_type == "Def_DETR" and use_aug == "True":
            continue  # because this model has no data augmentation
        for lr in list_learning_rate:
            for warm_up in list_warm_up:
                dict_tmp = get_training_dict(model_type, use_aug, lr, val_max_epochs, warm_up, val_steps_decay)
                argparse_train_dict.update(dict_tmp)
                if use_aug == "True":
                    dict_tmp = get_data_aug_dict()
                    argparse_train_dict.update(dict_tmp)
                try:
                    train_main(args_train)
                except:
                    print("Something has gone wrong!")

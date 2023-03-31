import glob
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import pkg_resources
import yaml
from mmcv import Config
from yacs.config import CfgNode as CN

logger = logging.getLogger(__name__)

import argparse

from mmdet.apis import init_detector


def load_model(
    model_type: str,
    use_aug: bool,
    model_epoch: str,
    param_config: str,
    model: Optional[str] = None,
    device: str = "cuda:0",
    return_path: bool = False,
) -> Tuple[Any, str]:
    """Load frozen model
    Args:
        param_config (str): contains a pregenerated string of the tweaked hyperparameters used to navigate through
                            model folders
    """

    model_folder = "tutorial_exps"
    cfg_path = pkg_resources.resource_filename("spine_detection", "configs/model_config_paths.yaml")
    paths_cfg = load_config(cfg_path)
    if model_type is None:
        model_type = "default"
    if model is None:
        checkpoint_file = get_checkpoint_path(model_type, model_folder, use_aug, paths_cfg)
    else:
        checkpoint_file = f"{model_folder}/{model}"
    config_file = get_config_path(model_type, use_aug, paths_cfg)
    logger.info("Loading model ...")
    checkpoint_file = str(Path(checkpoint_file) / param_config / (model_epoch + ".pth"))

    # construct checkpoint file name
    # checkpoint_file = os.path.join(checkpoint_file, os.path.join(param_config, model_epoch + ".pth"))

    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    if return_path:
        return model, checkpoint_file
    else:
        return model


def load_config(config_path: str):

    config = None
    with open(config_path, "r") as f:
        if config_path.endswith(".yaml"):
            config = yaml.safe_load(f)
    return CN(config)


def get_checkpoint_path(model_type: str, model_folder: str, use_aug: bool, paths_cfg: CN):
    """get path to model checkpoint

    :param model_type: one of the available model types
    :param model_folder: experiment folder where the checkpoints are saved
    :param use_aug: flag if augmentation should be used or not
    :param paths_cfg: config of models with their paths
    :raises AttributeError: if model type does not exist
    :return: path to model checkpoint
    """
    if model_type not in paths_cfg.model_paths:
        raise AttributeError(f"Attribute {model_type} under model_paths of the paths_cfg does not exist.")
    base_checkpoint = paths_cfg.model_paths[model_type].base_checkpoint
    if model_type != "default":
        add_data_aug = (1 - use_aug) * "_no" + "_data_augmentation"
    else:
        add_data_aug = ""
    return str(Path(model_folder) / (base_checkpoint + add_data_aug))


def get_config_path(model_type: str, use_aug: bool, paths_cfg: CN):
    """get path to model config file

    :param model_type: one of the available model types
    :param use_aug: flag if augmentation should be used or not
    :param paths_cfg: config of models with their paths
    :raises AttributeError: if model type does not exist
    :return: path to config file
    """
    if model_type not in paths_cfg.model_paths:
        raise AttributeError(f"Attribute {model_type} under model_paths of the paths_cfg does not exist.")
    base_path = paths_cfg.base_config_path
    base_config = paths_cfg.model_paths[model_type].base_config
    if model_type != "default":
        add_data_aug = "_SPINE" + use_aug * "_AUG" + ".py"
    else:
        add_data_aug = "_SPINE.py"
    config_path = str(Path(base_path) / (base_config + add_data_aug))
    return Config.fromfile(config_path)


def get_pretrained_checkpoint_path(model_type: str, paths_cfg: CN, model_suffix: Optional[str] = None):
    """get path to pretrained checkpoint

    :param model_type: one of the available model types
    :param paths_cfg: config of models with their paths
    :param model_suffix: suffix generated when downloading a pretrained checkpoint. If not set, search for any model.
    :raises AttributeError: if model type does not exist
    :return: path to pretrained checkpoint
    """
    if model_type not in paths_cfg.model_paths:
        raise AttributeError(f"Attribute {model_type} under model_paths of the paths_cfg does not exist.")
    base_path = paths_cfg.base_checkpoint_path
    base_checkpoint = paths_cfg.model_paths[model_type].base_config.split("/")[-1]
    if model_suffix is None:
        base_pretrained_path = Path(base_path) / base_checkpoint
        possible_paths = glob.glob(str(base_pretrained_path) + "*.pth")
        if len(possible_paths) == 1:
            final_path = str(possible_paths[0])
        elif len(possible_paths) > 1:
            print("Multiple checkpoints detected:")
            for i, path in enumerate(possible_paths):
                print(f"{str(i):>3s}: {str(path).split('/')[-1]}")
            version = input(f"Which checkpoint do you want to use? Enter a number between 0 and {i}. ")
            try:
                version = int(version)
                final_path = str(possible_paths[version])
            except Exception as e:
                raise ValueError(f"Your input was {version} but it needs to be a number between 0 and {i}.")
        else:
            raise ValueError(f"No checkpoint available inside {str(base_pretrained_path)}")
    else:
        final_path = str(Path(base_path) / (base_checkpoint + model_suffix + ".pth"))
    return final_path


class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    Class for customizing the help output
    """

    def _format_action_invocation(self, action):
        """
        Reformat multiple metavar output
            -d <host>, --device <host>, --host <host>
        to single output
            -d, --device, --host <host>
        """

        orgstr = argparse.ArgumentDefaultsHelpFormatter._format_action_invocation(self, action)
        if orgstr and orgstr[0] != "-":  # only optional arguments
            return orgstr
        res = getattr(action, "_formatted_action_invocation", None)
        if res:
            return res

        options = orgstr.split(", ")
        if len(options) <= 1:
            action._formatted_action_invocation = orgstr
            return orgstr

        return_list = []
        for option in options:
            meta = ""
            arg = option.split(" ")
            if len(arg) > 1:
                meta = arg[1]
            return_list.append(arg[0])
        if len(meta) > 0 and len(return_list) > 0:
            return_list[len(return_list) - 1] += " " + meta
        action._formatted_action_invocation = ", ".join(return_list)
        return action._formatted_action_invocation


def parse_args(mode: str = "predict") -> argparse.Namespace:
    desc = {
        "train": "Train a model with given config and checkpoint file",
        "predict": "Make prediction on images",
        "tracking": "Track spines in the whole stack",
    }
    parser = argparse.ArgumentParser(description=desc[mode], formatter_class=CustomHelpFormatter)

    if mode == "train":
        cfg_path = pkg_resources.resource_filename("spine_detection", "configs/model_config_paths.yaml")
        paths_cfg = load_config(cfg_path)
        parser.add_argument("-tr", "--train_csv", default=None, help="annotation file for training data")
        parser.add_argument(
            "-sp", "--special_term", default="", help="name appendix to store in different train folders"
        )
        parser.add_argument(
            "-m",
            "--model",
            help="Model used for prediction (without frozen_inference_graph.pb!) or folder "
            "where csv files are saved",
        )
        parser.add_argument(
            "-mt",
            "--model_type",
            choices=paths_cfg["model_paths"].keys(),
            help="decide which model to use as config and checkpoint file.",
        )
        parser.add_argument("-ms", "--model_suffix", help="Suffix of checkpoint model, usually starting with the date.")
        parser.add_argument(
            "-ua",
            "--use_aug",
            action="store_true",
            help="decide to load the config file with or without data augmentation",
        )
        parser.add_argument("-sd", "--seed_data", default=0, help="seed for the data loader, random if None")
        parser.add_argument("-sw", "--seed_weights", default=0, help="seed for initial random weights")
        # The following arguments are primary parameters for optimization
        # if there is no default defined, this parameter will take up the predefined value in the config file
        parser.add_argument("-lr", "--learning_rate", default=0.0005, type=float)
        parser.add_argument("-me", "--max_epochs", default=10, type=int)
        parser.add_argument("-wu", "--warm_up", default=None, help="learning rate warm up, use None to disable")
        parser.add_argument("-st", "--steps_decay", default=None, help="steps for lr decay")
        parser.add_argument("-mom", "--momentum", default=None, help="only for optimizer SGD")
        parser.add_argument("-wd", "--weight_decay", default=None, help="only for optimizer SGD")
        parser.add_argument("-do", "--dropout", default=None, help="overloading this parameter, varies by model type!")

        # The following arguments are secondary parameters used for data augmentation
        parser.add_argument("-prbc", "--p_rbc", default=None, help="p for RandomBrightnessContrast")
        parser.add_argument("-rb", "--random_brightness", default=None, help="from RandomBrightnessContrast")
        parser.add_argument("-rc", "--random_contrast", default=None, help="from RandomBrightnessContrast")
        parser.add_argument("-vf", "--vertical_flip", default=None, help="p for VerticalFlip")
        parser.add_argument("-hf", "--horizontal_flip", default=None, help="p for HorizontalFlip")
        parser.add_argument("-ro", "--rotate", default=None, help="p for Rotate")
        parser.add_argument("-r", "--resume", action="store_true", help="resume from checkpoint")
        parser.add_argument(
            "-cp",
            "--checkpoint",
            help="If resume flag is set, provide specific checkpoint, otherwise finding the checkpoint automatically will be tried",
        )
    else:
        parser.add_argument(
            "-m",
            "--model",
            help="Model used for prediction (without frozen_inference_graph.pb!) or folder "
            "where csv files are saved",
        )
        parser.add_argument(
            "-t", "--delta", help="Threshold for delta (detection threshold, score level)", default=0.5, type=float
        )
        parser.add_argument(
            "-th", "--theta", help="Threshold for theta (detection similarity threshold)", default=0.5, type=float
        )
        parser.add_argument(
            "-i ",
            "--input",
            help="Path to input image(s), ready for prediction. "
            'Path can contain wildcards but must start and end with "',
        )
        parser.add_argument(
            "-s", "--save_images", action="store_true", help="Activate this flag if images should be saved"
        )
        parser.add_argument(
            "-o", "--output", required=False, help="Path where prediction images and csvs should be saved"
        )

        # For load_model() use_aug, model_epoch
        parser.add_argument(
            "-mt",
            "--model_type",
            help="decide which model to use as config and checkpoint file. "
            "use one of [Cascade_RCNN, GFL, VFNet, Def_DETR]",
        )
        parser.add_argument(
            "-ua",
            "--use_aug",
            action="store_true",
            help="decide to load the config file with or without data augmentation",
        )
        parser.add_argument(
            "-me",
            "--model_epoch",
            default="epoch_1",
            help="decide the epoch number for the model weights. use the format of the default value",
        )
        parser.add_argument(
            "-pc",
            "--param_config",
            default="",
            help="string that contains all parameters intentionally tweaked during optimization",
        )
        if mode == "predict":
            parser.add_argument(
                "-C", "--use_csv", action="store_true", help="activate this flag if you want to use the given csv files"
            )

        if mode == "tracking":
            parser.add_argument(
                "-T",
                "--tif",
                required=False,
                help="Path to input tif stack, if this and image-flag are set, images are priorized",
            )
            parser.add_argument("-ta", "--tau", help="Threshold for tau (tracking threshold)", default=0.3, type=float)
            parser.add_argument("-a", "--appeared", help="appeared counter", default=0, type=int)
            parser.add_argument("-d", "--disappeared", help="disappeared counter", default=0, type=int)
            parser.add_argument(
                "-c",
                "--csv",
                required=False,
                help="Single file or folder of csv files for previous prediction. "
                "If this flag is set, no model prediction will be executed",
            )

            parser.add_argument("-f", "--file-save", help="Name of tracked data csv file", default="data_tracking")
            parser.add_argument(
                "-mc",
                "--metric",
                default="iom",
                help="Metric which should be used for evaluating. Currently available: iom, iou. "
                "Own metric can be implemented as lambda function which takes two arguments and returns one.",
            )
            parser.add_argument("-uo", "--use_offsets", default="False", help="whether offsets should be used or not")
            # For load_model()
            parser.add_argument(
                "-im",
                "--input_mode",
                default="Test",
                help="defines the proper way of loading either train, val or test data as input",
            )

    return parser.parse_args()

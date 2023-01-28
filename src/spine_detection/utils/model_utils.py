import glob
import logging
from pathlib import Path
from typing import Optional

import pkg_resources
import yaml
from mmcv import Config
from yacs.config import CfgNode as CN

logger = logging.getLogger(__name__)

from mmdet.apis import init_detector


def load_model(model_type: str, use_aug: bool, model_epoch: str, param_config: str, device: str = "cuda:0"):
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
    checkpoint_file = get_checkpoint_path(model_type, model_folder, use_aug, paths_cfg)
    config_file = get_config_path(model_type, use_aug, paths_cfg)
    logger.info("Loading model ...")

    checkpoint_file = str(Path(checkpoint_file) / param_config / (model_epoch + ".pth"))
    # construct checkpoint file name
    # checkpoint_file = os.path.join(checkpoint_file, os.path.join(param_config, model_epoch + ".pth"))

    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
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

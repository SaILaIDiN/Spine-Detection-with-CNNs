import logging
from collections import OrderedDict

import cv2
import numpy as np


def draw_boxes(img: np.ndarray, objects: OrderedDict) -> np.ndarray:
    """Draw boxes onto image
    Args:
        img (np.ndarray): image input to draw on
        objects (OrderedDict): Dictionary of objects of format (cX, cY, w, h, conf)
    Returns:
        np.ndarray: output image with drawn boxes
    """

    for key in objects:
        # w, h = 512, 512
        cX, cY, width, height, conf = objects[key]
        x1, x2 = int(cX - width / 2), int(cX + width / 2)
        y1, y2 = int(cY - height / 2), int(cY + height / 2)
        # correct colored rectangle
        # opencv : BGR!!!! NO RGB!!
        # linear from (0,0,255) to (255,255,0)

        # color = (255*(1-conf), 255*conf, 255*conf)
        color = (0, 255, 0)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

        # green filled rectangle for text
        color = (0, 255, 0)
        text_color = (0, 0, 0)
        img = cv2.rectangle(img, (x1, y1), (x1 + 25, y1 - 12), color, thickness=-1)

        # text
        img = cv2.putText(
            img, "{:02.0f}%".format(conf * 100), (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1
        )
    return img


def draw_boxes_predict(orig_img, boxes, scores, thresh=0.3, disable_thresh=False):
    """Draw detection boxes onto image
    Args:
        orig_img (np.ndarray): original image to draw on
        boxes (np.ndarray): detection boxes
        scores (np.ndarray): detection confidence scores
        thresh (float, optional): min confidence necessary to count as spine. Defaults to 0.3.
        disable_thresh (bool, optional): Flag whether to use differentiation by confidence score. Defaults to False.
    Returns:
        np.ndarray:
    """
    img = image_decode(img=orig_img)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        x1, y1, x2, y2 = image_decode(rect=(x1, y1, x2, y2))
        if not disable_thresh and conf < thresh:
            continue

        # correct colored rectangle
        # opencv : BGR!!!! NO RGB!!
        # linear from (0,0,255) to (255,255,0)
        # color = (255*(1-conf), 255*conf, 255*conf)
        color = (0, 255, 0)
        text_color = (0, 0, 0)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)

        # green filled rectangle for text and adding border as well
        # width of rect depends on width of text
        text_width = 23 if conf < 0.995 else 30
        img = cv2.rectangle(img, (x1, y1), (x1 + text_width, y1 - 12), color, thickness=-1)
        img = cv2.rectangle(img, (x1, y1), (x1 + text_width, y1 - 12), color, thickness=1)

        # text
        img = cv2.putText(
            img, "{:02.0f}%".format(conf * 100), (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1
        )
    return img


def image_load_encode(img_path):
    """Load image from path to 512x512 format
    Args:
        img_path (str): path to image file
    Returns:
        Tuple[np.ndarray, int, int]: image as np-array, its width and height
    """
    # function to read img from given path and convert to get correct 512x512 format
    # new_img = np.zeros((512, 512, 3))
    # new_img[:256, :] = image[:, :512]
    # new_img[256:, :] = image[:, 512:]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    return img.copy(), w, h


def image_decode(img=None, rect=None):
    """Reverse image encoding potentially applied to rects as well
    Args:
        img (Optional[np.ndarray]): input (512x512) image to decode
        rect (Optional[List]): rect in (x1, y1, x2, y2) format to decode
    Raises:
        AttributeError: At least img or rect must be not None to get a result
    Returns:
        np.ndarray: Depending on the non-None inputs decoded output of either img, rect or (img, rect)
    """
    # function to decode img or detection, depending which type is provided to get original img/detection back
    # rects have x/y values between 0 and 512 and are of type xmin, ymin, xmax, ymax
    # convert img back to 1024/256
    # img = np.zeros((256, 1024, 3))
    # img[:, :512] = orig_img[:256, :]
    # img[:, 512:] = orig_img[256:, :]
    if img is None and rect is None:
        raise AttributeError("At least one of img or rect need to have not None values.")
    if img is None:
        return np.array(rect).astype(int)
    if rect is None:
        return img
    else:
        return img, rect

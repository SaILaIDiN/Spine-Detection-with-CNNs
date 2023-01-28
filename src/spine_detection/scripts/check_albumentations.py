""" This script is to visualize different augmentations on the training data using albumentations. """

from os.path import join
from PIL import Image
# import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import VerticalFlip, HorizontalFlip, \
    RandomBrightnessContrast, GaussNoise, GaussianBlur
from albumentations.augmentations.geometric import Rotate


path = join("data", "raw", "person1", "SR052N1D1day1stack1-17.png")

img = Image.open(path)
img = img.resize((512, 512))
image = np.asarray(img)


# Original Image
original_image = Image.fromarray(image)
original_image.show()
original_image.save("albumentations_plots/original.png")

# # VerticalFlip
# transform = VerticalFlip(p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
# augmented_image.save("albumentations_plots/vertical_flip.png")
#
# # HorizontalFlip
# transform = HorizontalFlip(p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
# augmented_image.save("albumentations_plots/horizontal_flip.png")
#
# # Rotate (90 Degrees)
# transform = Rotate(limit=(90, 90), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
# augmented_image.save("albumentations_plots/rotate_90.png")
#
# # HorizontalFlip, VerticalFlip
# transform = Compose([HorizontalFlip(p=1.0), VerticalFlip(p=1.0)], p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
# augmented_image.save("albumentations_plots/h_flip_v_flip.png")
#
# # HorizontalFlip, Rotate (90 Degrees)
# transform = Compose([HorizontalFlip(p=1.0), Rotate(limit=(90, 90), p=1.0)], p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
# augmented_image.save("albumentations_plots/h_flip_rotate_90.png")
#
# # VerticalFlip, Rotate (90 Degrees)
# transform = Compose([VerticalFlip(p=1.0), Rotate(limit=(90, 90), p=1.0)], p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
# augmented_image.save("albumentations_plots/v_flip_rotate_90.png")
#
# # HorizontalFlip, VerticalFlip, Rotate (90 Degrees)
# transform = Compose([HorizontalFlip(p=1.0), VerticalFlip(p=1.0), Rotate(limit=(90, 90), p=1.0)], p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
# augmented_image.save("albumentations_plots/h_flip_v_flip_rotate_90.png")


# RandomBrightnessContrast
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Nothing")

# transform = RandomBrightnessContrast(brightness_limit=(-1.0, -1.0), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Brightness_NEG")
#
# transform = RandomBrightnessContrast(brightness_limit=(1.0, 1.0), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Brightness_POS")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(1.0, 1.0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Contrast_POS")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(-1.0, -1.0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Contrast_NEG")

# transform = RandomBrightnessContrast(brightness_limit=(-0.8, -0.8), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Brightness_NEG")
#
# transform = RandomBrightnessContrast(brightness_limit=(0.8, 0.8), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Brightness_POS")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(0.8, 0.8), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Contrast_POS")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(-0.8, -0.8), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Contrast_NEG")

# transform = RandomBrightnessContrast(brightness_limit=(-0.5, -0.5), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Brightness_NEG")
#
# transform = RandomBrightnessContrast(brightness_limit=(0.5, 0.5), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Brightness_POS")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(0.5, 0.5), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Contrast_POS")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(-0.5, -0.5), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show(title="Contrast_NEG")

# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(-0.2, -0.2), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/contrast_-02.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(-0.5, -0.5), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/contrast_-05.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(-0.8, -0.8), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/contrast_-08.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(-1.0, -1.0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/contrast_-10.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(0.2, 0.2), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/contrast_+02.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(0.5, 0.5), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/contrast_+05.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(0.8, 0.8), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/contrast_+08.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(1.0, 1.0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/contrast_+10.png")

# transform = RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show(title="Brightness_POS")
# augmented_image.save("albumentations_plots/brightness_+02.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(0.5, 0.5), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show(title="Brightness_POS")
# augmented_image.save("albumentations_plots/brightness_+05.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(0.8, 0.8), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show(title="Brightness_POS")
# augmented_image.save("albumentations_plots/brightness_+08.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(1.0, 1.0), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show(title="Brightness_POS")
# augmented_image.save("albumentations_plots/brightness_+10.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(-0.2, -0.2), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show(title="Brightness_POS")
# augmented_image.save("albumentations_plots/brightness_-0.2.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(-0.5, -0.5), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show(title="Brightness_POS")
# augmented_image.save("albumentations_plots/brightness_-05.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(-0.8, -0.8), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show(title="Brightness_POS")
# augmented_image.save("albumentations_plots/brightness_-08.png")
#
# transform = RandomBrightnessContrast(brightness_limit=(-1.0, -1.0), contrast_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show(title="Brightness_POS")
# augmented_image.save("albumentations_plots/brightness_-10.png")


# GaussNoise

# transform = GaussNoise(var_limit=(0, 2000), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
#
# transform = GaussNoise(var_limit=(0, 2000), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
#
# transform = GaussNoise(var_limit=(0, 2000), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()

# what we choose (but with (0, X))
transform = GaussNoise(var_limit=(25000, 25000), p=1.0)
augmented_image = transform(image=image)['image']
augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
augmented_image.save("albumentations_plots/noise_25000_p_10.png")

transform = GaussNoise(var_limit=(15000, 15000), p=1.0)
augmented_image = transform(image=image)['image']
augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
augmented_image.save("albumentations_plots/noise_15000_p_10.png")

transform = GaussNoise(var_limit=(10000, 10000), p=1.0)
augmented_image = transform(image=image)['image']
augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
augmented_image.save("albumentations_plots/noise_10000_p_10.png")

transform = GaussNoise(var_limit=(2000, 2000), p=1.0)
augmented_image = transform(image=image)['image']
augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
augmented_image.save("albumentations_plots/noise_2000_p_10.png")

transform = GaussNoise(var_limit=(1000, 1000), p=1.0)
augmented_image = transform(image=image)['image']
augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
augmented_image.save("albumentations_plots/noise_1000_p_10.png")

transform = GaussNoise(var_limit=(500, 500), p=1.0)
augmented_image = transform(image=image)['image']
augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
augmented_image.save("albumentations_plots/noise_500_p_10.png")

transform = GaussNoise(var_limit=(250, 250), p=1.0)
augmented_image = transform(image=image)['image']
augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
augmented_image.save("albumentations_plots/noise_250_p_10.png")

transform = GaussNoise(var_limit=(100, 100), p=1.0)
augmented_image = transform(image=image)['image']
augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
augmented_image.save("albumentations_plots/noise_100_p_10.png")


# # # GaussianBlur

# # Check blur_limit
# transform = GaussianBlur(blur_limit=(1, 1), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()

# transform = GaussianBlur(blur_limit=(3, 3), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()

# transform = GaussianBlur(blur_limit=(5, 5), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
#
# transform = GaussianBlur(blur_limit=(7, 7), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
#
# transform = GaussianBlur(blur_limit=(9, 9), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
#
# transform = GaussianBlur(blur_limit=(11, 11), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()

# transform = GaussianBlur(blur_limit=(19, 19), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()

# # check sigma_limit
# transform = GaussianBlur(blur_limit=(7, 7), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
#
# transform = GaussianBlur(blur_limit=(7, 7), sigma_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()

# transform = GaussianBlur(blur_limit=(7, 7), sigma_limit=(2, 2), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
#
# transform = GaussianBlur(blur_limit=(7, 7), sigma_limit=(5, 5), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()

# transform = GaussianBlur(blur_limit=(19, 19), sigma_limit=(0, 0), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
#
# transform = GaussianBlur(blur_limit=(19, 19), sigma_limit=(20, 20), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()
#
# transform = GaussianBlur(blur_limit=(19, 19), sigma_limit=(200, 200), p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# augmented_image.show()

# what we choose (but with (0, X))
# transform = GaussianBlur(blur_limit=(3, 3), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/blur_3_s_0_p_10.png")
#
# transform = GaussianBlur(blur_limit=(7, 7), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/blur_7_s_0_p_10.png")
#
# transform = GaussianBlur(blur_limit=(13, 13), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/blur_13_s_0_p_10.png")
#
# transform = GaussianBlur(blur_limit=(19, 19), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/blur_19_s_0_p_10.png")
#
# transform = GaussianBlur(blur_limit=(35, 35), sigma_limit=0, p=1.0)
# augmented_image = transform(image=image)['image']
# augmented_image = Image.fromarray(augmented_image)
# # augmented_image.show()
# augmented_image.save("albumentations_plots/blur_35_s_0_p_10.png")

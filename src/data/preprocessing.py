import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import os
import re

from skimage import io
from PIL import Image

height, width = 256, 256


def labels_encoding(labels: np.ndarray):
    new_labels = np.zeros((labels.shape[0], 256, 256, 2))
    for i in range(len(labels)):
        for j in range(256):
            for k in range(256):
                if labels[i][j][k] == 0:
                    new_labels[i][j][k][0] = 1
                else:
                    new_labels[i][j][k][1] = 1
    return new_labels


def labeled_image_to_color(image):
    output = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            output[i][j] = np.argmax(image[i][j])

    final_image = np.zeros((height, width, 3), dtype=np.uint8)
    for j in range(height):
        for k in range(width):
            if output[j][k] == 0:
                final_image[j][k] = [0, 0, 0]
            else:
                final_image[j][k] = [255, 255, 255]
    return final_image


def resize_file(src: str, dest: str):
    image = Image.open(src)
    image = image.crop((0, 200, 640, 840))
    image = image.resize((height, width))
    plt.imshow(image, cmap='gray')
    image.save(dest)


def preprocessing_files(src: str, dest: str):
    files = [src + n for n in os.listdir(src)]
    files = sorted(files, key=lambda file: int(re.findall('\d+', file.split('_')[-1])[0]))
    for i, f in enumerate(files):
        resize_file(f, dest + str(i + 1) + '.png')


def filter_images(src: str):
    files = [src + str(i) + '.png' for i in range(1, 86)]
    for f in files:
        processed_image = cv2.medianBlur(cv2.imread(f), 5)
        cv2.imwrite(f, processed_image)


def resize_image(src: str):
    image = io.imread(src, as_gray=True)
    image = Image.fromarray(image)
    image = image.crop((0, 200, 640, 840))
    image = image.resize((height, width))
    image = np.asarray(image)
    image = image.reshape((1, height, width, 1))
    return image


def preprocessing_images(src: str):
    files = [src + n for n in os.listdir(src)]
    return [resize_image(f) for f in sorted(files, key=lambda file: int(re.findall('\d+', file.split('_')[-1])[0]))]

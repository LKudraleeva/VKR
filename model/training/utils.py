import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def plot_images_and_labels(img_path, label_path):
    image = io.imread(img_path, as_gray=True)
    label = io.imread(label_path)

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='gray')
    plt.title('Маска')

    plt.show()


def labeled_image_to_color(image):
    output = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            output[i][j] = np.argmax(image[i][j])

    final_image = np.zeros((256, 256, 3), dtype=np.uint8)
    for j in range(256):
        for k in range(256):
            if output[j][k] == 0:
                final_image[j][k] = [0, 0, 0]
            else:
                final_image[j][k] = [255, 255, 255]
    return final_image

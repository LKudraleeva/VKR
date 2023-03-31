import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_image_from_file(filepath: str):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


def find_contour(image: np.ndarray):
    height = image.shape[0]
    width = image.shape[1]
    contour = np.array([[i, height - 1 - np.where(image[:, i] == 255)[0][0]] for i in range(width)
                        if np.any(image[:, i] == 255)])
    return contour


def draw_contour(contour: np.ndarray, height: int = 256, width: int = 256):
    x, y = contour[:, 0], contour[:, 1]
    plt.plot(x, y)
    plt.ylim((0, height))
    plt.xlim((0, width))
    plt.show()


def find_interpolation(contour: np.ndarray, width: int = 256):
    x, y = contour[:, 0], contour[:, 1]
    t = np.polyfit(x, y, 3)
    f = np.poly1d(t)
    x_new = np.arange(width)
    y_new = np.around(f(x_new)).astype(int)
    return np.stack((x_new, y_new), axis=-1)


def draw_contour_with_interpolation(contour: np.ndarray,
                                    interpolation: np.ndarray,
                                    h: int = 256,
                                    w: int = 256):
    x, y = contour[:, 0], contour[:, 1]
    x_i, y_i = interpolation[:, 0], interpolation[:, 1]
    fig = plt.figure(figsize=(8, 8))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes((left, bottom, width, height))
    ax.set(facecolor="black")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.plot(x, y, 'w')
    ax.plot(x_i, y_i, 'w')
    plt.show()


def get_dispersion(contour: np.ndarray, interpolation: np.ndarray):
    if contour.shape[0] == interpolation.shape[0]:
        return sum((contour[:, 1] - interpolation[:, 1]) ** 2) / interpolation.shape[0]


if __name__ == '__main__':
    pigment_filenames = []
    for p in range(1, 2):
        path_labels = os.listdir('labeled_images/' + str(p) + '/Pigment/')
        for n in path_labels:
            pigment_filenames.append('labeled_images/' + str(p) + '/Pigment/' + n)
    images = [get_image_from_file(file) for file in pigment_filenames]

    value_h = 0
    print('Healthy')
    d_h = []
    for im in images:
        cont = find_contour(im)
        inter = find_interpolation(cont, im.shape[1])
        # draw_contour_with_interpolation(cont, inter, im.shape[0], im.shape[1])
        d = get_dispersion(cont, inter)
        if d > 30:
            draw_contour_with_interpolation(cont, inter, im.shape[0], im.shape[1])
        # print(d)
        d_h.append(d)
        value_h += d

    n_bins = len(d_h)
    ran = [i for i in range(85)]
    plt.plot(d_h)
    plt.show()
    print(value_h / len(images))

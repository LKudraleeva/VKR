import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def find_contour(image: np.ndarray):
    height = image.shape[0]
    width = image.shape[1]
    contour = np.array([[i, height - 1 - np.where(image[:, i] == 255)[0][0]] for i in range(width)
                        if np.any(image[:, i] == 255)])
    return contour


def draw_contour(contour: np.ndarray, height: int = 256, width: int = 256):
    x, y = contour[:, 0], contour[:, 1]
    plt.plot(x, y)
    plt.xlim((0, width))
    plt.ylim((0, height))
    plt.show()


def find_interpolation(contour: np.ndarray, width: int = 256, degree: int = 3):
    x, y = contour[:, 0], contour[:, 1]
    t = np.polyfit(x, y, degree)
    f = np.poly1d(t)
    x_new = np.arange(width)
    y_new = np.around(f(x_new)).astype(int)
    return np.stack((x_new, y_new), axis=-1)


def draw_contour_with_interpolation(contour: np.ndarray,
                                    interpolation: np.ndarray,
                                    height: int = 256,
                                    width: int = 256):
    x, y = contour[:, 0], contour[:, 1]
    x_i, y_i = interpolation[:, 0], interpolation[:, 1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set(facecolor="black")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.plot(x, y, 'w')
    ax.plot(x_i, y_i, 'w')
    plt.show()


def get_dispersion(contour: np.ndarray, interpolation: np.ndarray):
    if contour.shape[0] == interpolation.shape[0]:
        return sum((contour[:, 1] - interpolation[:, 1]) ** 2) / interpolation.shape[0]
    else:
        dispersion = []
        for i in range(contour.shape[0]):
            dispersion.append((contour[i, 1] - interpolation[i, 1]) ** 2)
        return sum(dispersion)/contour.shape[0]


if __name__ == '__main__':

    pigment_filenames = []
    for p in range(5, 6):
        path_name = 'ImagesWithLabels/' + str(p) + '/ResizedPigments/'
        for n in os.listdir(path_name):
            pigment_filenames.append(path_name + n)

    pigment_filenames = sorted(pigment_filenames, key=lambda file: int(file.split('/')[-1].split('.')[0]))

    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in pigment_filenames]

    disp = []
    for im in images:
        cont = find_contour(im)
        inter = find_interpolation(cont, im.shape[1])
        disp.append(get_dispersion(cont, inter))

    gauss = find_interpolation(np.stack((np.arange(85), np.asarray(disp)), axis=-1), 85, 3)
    plt.plot(np.arange(85), np.asarray(disp), gauss[:, 0], gauss[:, 1])
    plt.show()

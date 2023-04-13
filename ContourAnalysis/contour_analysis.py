import matplotlib.pyplot as plt
from PIL import Image
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

    plt.savefig('output.png')
    plt.show()


def get_image(line: np.ndarray, height: int = 256, width: int = 256):
    x, y = line[:, 0], line[:, 1]
    output = np.zeros((height, width), dtype=np.uint8)
    for i, j in zip(y, x):
        output[i][j] = 255
    img = Image.fromarray(output)
    img.save('output.png')


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
        dispersion = [(contour[i, 1] - interpolation[i, 1]) ** 2 for i in range(contour.shape[0])]
        return sum(dispersion)/contour.shape[0]


def median_filter(rang: np.array, size: int = 3):
    s = size//2
    rang = np.insert(rang, 0, rang[0] * s)
    rang = np.insert(rang, -1, rang[-1] * s)
    return [np.median(rang[i:i+size]) for i in range(0, rang.shape[0]-size+1)]


if __name__ == '__main__':

    path_name = 'C:/Users/user/PycharmProjects/VKR/data/labeled_images/4/Resize/'
    filenames = [path_name + str(i) + '.png' for i in range(1, 86)]
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in filenames]
    disp = []
    for im in images:
        cont = find_contour(im)
        inter = find_interpolation(cont, im.shape[1])
        disp.append(get_dispersion(cont, inter))

    x = np.arange(85)
    y = np.array(disp)

    plt.plot(x, y)
    plt.show()



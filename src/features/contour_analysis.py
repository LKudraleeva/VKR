import matplotlib.pyplot as plt
import numpy as np
import cv2

from PIL import Image


def find_contour(image: np.ndarray):
    height = image.shape[0]
    width = image.shape[1]
    return np.array([[i, height - 1 - np.where(image[:, i] == 255)[0][0]] for i in range(width)
                     if np.any(image[:, i] == 255)])


def draw_contour(contour: np.ndarray,
                 height: int = 256,
                 width: int = 256):
    x, y = contour[:, 0], contour[:, 1]
    plt.plot(x, y)
    plt.xlim((0, width))
    plt.ylim((0, height))
    plt.show()


def get_image(line: np.ndarray,
              height: int = 256,
              width: int = 256):
    x, y = line[:, 0], line[:, 1]
    output = np.zeros((height, width), dtype=np.uint8)
    for i, j in zip(y, x):
        output[i][j] = 255
    img = Image.fromarray(output)
    return img


def find_interpolation(contour: np.ndarray,
                       width: int = 256,
                       degree: int = 3) -> np.ndarray:
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
    ax.plot(x, y, 'w', x_i, y_i, 'w')
    plt.show()


def get_dispersion(contour: np.ndarray,
                   interpolation: np.ndarray) -> float:
    return sum([(contour[i, 1] - interpolation[contour[i, 0], 1]) ** 2
                for i in range(contour.shape[0])]) / contour.shape[0]


def median_filter(arr: np.ndarray,
                  size: int = 3) -> np.ndarray:
    s = size // 2
    arr = np.insert(arr, 0, arr[0] * s)
    arr = np.insert(arr, -1, arr[-1] * s)
    return np.array([np.median(arr[i:i + size]) for i in range(0, arr.shape[0] - size + 1)])


def extra_filter(arr: np.array,
                 size: int = 3) -> np.ndarray:
    result = []
    s = size // 2
    arr = np.insert(arr, 0, arr[0] * s)
    arr = np.insert(arr, -1, arr[-1] * s)
    for i in range(1, len(arr) - 1):
        if abs(arr[i + 1] - arr[i]) > abs(arr[i - 1] - arr[i]):
            result.append(arr[i - 1])
        elif abs(arr[i + 1] - arr[i]) < abs(arr[i - 1] - arr[i]):
            result.append(arr[i + 1])
        else:
            result.append(arr[i])
    return np.array(result)


if __name__ == '__main__':

    path_names = ['C:/Users/user/PycharmProjects/VKR/data/vmd/1/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/vmd/2/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/vmd/3/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/vmd/4/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/vmd/5/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/vmd/6/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/vmd/7/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/vmd/8/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/vmd/9/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/vmd/10/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/vmd/11/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/vmd/12/Preds/',
                  'C:/Users/user/PycharmProjects/VKR/data/labeled_images/5/Resize/',
                  'C:/Users/user/PycharmProjects/VKR/data/labeled_images/6/Resize/',
                  'C:/Users/user/PycharmProjects/VKR/data/labeled_images/4/Resize/',
                  ]

    # path_names = ['C:/Users/user/PycharmProjects/VKR/data/labeled_images/1/Resize/',
    #               'C:/Users/user/PycharmProjects/VKR/data/labeled_images/2/Resize/',
    #               'C:/Users/user/PycharmProjects/VKR/data/labeled_images/3/Resize/',
    #               'C:/Users/user/PycharmProjects/VKR/data/labeled_images/4/Resize/'
    #               ]

    ex_array, dx_array = [], []
    label_array = [1] * 4

    for idx, path_name in enumerate(path_names):
        filenames = [path_name + str(idx) + '.png' for idx in range(1, 86)]
        images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in filenames]
        disp = []
        for im in images:
            cont = find_contour(im)
            inter = find_interpolation(cont, im.shape[1], 3)
            disp.append(get_dispersion(cont, inter))

        disp = np.array(disp)
        # disp = np.array(extra_filter(extra_filter(disp)))

        ex = sum(disp) / disp.shape[0]
        dx = (sum((disp - ex) ** 2) / disp.shape[0]) ** 1 / 2
        ex_array.append(ex)
        dx_array.append(dx)
        plt.subplot(5, 3, idx+1)
        plt.plot(disp, label='Исходное')

        # plt.fill_between(np.arange(1, 86),
        #                  np.array([ex + dx] * 85),
        #                  np.array([ex - dx] * 85),
        #                  facecolor='r',
        #                  alpha=0.5,)
        # plt.plot(median_filter(disp), label='Медианный')
        # plt.plot(x, np.convolve(extra_filter(extra_filter(extra_filter(y))), [1 / 3, 1 / 3, 1 / 3], 'same'))
        plt.plot(np.arange(1, 86), extra_filter(extra_filter(extra_filter(disp))), label='Экстремальный 3')
        # plt.plot(np.arange(1, 86), np.convolve(disp, [1 / 3, 1 / 3, 1 / 3], 'same'), label='3x1')
        y2 = extra_filter(extra_filter(extra_filter(disp)))
        ex2 = sum(y2) / y2.shape[0]

        plt.plot([ex] * 85, label='МО', c='r')

        plt.plot([ex2] * 85, label='МО экстремальный')
        # plt.legend()
    plt.show()


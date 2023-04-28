import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from PIL import Image

LEFT = 20
RIGHT = 64


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


def get_index(dispersion):
    mx = sum(dispersion) / dispersion.shape[0]
    return np.argwhere(dispersion > mx)[0][0], np.argwhere(dispersion > mx)[-1][0]


def get_interval(dirs):
    start, finish = 0, 0
    for idx, path_name in enumerate(dirs):
        filenames = [path_name + str(idx) + '.png' for idx in range(1, 86)]
        images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in filenames]
        disp = scan_analiz(images)
        start += get_index(disp)[0]
        finish += get_index(disp)[1]

    return int(np.around(start / len(dirs))), int(np.around(finish / len(dirs)))


def get_stats_in_interval(dirs, start, finish):
    ex_array, dx_array = [], []

    for idx, path_name in enumerate(dirs):
        filenames = [path_name + str(idx) + '.png' for idx in range(start, finish + 1)]
        images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in filenames]
        disp = scan_analiz(images)
        ex = sum(disp) / disp.shape[0]
        ex_array.append(sum(disp) / disp.shape[0])
        dx_array.append((sum((disp - ex) ** 2) / disp.shape[0]) ** 1 / 2)
        # print(ex, (sum((disp - ex) ** 2) / disp.shape[0]) ** 1 / 2)

    return np.array([ex_array, dx_array])


def scan_analiz(images):
    disp = []
    for im in images:
        cont = find_contour(im)
        inter = find_interpolation(cont, im.shape[1], 4)
        disp.append(get_dispersion(cont, inter))
    return np.array(disp)


def classificator(people):
    X = people[:, 0:2].astype('float32')
    y = people[:, 2].astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    logr = linear_model.LogisticRegression()
    logr.fit(X_train, y_train)

    scores = cross_val_score(logr, X, y, cv=10)
    print('Cross-Validation Accuracy Scores', scores)
    scores = pd.Series(scores)
    print(scores.mean())


if __name__ == '__main__':
    ill = ['C:/Users/user/PycharmProjects/VKR/data/vmd/' + str(i) + '/Predictions/' for i in range(1, 14)]
    normal = ['C:/Users/user/PycharmProjects/VKR/data/normal/' + str(i) + '/Predictions/' for i in range(1, 16)]
    a = 20
    b = 64
    ill_stats = get_stats_in_interval(ill, a, b).T
    normal_stats = get_stats_in_interval(normal, a, b).T

    plt.scatter(ill_stats[:, 0], ill_stats[:, 1], label='ВМД')
    plt.scatter(normal_stats[:, 0], normal_stats[:, 1], label='Здоровые')
    plt.legend()
    plt.xlim(-1, 12)
    plt.ylim(-1, 12)
    plt.xlabel('Мат. ожидание')
    plt.ylabel('Дисперсия')
    plt.grid()
    plt.show()

    ill_stats = np.column_stack((ill_stats, [1] * 13))
    normal_stats = np.column_stack((normal_stats, [0] * 15))
    all_people = np.append(ill_stats, normal_stats, axis=0)
    classificator(all_people)

    # for idx, path_name in enumerate(path_names):
    #     filenames = [path_name + str(idx) + '.png' for idx in range(20, 65)]
    #     images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in filenames]
    #     disp = []
    #     for im in images:
    #         cont = find_contour(im)
    #         inter = find_interpolation(cont, im.shape[1], 4)
    #         disp.append(get_dispersion(cont, inter))

    #     disp = np.array(disp)
    #     # disp = np.array(extra_filter(extra_filter(disp)))
    #
    #     ex = sum(disp) / disp.shape[0]
    #     dx = (sum((disp - ex) ** 2) / disp.shape[0]) ** 1 / 2
    #     ex_array.append(ex)
    #     dx_array.append(dx)
    #     plt.subplot(5, 3, idx + 1)
    #     # plt.plot(disp, label='Исходное')
    #     print(ex, dx)
    #
    #     # plt.plot(median_filter(disp), label='Медианный')
    #     # plt.plot(x, np.convolve(extra_filter(extra_filter(extra_filter(y))), [1 / 3, 1 / 3, 1 / 3], 'same'))
    #     plt.plot(np.arange(20, 65), extra_filter(extra_filter(extra_filter(disp))), label='Экстремальный 3')
    #     y2 = extra_filter(extra_filter(extra_filter(disp)))
    #     ex2 = sum(y2) / y2.shape[0]
    #
    #     # plt.plot([ex] * 85, label='МО', c='r')
    #
    #     plt.plot(np.arange(20, 65), [ex2] * 45, label='МО экстремальный')
    #     # plt.legend()

    # plt.show()

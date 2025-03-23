import os
import time

import numba
from matplotlib import pyplot as plt
from numba import jit, stencil, uint8, uint16
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view


def pic_to_array(path: str):
    return np.array(Image.open(path).convert('L'))


def save_img(array, name):
    im = Image.fromarray(array)
    im.save(f"{name}.jpeg")


def filter2d(img, func, w_size=3):
    assert w_size % 2 == 1

    pad_width = w_size // 2
    v = sliding_window_view(np.pad(img, [pad_width, pad_width]), (w_size, w_size))
    res = np.zeros(img.shape, dtype=np.uint8)
    for i, j in np.ndindex(img.shape):
        s = func(v[i, j])
        res[i, j] = s

    return res


def get_weigths(w_size):
    assert w_size % 2 == 1
    c = w_size // 2
    center = c + 1j*c
    weights = np.zeros((w_size, w_size), dtype=float)
    for i in range(w_size):
        for j in range(w_size):
            distance = np.abs(center - (i + 1j*j))
            weights[i, j] = distance**1.0

    weights[c, c] = 0.5 * np.min(weights[weights > 0])
    weights = 1 / weights
    # print(weights)

    return weights


if __name__ == '__main__':

    # p = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\DIP3E_Original_Images_CH03\Fig0333(a)(test_pattern_blurring_orig).tif'
    #
    # for w_size in [3, 5, 9, 15, 25]:
    #     print(f"Однородное среднее {w_size}x{w_size} ...")
    #
    #     img = pic_to_array(p)
    #     res = filter2d(img, np.mean, w_size)
    #     save_img(res, f"../Results/homo_blur_{w_size}x{w_size}")
    #
    # for w_size in [3, 5, 9, 15, 25]:
    #     print(f"Взвешенное среднее {w_size}x{w_size} ...")
    #
    #     w = get_weigths(w_size)
    #     def weighted_average(x):
    #         return np.sum(x * w) / np.sum(w)
    #
    #     img = pic_to_array(p)
    #     res = filter2d(img, weighted_average, w_size)
    #     save_img(res, f"../Results/weighted_blur_{w_size}x{w_size}")

    p = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\DIP3E_Original_Images_CH03\Fig0334(a)(hubble-original).tif'
    img = pic_to_array(p)

    res = filter2d(img, np.mean, w_size=15)
    counts, bins = np.histogram(res, range(257))

    plt.plot(np.cumsum(counts) / np.prod(img.shape) * 100)
    plt.title("Доля пикселей с яркостью не превышающей порог")
    plt.ylabel("Доля пикселей, % ")
    plt.xlabel("Порог")
    # plt.axvline(56, 0, 100, c='r')
    # plt.xlim(0, 100)
    # plt.ylim(50, 100)
    plt.show()

    def get_threshold(remove_pixel_percent):
        i_bin = np.where((np.cumsum(counts) / np.prod(img.shape)) * 100 > remove_pixel_percent)[0][0]
        thresh = bins[i_bin]
        print(f"Порог для отсеивания {remove_pixel_percent}% пикселей равен {thresh}")
        return thresh

    for _threshold in [get_threshold(95)]:
        print(f"Пример про снимок с телескопа Хаббл, _threshold={_threshold} ...")

        def smoothing_and_thresholding(window):
            return 255 if np.mean(window) > _threshold else 0

        res = filter2d(img, smoothing_and_thresholding, w_size=15)
        save_img(res, f"../Results/hubble_treshold_{_threshold}")

    # path = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\DIP3E_Original_Images_CH03\Fig0335(a)(ckt_board_saltpep_prob_pt05).tif'
    # img = pic_to_array(path)
    #
    # print("Нелинейная фильтрация (медианная) ...")
    # res = filter2d(img, np.median)
    # save_img(res, "../Results/salt_peper")


    # Методы, основанны на производных
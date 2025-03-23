import os
import time

import numba
from matplotlib import pyplot as plt
from numba import jit, stencil
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@jit
def f_mu0(delta):
    F = 15
    if delta >= F:
        return 0
    if delta <= -F:
        return 1
    return (F - delta) / (2*F)


@jit
def f_mu1(delta):
    F = 15
    if delta >= F:
        return 0
    if delta <= -F:
        return 1
    return (F + delta) / (2*F)


@jit
def my_lbp_looped(img):
    _vals = np.array([
        [1, 2, 4],
        [8, 0, 16],
        [32, 64, 128]
    ])
    res = np.zeros((img.shape[0]-2, img.shape[1]-2), dtype=np.uint8)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            c = img[i, j]
            for row in range(i-1, i+2):
                for col in range(j-1, j+2):
                    res[i - 1, j - 1] += (img[row, col] >= c) * _vals[row-i+1, col-j+1]
    return res


@jit
def my_flbp_looped(img):
    _vals = np.array([
        [1, 2, 4],
        [8, 0, 16],
        [32, 64, 128]
    ])
    res = np.zeros((img.shape[0]-2, img.shape[1]-2, 2), dtype=np.uint8)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            c = numba.int32(img[i, j])
            for row in range(i-1, i+2):
                for col in range(j-1, j+2):
                    delta = img[row, col] - c
                    # if delta < 0:
                    #     print("my_flbp_looped: delta < 0")
                    mu0 = f_mu0(delta)
                    mu1 = f_mu1(delta)
                    res[i - 1, j - 1, 0] += round(mu0) * _vals[row-i+1, col-j+1]
                    res[i - 1, j - 1, 1] += round(mu1) * _vals[row-i+1, col-j+1]
    return res


if __name__ == '__main__':

    img = np.array([
        [90,  200, 120],
        [180, 142, 100],
        [182, 181, 152]
    ], dtype=np.uint8)

    x = my_flbp_looped(img)

    path_train0 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\Spoof_data\Training Biometrika Live\live'
    path_train1 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\Spoof_data\Training Biometrika Spoof\Training Biometrika Spoof\spoof'
    path_test0 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\Spoof_data\Testing Biometrika Live\live'
    path_test1 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\Spoof_data\Testing Biometrika Spoof\Testing Biometrika Spoof\spoof'

    def pic_to_array(path: str):
        return np.array(Image.open(path).convert('L'))

    def prepare_data(path, X):
        i = 0
        for subdir, dirs, files in os.walk(path):
            for file in files:
                img_path = subdir + os.sep + file
                img = pic_to_array(img_path)

                t = time.time()
                out = my_lbp_looped(img)
                print(time.time() - t)

                out = np.reshape(out, -1)

                assert len(out) == X.shape[1]
                X[i, :] = out

                i += 1
        return X

    def get_n_files(path):
        cnt = 0
        for subdir, dirs, files in os.walk(path):
            cnt += len(files)
        return cnt


    img_shape = (372, 312)
    n_features = np.prod(np.array(img_shape)-2)

    n0 = get_n_files(path_train0)
    n1 = get_n_files(path_train1)
    n2 = get_n_files(path_test0)
    n3 = get_n_files(path_test1)

    tmp0 = np.empty((n0, n_features), dtype=int)
    tmp1 = np.empty((n1, n_features), dtype=int)
    tmp2 = np.empty((n2, n_features), dtype=int)
    tmp3 = np.empty((n3, n_features), dtype=int)

    prepare_data(path_train0, tmp0)
    prepare_data(path_train1, tmp1)
    prepare_data(path_test0, tmp2)
    prepare_data(path_test1, tmp3)

    X_train = np.concatenate([tmp0, tmp1])
    y_train = np.concatenate([np.zeros(len(tmp0)), np.ones(len(tmp1))])
    X_test = np.concatenate([tmp2, tmp3])
    y_test = np.concatenate([np.zeros(len(tmp2)), np.ones(len(tmp3))])

    # np.save('X_train.npy', X_train)
    # np.save('y_train.npy', y_train)
    # np.save('X_test.npy', X_test)
    # np.save('y_test.npy', y_test)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    print(classification_report(y_test, y_predicted))

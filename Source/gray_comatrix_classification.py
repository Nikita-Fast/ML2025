import os
import sys
import time

from skimage.feature import graycomatrix, graycoprops
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from numba import jit, njit

if __name__ == '__main__':
    np.random.seed(13)

    img = np.array([
        [0,0,1,1],
        [0,0,1,1],
        [0,2,2,2],
        [2,2,3,3]
    ])
    P = graycomatrix(img, distances=[1], angles=[0, -np.pi/4, -np.pi/2, -3*np.pi/4], levels=4, symmetric=True)
    m = P[:,:,0,:]
    s = np.sum(m, axis=(0,1))
    s = np.ones(4)
    m0 = m[..., 0] / s[0]
    m45 = m[..., 1] / s[1]
    m90 = m[..., 2] / s[2]
    m135 = m[..., 3] / s[3]

    def mean_x(mat):
        row_sum = np.sum(mat, axis=1)
        return np.sum(row_sum * np.arange(mat.shape[0]))

    def mean_y(mat):
        col_sum = np.sum(mat, axis=0)
        return np.sum(col_sum * np.arange(mat.shape[1]))

    def var_x(mat):
        u_x = mean_x(mat)
        tmp = (np.arange(mat.shape[0]) - u_x) ** 2
        row_sum = np.sum(mat, axis=1)
        return np.sum(tmp * row_sum)

    def var_y(mat):
        u_y = mean_y(mat)
        tmp = (np.arange(mat.shape[0]) - u_y) ** 2
        col_sum = np.sum(mat, axis=0)
        return np.sum(tmp * col_sum)

    def f1_odnorodnost(mat):
        return np.sum(mat**2)

    def f2_contrast(mat):
        N_g = mat.shape[0]
        ij_dist = np.abs(np.arange(N_g).reshape(-1, 1) - np.arange(N_g))
        _contrast = 0
        for n in range(N_g):
            _contrast += n**2 * np.sum(mat[ij_dist == n])
        return _contrast

    def f3_corr(mat):
        u_x, u_y = mean_x(mat), mean_y(mat)
        sigma_x, sigma_y = var_x(mat), var_y(mat)
        N_g = mat.shape[0]
        ij = np.arange(N_g).reshape(-1, 1) * np.arange(N_g)
        return (np.sum(ij * mat) - u_x * u_y) / (np.sqrt(sigma_x) * np.sqrt(sigma_y))


    def f4_entropy(mat):
        return -1 * np.sum(mat * np.log(mat, where=mat > 0))


    def pic_to_array(path: str):
        return np.array(Image.open(path).convert('L'))


    @jit
    def my_graycomatrix(img, N_g, angle, res):
        rows_shift = 0 if angle == 0 else -1
        cols_shift = {0: 1, 1: 1, 2: 0, 3: -1}[angle]

        col_start = 0 if cols_shift >= 0 else -1*cols_shift
        col_stop = img.shape[1] - 1 if cols_shift == 1 else img.shape[1]

        for i in range(N_g):
            for j in range(N_g):
                cnt = 0
                for ii in range(-1*rows_shift, img.shape[0]):
                    for jj in range(col_start, col_stop):
                        ii_shift = ii + rows_shift
                        jj_shift = jj + cols_shift

                        p1 = img[ii_shift, jj_shift]
                        p2 = img[ii, jj]

                        if (p1, p2) == (i, j):
                            cnt += 1
                        if (p1, p2) == (j, i):
                            cnt += 1
                res[i, j] = cnt
        return res

    # img = np.array([
    #     [0, 0, 3, 1, 3],
    #     [1, 3, 0, 0, 1],
    #     [0, 3, 1, 0, 2],
    #     [1, 2, 0, 2, 1],
    #     [2, 0, 3, 0, 2]])
    #
    # actual = np.zeros((4, 4), dtype=int)
    # my_graycomatrix(img, 4, 0, actual)
    # print("actual")
    # print(actual)
    # print()
    # print("expected")
    # expected = graycomatrix(img, [1], [0], 4, symmetric=True)
    # print(expected[:,:,0,0])

    def test_my_graycomatrix():
        N_g = 64
        for _ in range(10):
            img_shape = np.random.randint(10, 32, 2)
            img = np.random.randint(0, N_g, np.prod(img_shape)).reshape(img_shape)
            for angle in range(4):
                t = time.time()
                m_expected = graycomatrix(img, [1], [-1 *angle * np.pi / 4], levels=N_g, symmetric=True)[:,:,0,0]
                t_expected = time.time() - t

                m_actual = np.zeros((N_g, N_g), dtype=int)
                t = time.time()
                my_graycomatrix(img, N_g, angle, m_actual)
                print(t_expected, time.time() - t)
                assert np.allclose(m_expected, m_actual)
        print("tests passed")


    test_my_graycomatrix()

    # print(mean_x(m0), mean_y(m0), var_x(m0), var_y(m0))
    # print(f"{'f1_odnorodnost':.<20s}", f1_odnorodnost(m0))
    # print(f"{'f2_contrast':.<20s}", f2_contrast(m0))
    # print(f"{'f3_corr':.<20s}", f3_corr(m0))
    # print(f"{'f4_entropy':.<20s}", f4_entropy(m0))
    #
    # print("----------------")
    # for p in ['contrast', "homogeneity", 'correlation', 'ASM', 'mean', 'variance', 'entropy']:
    #     print(f"{p:.<20s}", graycoprops(P, p)[0,0])

    path_train0 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\Spoof_data\Training Biometrika Live\live'
    path_train1 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\Spoof_data\Training Biometrika Spoof\Training Biometrika Spoof\spoof'
    path_test0 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\Spoof_data\Testing Biometrika Live\live'
    path_test1 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\Spoof_data\Testing Biometrika Spoof\Testing Biometrika Spoof\spoof'

    # dataset link https://github.com/abin24/Textures-Dataset?tab=readme-ov-file
    # pp1 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\KTH\train\KTH_corduroy'
    # pp2 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\KTH\train\KTH_cotton'
    # pp3 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\KTH\valid\KTH_corduroy'
    # pp4 = r'D:\Projects\pythonProject\pythonProject\ML2025\Resources\KTH\valid\KTH_cotton'
    #
    def prepare_data(path, X):
        _res = np.zeros((256, 256))
        for subdir, dirs, files in os.walk(path):
            for i, file in enumerate(files):
                img_path = subdir + os.sep + file
                img = pic_to_array(img_path)

                # P = graycomatrix(img, distances=[1], angles=[0, -np.pi / 4, -np.pi / 2, -3 * np.pi / 4], levels=256, symmetric=True)
                glcm = my_graycomatrix(img, 256, 0, _res)

                props = ['contrast', 'correlation', 'entropy', 'ASM']
                n_props = len(props)
                for prop_i, prop in enumerate(props):
                    X[i][n_props*prop_i: n_props*(prop_i+1)] = graycoprops(P, prop)[0]

    n_features = 4
    # X_train0 = np.zeros((200, 4 * n_features), dtype=float)
    # X_train1 = np.zeros((207, 4 * n_features), dtype=float)
    # X_test0 = np.zeros((200, 4 * n_features), dtype=float)
    # X_test1 = np.zeros((200, 4 * n_features), dtype=float)

    tmp1 = np.zeros((256, 4 * n_features), dtype=float)
    tmp2 = np.zeros((256, 4 * n_features), dtype=float)
    tmp3 = np.zeros((256, 4 * n_features), dtype=float)
    tmp4 = np.zeros((257, 4 * n_features), dtype=float)

    prepare_data(path_train0, tmp1)
    prepare_data(pp2, tmp2)
    prepare_data(pp3, tmp3)
    prepare_data(pp4, tmp4)

    X_train = np.concatenate([tmp1, tmp2])
    y_train = np.concatenate([np.zeros(len(tmp1)), np.ones(len(tmp2))])

    X_test = np.concatenate([tmp3, tmp4])
    y_test = np.concatenate([np.zeros(len(tmp3)), np.ones(len(tmp4))])

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=11)

    # prepare_data(path_test0, X_test0)
    # prepare_data(path_test1, X_test1)

    # X_train = np.concatenate([X_train0, X_train1])
    # y_train = np.concatenate([np.zeros(len(X_train0), dtype=int), np.ones(len(X_train1), dtype=int)])
    #
    # X_test = np.concatenate([X_test0, X_test1])
    # y_test = np.concatenate([np.zeros(len(X_test0), dtype=int), np.ones(len(X_test1), dtype=int)])



    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)

    print(classification_report(y_test, y_predicted, target_names=['live', 'spoof']))


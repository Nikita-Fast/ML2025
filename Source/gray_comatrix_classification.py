from skimage.feature import graycomatrix, graycoprops
import numpy as np


if __name__ == '__main__':
    img = np.array([
        [0,0,1,1],
        [0,0,1,1],
        [0,2,2,2],
        [2,2,3,3]
    ])
    P = graycomatrix(img, distances=[1], angles=[0, -np.pi/4, -np.pi/2, -3*np.pi/4], levels=4, symmetric=True)
    m = P[:,:,0,:]
    s = np.sum(m, axis=(0,1))
    # s = np.ones(4)
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

    print(m0)

    print(mean_x(m0), mean_y(m0), var_x(m0), var_y(m0))
    print(f"{'f1_odnorodnost':.<20s}", f1_odnorodnost(m0))
    print(f"{'f2_contrast':.<20s}", f2_contrast(m0))
    print(f"{'f3_corr':.<20s}", f3_corr(m0))
    print(f"{'f4_entropy':.<20s}", f4_entropy(m0))

    print("----------------")
    for p in ['contrast', "homogeneity", 'correlation', 'ASM', 'mean', 'variance', 'entropy']:
        print(f"{p:.<20s}", graycoprops(P, p)[0,0])

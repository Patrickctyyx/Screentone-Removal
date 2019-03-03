import cv2
import numpy as np


class EdgeTangentFlow:

    def __init__(self, size=(300, 300)):
        self.flow_field = np.zeros((size[0], size[1], 3), dtype=np.float32)
        self.refined_etf = np.zeros((size[0], size[1], 3), dtype=np.float32)
        self.gradient_mag = np.zeros((size[0], size[1], 3), dtype=np.float32)

    def initial_etf(self, filename):

        src = cv2.imread(filename, 1)
        size = (src.shape[0], src.shape[1])

        self.__resize_mat(size)

        src_n = cv2.normalize(src, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC1)

        # size of Sobel kernel is 5
        grad_x = cv2.Sobel(src_n, cv2.CV_32FC1, 1, 0, 5)
        grad_y = cv2.Sobel(src_n, cv2.CV_32FC1, 0, 1, 5)

        # compute gradient
        cv2.magnitude(grad_x, grad_y, self.gradient_mag)
        cv2.normalize(self.gradient_mag, self.gradient_mag, 0, 1, cv2.NORM_MINMAX)

        # init flow_field
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                u = grad_x[i][j]
                v = grad_y[i][j]

                self.flow_field[i][j] = cv2.normalize(np.array(
                    [v[0], u[0], 0], dtype=np.float32
                ), None).reshape(self.flow_field[i][j].shape)

        self.rotate_flow(self.flow_field, 90)

    def refine_etf(self, kernel):
        for r in range(self.flow_field.shape[0]):
            for c in range(self.flow_field.shape[1]):
                self.__compute_new_vector(c, r, kernel)

        self.flow_field = self.refined_etf

    def rotate_flow(self, src, theta):
        theta = theta / 180 * np.pi

        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                v = src[i][j]
                rx = v[0] * np.cos(theta) - v[1] * np.sin(theta)
                ry = v[1] * np.cos(theta) + v[0] * np.sin(theta)
                src[i][j] = np.array([rx, ry, 0], dtype=np.float32)

    def __resize_mat(self, size):
        self.flow_field.resize((size[0], size[1], 3))
        self.refined_etf.resize((size[0], size[1], 3))
        self.gradient_mag.resize((size[0], size[1], 3))

    def __compute_new_vector(self, x, y, kernel):
        t_tur_x = self.flow_field[y][x]
        t_new = np.zeros(3, dtype=np.float32)

        for r in range(y - kernel, y + kernel + 1):
            for c in range(x - kernel, x + kernel + 1):
                if r < 0 or r >= self.refined_etf.shape[0] or c < 0 or c >= self.refined_etf.shape[1]:
                    continue

                t_tur_y = self.flow_field[r][c]
                phi = self.__compute_phi(t_tur_x, t_tur_y)
                w_s = self.__compute_ws(np.array((x, y)), np.array((c, r)), kernel)
                w_m = self.__compute_wm(cv2.norm(self.gradient_mag[y][x]), cv2.norm(self.gradient_mag[r][c]))
                w_d = self.__compute_wd(t_tur_x, t_tur_y)

                t_new += phi * t_tur_y * w_s * w_m * w_d

        self.refined_etf[y][x] = cv2.normalize(t_new, None).reshape(self.refined_etf[y][x].shape)

    def __compute_phi(self, x, y):
        if x.dot(y) > 0:
            return 1
        return -1

    def __compute_ws(self, x, y, r):
        if cv2.norm(x - y) < r:
            return 1
        return 0

    def __compute_wm(self, gradmag_x, gradmag_y):
        return (1 + np.tanh(gradmag_x - gradmag_y)) / 2

    def __compute_wd(self, x, y):
        return abs(x.dot(y))


if __name__ == '__main__':
    # todo: test this module
    etf = EdgeTangentFlow()
    etf.initial_etf('../imgs/kiana.jpg')
    etf.refine_etf(5)

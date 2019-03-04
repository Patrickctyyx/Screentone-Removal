import cv2
import numpy as np
from .utils import make_gauss_vector
from .edge_tangent_flow import EdgeTangentFlow


# todo: test this class
# todo: some parts can be replaced by cv2 functions
class CoherentLineDrawing:

    def __init__(self, size=(300, 300)):
        self.SIGMA_RATIO = 1.6
        self.STEPSIZE = 1.0

        # one channel
        self.original_img = np.zeros(size, dtype=np.uint8)
        self.result = np.zeros(size, dtype=np.uint8)
        self.dog = np.zeros(size, np.float32)
        self.fdog = np.zeros(size, np.float32)

        self.etf = EdgeTangentFlow(size)

        self.sigma_m = 3.0
        self.sigma_c = 1.0
        self.rho = 0.997
        self.tau = 0.8

    def read_src(self, filename):
        self.original_img = cv2.imread(filename, 0)
        size = self.original_img.shape

        self.result = np.zeros(size, dtype=np.uint8)
        self.dog = np.zeros(size, np.float32)
        self.fdog = np.zeros(size, np.float32)

        self.etf.initial_etf(filename)

    def gen_cld(self):
        original_img_32fc1 = self.original_img / 255

        self.dog = self.gradient_dog(original_img_32fc1, self.rho, self.sigma_c)
        self.fdog = self.flow_dog(self.dog, self.sigma_m)

        self.result = self.binary_thresholding(self.fdog, self.tau)

    def combine_image(self):
        self.original_img = cv2.bitwise_and(self.original_img, self.result)

    def gradient_dog(self, src, rho, sigma_c):
        dst = np.zeros(src.shape, src.dtype)

        sigma_s = self.SIGMA_RATIO * sigma_c
        gau_c = make_gauss_vector(sigma_c)
        gau_s = make_gauss_vector(sigma_s)

        kernel = len(gau_s) - 1

        for y in range(src.shape[0]):
            for x in range(src.shape[1]):
                gau_c_acc = 0
                gau_s_acc = 0
                gau_c_weight_acc = 0
                gau_s_weight_acc = 0

                tmp = self.etf.flow_field[y][x]
                gradient = [-tmp[0], tmp[1]]

                if gradient[0] == 0 and gradient[1] == 0:
                    continue

                for step in range(-kernel, kernel + 1):
                    row = y + gradient[1] * step
                    col = x + gradient[0] * step

                    if col > src.shape[1] - 1 or col < 0 or row > src.shape[0] - 1 or row < 0:
                        continue

                    value = src[round(row)][round(col)]

                    gau_idx = abs(step)
                    gau_c_weight = 0.0
                    if gau_idx < len(gau_c):
                        gau_c_weight = gau_c[gau_idx]
                    gau_s_weight = gau_s[gau_idx]

                    gau_c_acc += value * gau_c_weight
                    gau_s_acc += value * gau_s_weight
                    gau_c_weight_acc += gau_c_weight
                    gau_s_weight_acc += gau_s_weight

                v_c = gau_c_acc / gau_c_weight_acc
                v_s = gau_s_acc / gau_s_weight_acc
                dst[y][x] = v_c - rho * v_s

        return dst


    def flow_dog(self, src, sigma_m):
        dst = np.zeros(src.shape, src.dtype)

        gau_m = make_gauss_vector(sigma_m)
        kernel_half = len(gau_m) - 1

        for y in range(src.shape[0]):
            for x in range(src.shape[1]):
                gau_m_acc = -gau_m[0] * src[y][x]
                gau_m_weight_acc = -gau_m[0]

                pos = [x, y]

                for step in range(kernel_half):
                    tmp = self.etf.flow_field[round(pos[1])][round(pos[0])]
                    direction = [tmp[1], tmp[0]]

                    if direction[0] == 0 and direction[1] == 0:
                        break

                    if pos[0] > src.shape[1] - 1 or pos[0] < 0 or pos[1] > src.shape[0] - 1 or pos[1] < 0:
                        break

                    value = src[round(pos[1])][round(pos[0])]
                    weight = gau_m[step]

                    gau_m_acc += value * weight
                    gau_m_weight_acc += weight

                    pos[0] += direction[0]
                    pos[1] += direction[1]

                    if round(pos[0]) > src.shape[1] - 1 or round(pos[0])< 0 \
                            or round(pos[1]) > src.shape[0] - 1 or round(pos[1]) < 0:
                        break

                pos = [x, y]

                for step in range(kernel_half):
                    tmp = -self.etf.flow_field[round(pos[1])][round(pos[0])]
                    direction = [tmp[1], tmp[0]]

                    if direction[0] == 0 and direction[1] == 0:
                        break

                    if pos[0] > src.shape[1] - 1 or pos[0] < 0 or pos[1] > src.shape[0] - 1 or pos[1] < 0:
                        break

                    value = src[round(pos[1])][round(pos[0])]
                    weight = gau_m[step]

                    gau_m_acc += value * weight
                    gau_m_weight_acc += weight

                    pos[0] += direction[0]
                    pos[1] += direction[1]

                    if round(pos[0]) > src.shape[1] - 1 or round(pos[0]) < 0 \
                            or round(pos[1]) > src.shape[0] - 1 or round(pos[1]) < 0:
                        break

                if (gau_m_acc / gau_m_weight_acc) > 0:
                    dst[y][x] = 1.0
                else:
                    dst[y][x] = 1 + np.tanh(gau_m_acc / gau_m_weight_acc)

        return cv2.normalize(dst, None, 0, 1, cv2.NORM_MINMAX)

    def binary_thresholding(self, src, tau):
        _, dst = cv2.threshold(src, tau, 255, cv2.THRESH_BINARY)

        return dst

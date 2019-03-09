import cv2
import numpy as np


def get_etf(flow_field):
    """
    Visualize the ETF

    :param flow_field:
    :return: etf in float format
    """

    noise = np.zeros((int(flow_field.shape[1] / 2), int(flow_field.shape[0] / 2)), 'float32')
    dis = np.zeros(flow_field.shape[: 2], 'float32')
    cv2.randu(noise, 0, 1.0)
    noise = cv2.resize(noise, flow_field.shape[: 2], None, 0, 0, cv2.INTER_NEAREST)

    s = 10
    n_rows = noise.shape[0]
    n_cols = noise.shape[1]
    sigma = 2 * (s ** 2)

    for i in range(n_rows):
        for j in range(n_cols):
            w_sum = 0
            x = i
            y = j

            for k in range(s):
                v = cv2.normalize(
                    flow_field[int(x + n_rows) % n_rows][int(y + n_cols) % n_cols], None
                ).reshape(flow_field[0][0].shape)
                if v[0] != 0:
                    x = x + abs(v[0]) / (abs(v[0]) + abs(v[1])) * (abs(v[0]) / v[0])
                if v[1] != 0:
                    y = y + abs(v[1]) / (abs(v[0]) + abs(v[1])) * (abs(v[1]) / v[1])

                r2 = k ** 2
                w = 1 / (np.pi * sigma) * np.exp(-r2 / sigma)
                dis[i][j] += w * noise[int(x + n_rows) % n_rows][int(y + n_cols) % n_cols]
                w_sum += w

            x = i
            y = j

            for k in range(s):
                v = -cv2.normalize(
                    flow_field[int(x + n_rows) % n_rows][int(y + n_cols) % n_cols], None
                ).reshape(flow_field[0][0].shape)
                if v[0] != 0:
                    x = x + abs(v[0]) / (abs(v[0]) + abs(v[1])) * (abs(v[0]) / v[0])
                if v[1] != 0:
                    y = y + abs(v[1]) / (abs(v[0]) + abs(v[1])) * (abs(v[1]) / v[1])

                r2 = k ** 2
                w = 1 / (np.pi * sigma) * np.exp(-r2 / sigma)
                dis[i][j] += w * noise[int(x + n_rows) % n_rows][int(y + n_cols) % n_cols]
                w_sum += w

            dis[i][j] /= w_sum

    return dis


def get_flow_field(flow_field, dis):
    """
    Visual ETF by drawing red arrow line

    :param flow_field:
    :param dis: original image
    :return: arrow line image
    """

    RESOLUTION = 10
    dis = cv2.cvtColor(dis, cv2.COLOR_GRAY2BGR)

    for i in range(0, flow_field.shape[0], RESOLUTION):
        for j in range(0, flow_field.shape[1], RESOLUTION):
            v = flow_field[i][j]
            p1 = (j, i)
            p2 = (int(j + v[1] + 5), int(i + v[0] * 5))
            cv2.arrowedLine(dis, p1, p2, (255, 0, 0), 1, 8, 0, 0.3)

    return dis


def get_anti_alias(src):

    """
    anti alias for the result

    :param src: the result image
    :return: processed image
    """

    BLUR_SIZE = 3

    dst = cv2.normalize(src, None, 60, 255, cv2.NORM_MINMAX)
    dst = cv2.GaussianBlur(dst, (BLUR_SIZE, BLUR_SIZE), 0)

    return dst.astype('uint8')

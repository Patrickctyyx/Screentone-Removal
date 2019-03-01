import cv2
from skimage.measure import label
import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

def get_ncc(img):  # NCC 是连通域数量
    _, ncc = label(img, neighbors=8, return_num=True)
    return ncc

def get_ccc(img, i):
    mask_log_i = get_mask_log(img, i)
    ncc_i = get_ncc(mask_log_i)

    mask_log_i_2 = get_mask_log(img, i - 2)
    ncc_i_2 = get_ncc(mask_log_i_2)

    return 1 - ncc_i / ncc_i_2

def get_stc(img, i):
    mask_log_1 = get_mask_log(img, 1)
    ncc_1 = get_ncc(mask_log_1)

    mask_log_i = get_mask_log(img, i)
    ncc_i = get_ncc(mask_log_i)

    mask_log_i_2 = get_mask_log(img, i - 2)
    ncc_i_2 = get_ncc(mask_log_i_2)

    return ncc_i / ncc_1 * abs(ncc_i_2 - ncc_i)

def get_mask_log(img, i):
    sigma = 0.3 * ((i - 1) * 0.5 - 1) + 0.8
    log = nd.gaussian_laplace(img, sigma)

    log[log < 0] = 0

    log = log.astype('uint8')
    # 返回的第一个值是门槛值
    _, log = cv2.threshold(log, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    return log

def screentone_removal(filename):
    # params
    i = 3
    ccc_max = 0  # ccc is Connected Component Criteria
    i_log = 1

    ALPHA = 1  # threshold of stc
    BETA = 0.8  # relaxation factor

    # read image as grey image
    img = cv2.imread(filename, 0)

    stc_i = get_stc(img, i)

    while stc_i > ALPHA:
        ccc_i = get_ccc(img, i)
        if ccc_i >= BETA * ccc_max:
            i_log = i
            ccc_max = max(ccc_i, ccc_max)
        i += 2

        # update stc_i
        stc_i = get_stc(img, i)

    i_log += 4
    i_base = min(int(i / 2), i_log)

    mask_rm = cv2.bitwise_and(get_mask_log(img, i_log), get_mask_log(img, i_base))
    
    print(i_log)
    
    plt.subplot(131)
    plt.title('original')
    plt.imshow(img, "gray")
    plt.subplot(132)
    plt.title('without base mask')
    plt.imshow(get_mask_log(img, i_log), "gray")
    plt.subplot(133)
    plt.title('with base mask')
    plt.imshow(mask_rm, "gray")
    plt.show()


if __name__ == "__main__":
    # screentone_removal("imgs/15.png")
    screentone_removal("imgs/kanshan.jpg")

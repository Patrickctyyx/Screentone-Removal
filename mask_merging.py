import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label

from remove_mask import screentone_removal
from coherent_line_drawing.coherent_line_drawing import CoherentLineDrawing


def get_final_mask(filename):
    # get mask removal
    _, i_base, mask_rm = screentone_removal(filename)

    # get mask line preserving
    w_dog = i_base - 2
    sigma_c = 0.3 * ((w_dog - 1) * 0.5 - 1) + 0.8
    cld = CoherentLineDrawing(sigma_c=sigma_c)
    cld.read_src(filename)
    cld.gen_cld()
    mask_lp = cld.result

    plt.title('line preserving')
    plt.imshow(mask_lp, "gray")
    plt.show()

    # combine the two masks through connected components
    mask_comb = cv2.bitwise_or(mask_rm, mask_rm, mask=mask_lp)
    m_final = np.zeros(mask_comb.shape, mask_comb.dtype)

    labeled_img, ncc = label(mask_comb, connectivity=2, return_num=True)
    for n in range(1, ncc + 1):
        # traverse all connected component
        idx = np.where(labeled_img == n)
        # if relevant area in mask removal is not all zero
        # then set final values to 1
        arr = mask_rm[idx]
        none_zero_num = np.count_nonzero(arr)
        if none_zero_num > 0:
            m_final[idx] = 255

    plt.title('merged mask')
    plt.imshow(m_final, "gray")
    plt.show()

    return m_final


if __name__ == '__main__':
    filename = 'mangas/manga7.png'
    get_final_mask(filename)

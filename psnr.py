# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 17:58
# @Author  : huangwei
# @File    : psnr.py
# @Software: PyCharm
import cv2
import math
import numpy as np

def psnr(file1, file2):
    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)

    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100

    PSNR = 20 * math.log10(1 / math.sqrt(mse))
    return PSNR


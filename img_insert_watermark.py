# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 13:48
# @Author  : huangwei
# @File    : img_insert_watermark.py
# @Software: PyCharm
from img_function import *

"""
    原图片和水印存储文件夹为：/home/data/picture/origin
    生成的图片文件存储路径为：/home/data/picture/target
"""

input_file_path = "./images/src.jpg"
watermark_path = "./images/gray.png"

img_insert_watermark(input_file_path, watermark_path)



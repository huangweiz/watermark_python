# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 18:52
# @Author  : huangwei
# @File    : img_function.py
# @Software: PyCharm

import copy
import math
import cv2
import numpy as np

# 加入水印的标志，用于判断是否加入了水印
flag_start = np.array(
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
     1,
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
     1, 0, 1, 0, 1])

block_shape = np.array((8, 8))
markstrength = 100


def img_insert_watermark(input_filepath, watermark_filepath, output_filepath):
    """
    只能取 y 层向量进行水印的插入，取 uv 两层的话，变化太小提取不出正常的数据
    传入图片路径，将其转为 yuv444，取 y 层向量进行水印的插入
    :param output_filepath:
    :param input_filepath:
    :param watermark_filepath:
    :return:
    """

    image = cv2.imread(input_filepath)
    img_shape = image.shape

    # 将图片转 yuv
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # 用于将 向量划分成 block
    strides = 4 * np.array([img_shape[1] * block_shape[0], block_shape[1], img_shape[1], 1])
    frame_block_shape = (img_shape[0] // block_shape[0], img_shape[1] // block_shape[1], block_shape[0], block_shape[1])

    # 初始化 block 的 index
    block_index = [(i, j) for i in range(frame_block_shape[0]) for j in range(frame_block_shape[1])]

    # 生成水印 bit 信息和属性 bit 信息，即将要插入的数据转为二进制
    watermark = cv2.imread(watermark_filepath, cv2.IMREAD_GRAYSCALE)

    watermark_size = watermark.shape[0] * watermark.shape[1]
    insert_size = watermark_size + len(flag_start) + 100

    # 如果水印过大，则将其进行缩放到合适的大小
    if insert_size > len(block_index):
        print("最多可嵌入{}kb信息，水印大小{}kb，因此将水印进行缩放".format(len(block_index) / 1024, watermark_size / 1024))
        max_watermark_size = len(block_index) - len(flag_start) - 100
        scale = (max_watermark_size / watermark_size) ** 0.5
        shape0 = int(watermark.shape[0] * scale)
        shape1 = int(watermark.shape[1] * scale)
        watermark_size = shape0 * shape1

        watermark = cv2.resize(watermark, (shape1, shape0))

        print("新的水印大小为{0}*{1}。".format(watermark.shape[0], watermark.shape[1]))

    watermark_bit = watermark.flatten() > 128

    height = format(watermark.shape[0], "b").zfill(10)  # 将 height, width 转为长度为 10 的二进制数据
    width = format(watermark.shape[1], "b").zfill(10)
    watermark_attr_bit = (height + width) * 5

    # 取出yuv 中的 y 向量用于插入水印
    y_floor = img_yuv[:, :, 0]

    frame_block = np.lib.stride_tricks.as_strided(y_floor.astype(np.float32), frame_block_shape, strides)
    embed_frame = copy.deepcopy(y_floor)

    # 对前 80 个块进行标志位嵌入
    for k in range(len(flag_start)):
        # 对 frame_block[k] 进行离散余弦变换获得DCT系数矩阵
        # 运用余数定理实现水印嵌入
        frame_block_dct = cv2.dct(frame_block[block_index[k]])

        if flag_start[k] == 0:
            frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 10
        else:
            frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 30

        # 取整
        frame_block[block_index[k]] = np.rint(np.clip(cv2.idct(frame_block_dct), a_min=0, a_max=255))

    # 嵌入水印属性信息 100 bits
    for k in range(len(watermark_attr_bit)):
        frame_block_dct = cv2.dct(frame_block[block_index[k + len(flag_start)]])

        if int(watermark_attr_bit[k]):
            frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 30
        else:
            frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 10

        frame_block[block_index[k + len(flag_start)]] = np.rint(np.clip(cv2.idct(frame_block_dct), a_min=0, a_max=255))

    # 嵌入水印信息
    for k in range(watermark_size):
        frame_block_dct = cv2.dct(frame_block[block_index[k + len(flag_start) + len(watermark_attr_bit)]])

        if watermark_bit[k] == 0:
            frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 10
        else:
            frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 30

        frame_block[block_index[k + len(flag_start) + len(watermark_attr_bit)]] = np.rint(
            np.clip(cv2.idct(frame_block_dct), a_min=0, a_max=255))

    # 四维转为二维，还少了整除剩下的部分。
    part_frame = np.concatenate(np.concatenate(frame_block, 1), 1)

    # 将插入水印的部分放回原处补齐整除剩下的部分
    embed_frame[:part_frame.shape[0], :part_frame.shape[1]] = part_frame

    img_yuv[:, :, 0] = embed_frame
    embed_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(output_filepath, embed_img)
    print("watermark_python insert success and save in %s" % output_filepath)


def img_extract_watermark(input_filepath, output_filepath):
    """
    传入图片路径，将其转为 yuv444，取 v 层向量从其中提取出水印数据
    :param output_filepath:
    :param input_filepath:
    :return:
    """

    image = cv2.imread(input_filepath)
    img_shape = image.shape

    # 将图片转 yuv
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # 取出yuv 中的 y 向量用于提取水印
    v_floor = img_yuv[:, :, 0]

    # 用于将 向量划分成 block
    strides = 4 * np.array([img_shape[1] * block_shape[0], block_shape[1], img_shape[1], 1])
    frame_block_shape = (img_shape[0] // block_shape[0], img_shape[1] // block_shape[1], block_shape[0], block_shape[1])

    # 初始化 block 的 index
    block_index = [(i, j) for i in range(frame_block_shape[0]) for j in range(frame_block_shape[1])]

    frame_block = np.lib.stride_tricks.as_strided(v_floor.astype(np.float32), frame_block_shape, strides)

    # 提取出前80 个 block 中插入的数据
    temp_list1 = []
    for k in range(80):
        frame_block_dct = cv2.dct(frame_block[block_index[k]])
        temp_list1.append(frame_block_dct[-1][-1] % markstrength)
        frame_block[block_index[k]] = cv2.idct(frame_block_dct)

    list_bit1 = np.array(temp_list1) > 20

    # 将80 转成 16 去除误差
    temp_bit = []
    for k in range(16):
        temp = int(list_bit1[k]) + int(list_bit1[k + 16]) + int(list_bit1[k + 16 * 2]) + int(
            list_bit1[k + 16 * 3]) + int(list_bit1[k + 16 * 4])
        temp_bit.append(temp)

    flag_bit = np.array(temp_bit) > 2

    count_start = 0
    for k in range(16):
        if flag_bit[k] == flag_start[k]:
            count_start += 1

    # print(count_start)

    if count_start > 13:
        # 则说明该帧存在水印，提取出来
        # 先提取 81 到 180 个 block 的水印属性信息
        temp_list2 = []
        for k in range(80, 180):
            frame_block_dct = cv2.dct(frame_block[block_index[k]])
            temp_list2.append(frame_block_dct[-1][-1] % markstrength)
            frame_block[block_index[k]] = cv2.idct(frame_block_dct)

        list_bit2 = np.array(temp_list2) > 15

        height_arr = []
        width_arr = []
        for k in range(5):
            h_tmp = list_bit2[0 + k * 20] * 512 + list_bit2[1 + k * 20] * 256 + list_bit2[2 + k * 20] * 128 + list_bit2[
                3 + k * 20] * 64 + list_bit2[4 + k * 20] * 32 + list_bit2[5 + k * 20] * 16 + list_bit2[
                        6 + k * 20] * 8 + \
                    list_bit2[7 + k * 20] * 4 + list_bit2[8 + k * 20] * 2 + list_bit2[9 + k * 20]
            height_arr.append(h_tmp)

            w_tmp = list_bit2[10 + k * 20] * 512 + list_bit2[11 + k * 20] * 256 + list_bit2[12 + k * 20] * 128 + \
                    list_bit2[
                        13 + k * 20] * 64 + list_bit2[14 + k * 20] * 32 + list_bit2[15 + k * 20] * 16 + list_bit2[
                        16 + k * 20] * 8 + list_bit2[17 + k * 20] * 4 + list_bit2[18 + k * 20] * 2 + list_bit2[
                        19 + k * 20]
            width_arr.append(w_tmp)

        height = np.argmax(np.bincount(height_arr))
        width = np.argmax(np.bincount(width_arr))

        # 提取水印信息
        watermark_size = width * height

        temp_list3 = []
        for k in range(180, 180 + watermark_size):
            frame_block_dct = cv2.dct(frame_block[block_index[k]])
            temp_list3.append(frame_block_dct[-1][-1] % markstrength)
            frame_block[block_index[k]] = cv2.idct(frame_block_dct)

        list_bit3 = np.array(temp_list3) > 15

        watermark_data = [[0 for x in range(width)] for y in range(height)]

        index = 0
        for row in range(height):
            for col in range(width):
                watermark_data[row][col] = int(list_bit3[index])
                index += 1

        watermark = 255 * np.array(watermark_data)

        cv2.imwrite(output_filepath, watermark)
        print("watermark_python extract success and save in %s" % output_filepath)

# _*_ coding:utf-8 _*_
# 开发团队：huang
# 开发时间：2020/11/19 22:50
# 文件名称：function.py
# 开发工具：PyCharm

import copy
import math
import os
import shutil
import cv2
import ffmpeg
import numpy as np

# 需要先安装ffmpeg
# 加入水印的标志，这样在解水印过程中只要看前是否有这个标志而不需要对整帧数据进行扫描
flag_start = np.array(
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
     1,
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
     1, 0, 1, 0, 1])

block_shape = np.array((4, 4))
markstrength = 100


def delete_file(filepath):
    """
    用于删除中间产生的临时文件
    :param filepath:
    :return:
    """
    if os.path.exists(filepath):
        os.remove(filepath)
    else:
        print("file is not exists.")


def create_dir(dir_path):
    """创建文件夹"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print("%s is exists" % dir_path)


def delete_dir(dir_path):
    """删除文件夹"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    else:
        print("%s is not exists")


def audio_video_exists(input_filepath):
    """
    判断文件中是否存在视频流和音频流
    :param input_filepath:
    :return: 存在音频流则返回 True ，否则返回 False
    """
    probe = ffmpeg.probe(input_filepath)

    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

    if video_stream is None:
        raise AssertionError("please check your file, this may not be a video!")

    audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)

    if audio_stream is None:
        is_audio_exists = False
    else:
        is_audio_exists = True

    return is_audio_exists


def divorce_audio_video(input_filepath):
    """
    将音频和视频进行分离存储到临时文件夹中
    :return:存在音频流则返回 True ，否则返回 False
    """
    is_audio_exists = audio_video_exists(input_filepath)

    # 创建临时文件夹用来存储插入水印过程中生成的临时文件
    tmp_file_directory = "tmp_file_directory"
    create_dir(tmp_file_directory)

    if is_audio_exists:
        # 音频存储路径
        audio_path = "./%s/audio.m4a" % tmp_file_directory
        os.system("ffmpeg -y -i %s -acodec copy -vn %s" % (input_filepath, audio_path))

    # 音频存储路径
    video_path = "./%s/video.mp4" % tmp_file_directory
    os.system("ffmpeg -y -i %s -vcodec copy -an %s" % (input_filepath, video_path))

    return is_audio_exists


def get_video_param(video_path):
    """
    返回视频流中的元素
    :param video_path:
    :return: {duration, width, height, codec_name, pix_fmt, fps}
    """
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])

    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

    width = video_stream['width']
    height = video_stream['height']
    codec_name = video_stream['codec_name']
    pix_fmt = video_stream['pix_fmt']
    frame_rate = video_stream['avg_frame_rate']
    fps = round(eval(frame_rate), 2)

    return {"duration": duration, "width": width, "height": height, "codec_name": codec_name, "pix_fmt": pix_fmt,
            "fps": fps}


def cut_video(video_path, duration):
    """
    将视频进行分段
    :return:返回分的总段数
    """
    # 判断是否存在临时文件夹，不存在则创建
    tmp_file_directory = "tmp_file_directory"
    create_dir(tmp_file_directory)

    # 将视频转换为帧内编码使得视频分割更精确
    intra_path = "./%s/intra.mp4" % tmp_file_directory
    os.system("ffmpeg -y -i %s -strict -2 -qscale 0 -intra %s" % (video_path, intra_path))

    # 将视频均匀分段
    # 每一小段的视频为10s
    # 总的段数为 gap_num+1
    gap = 10
    gap_num = int(duration / gap)

    for i in range(gap_num):
        start_time = i * gap
        out_path = "./%s/gap%d.mp4" % (tmp_file_directory, i)
        os.system("ffmpeg -y -ss {0} -t {1} -i {2} -vcodec copy -an {3}".format(start_time, gap, intra_path, out_path))

    # 不够10s的部分
    start_time = gap_num * gap
    out_path = "./%s/gap%d.mp4" % (tmp_file_directory, gap_num)
    os.system(
        "ffmpeg -y -ss {0} -t {1} -i {2} -vcodec copy -an {3}".format(start_time, duration - start_time, intra_path,
                                                                      out_path))

    # 可以删除 intra.mp4节约空间
    delete_file(intra_path)

    return gap_num + 1


def video2yuv(video_path, yuv_path, codec_name, pix_fmt):
    """
    将视频流转换为 yuv 格式
    :param video_path: 输入视频路径
    :param yuv_path: 输出yuv文件路径
    :param codec_name: 输入文件编码
    :param pix_fmt: 输出文件编码
    :return:
    """

    os.system("ffmpeg -y -vcodec {0} -i {1} -pix_fmt {2} {3}".format(codec_name, video_path, pix_fmt, yuv_path))


def add_watermark(input_filepath, output_filepath, watermark_filepath, img_shape):
    """

    :param input_filepath:
    :param output_filepath:
    :param watermark_filepath:
    :param img_shape:
    :return:
    """
    fp = open(input_filepath, "rb")  # 读取yuv文件
    input_data = fp.read()

    file_length = len(input_data)  # 整个文件字符长度
    y_length = img_shape[0] * img_shape[1]  # 单帧的 y 向量大小
    yuv_length = int(y_length * 3 / 2)  # 单帧的 yuv 向量的大小
    frames_num = int(file_length / yuv_length)  # 总帧数

    frame = bytearray(yuv_length)  # 用于存储单帧的 yuv 数据

    # 用于存储单帧的 y 向量数据
    # 创建一个 img_shape[0] 行 img_shape[1] 列的二维数组
    data = [[0 for x in range(img_shape[1])] for y in range(img_shape[0])]

    # 用于将 y 向量划分成 8*8 的 block
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

    cv2.imwrite("./output/watermark_resize.jpg", watermark)
    print(watermark.shape)

    watermark_bit = watermark.flatten() > 128

    height = format(watermark.shape[0], "b").zfill(10)  # 将 height, width 转为长度为 10 的二进制数据
    width = format(watermark.shape[1], "b").zfill(10)
    watermark_attr_bit = (height + width) * 5

    # 依次取出每一帧，每 50 帧的前 5 帧插入水印
    frame_gap = 50
    for i in range(frames_num):
        for j in range(yuv_length):
            frame[j] = input_data[i * yuv_length + j]

        # 判断该帧是否插入数据
        if i % frame_gap < 5:
            # 取出该帧的 y 向量数据
            y_index = 0
            for row in range(img_shape[0]):
                for col in range(img_shape[1]):
                    data[row][col] = frame[y_index]
                    y_index += 1

            frame_y = np.array(data)

            # 将 y 向量划分为 8 * 8 的块
            frame_block = np.lib.stride_tricks.as_strided(frame_y.astype(np.float32), frame_block_shape, strides)
            embed_frame = copy.deepcopy(frame_y)

            print("插入数据帧: %d." % i)

            # 对前 80 个块进行标志位嵌入
            for k in range(len(flag_start)):
                # 对 frame_block[k] 进行离散余弦变换获得DCT系数矩阵
                # 运用余数定理实现水印嵌入
                frame_block_dct = cv2.dct(frame_block[block_index[k]])

                if flag_start[k] == 0:
                    frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 10
                else:
                    frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 30

                frame_block[block_index[k]] = cv2.idct(frame_block_dct)

            # 嵌入水印属性信息 100 bits
            for k in range(len(watermark_attr_bit)):

                frame_block_dct = cv2.dct(frame_block[block_index[k + len(flag_start)]])

                if int(watermark_attr_bit[k]):
                    frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 30
                else:
                    frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 10

                frame_block[block_index[k + len(flag_start)]] = cv2.idct(frame_block_dct)

            # 嵌入水印信息
            for k in range(watermark_size):

                frame_block_dct = cv2.dct(frame_block[block_index[k + len(flag_start) + len(watermark_attr_bit)]])

                if watermark_bit[k] == 0:
                    frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 10
                else:
                    frame_block_dct[-1][-1] = math.floor(frame_block_dct[-1][-1] / markstrength) * markstrength + 30

                frame_block[block_index[k + len(flag_start) + len(watermark_attr_bit)]] = cv2.idct(frame_block_dct)

            # 四维转为二维，还少了整除剩下的部分。
            part_frame = np.concatenate(np.concatenate(frame_block, 1), 1)

            # 将插入水印的部分放回原处补齐整除剩下的部分
            embed_frame[:part_frame.shape[0], :part_frame.shape[1]] = part_frame
            embed_frame = np.clip(embed_frame, a_min=0, a_max=255)

            cv2.imwrite("./images/embed_frame%d.jpg"%i, embed_frame)

            y_index = 0
            for row in range(img_shape[0]):
                for col in range(img_shape[1]):
                    frame[y_index] = int(embed_frame[row][col])
                    y_index += 1

        with open(output_filepath, "ab+") as wp:
            wp.write(frame)


def yuv2video(input_filepath, output_filepath, shape, pix_fmt, r):
    """
    yuv 转 mp4
    :param input_filepath: 输入的 yuv 文件路径
    :param output_filepath: 输出的 mp4 文件路径
    :param shape: 视频分辨率
    :param pix_fmt: 
    :param r: 帧率
    :return: 
    """

    os.system(
        "ffmpeg -y -s:v {0}x{1} -pix_fmt {2} -r {3} -i {4} {5}".format(shape[0], shape[1], pix_fmt, r, input_filepath,
                                                                       output_filepath))


def concat_gap_video(list_path, temp_concat_path):
    """短视频连接"""
    os.system("ffmpeg -y -f concat -safe 0 -i {0} -c copy {1}".format(list_path, temp_concat_path))


def video_add_audio(input_filepath, output_filepath):
    """音频视频连接"""
    audio_path = "./tmp_file_directory/audio.m4a"
    os.system("ffmpeg -y -i {0} -i {1} -c copy {2}".format(input_filepath, audio_path, output_filepath))


def video_copy(input_filepath, output_filepath):
    """视频复制"""
    os.system("ffmpeg -y -i {0} -c copy {1}".format(input_filepath, output_filepath))


def get_video(input_filepath, output_filepath, start_time=0, gap_time=5):
    """
    从视频中截取一段用来检测是否存在水印，默认0到5秒
    :param input_filepath: 输入文件
    :param start_time: 截取视频开始时间
    :param gap_time: 截取时长
    :param output_filepath: 输出文件
    :return:
    """

    # 判断开始时间和时长是否有效
    probe = ffmpeg.probe(input_filepath)
    duration = float(probe['format']['duration'])

    assert start_time >= 0, IndexError(
        "开始时间有误，不能小于0")

    assert gap_time <= 30 and gap_time <= duration, IndexError(
        "截取视频时长过长，缩短截取的时长")

    assert gap_time + start_time <= duration, IndexError(
        "截取部分超过视频范围")

    """ 1. 去除音频 """
    temp_video = "./temp_video.mp4"
    os.system("ffmpeg -y -i %s -vcodec copy -an %s" % (input_filepath, temp_video))

    """ 2. 截取视频 """
    os.system("ffmpeg -y -ss {0} -t {1} -i {2} -vcodec copy -an {3}".format(start_time, gap_time, temp_video,
                                                                            output_filepath))


def find_flag(input_filepath, img_shape):
    fp = open(input_filepath, "rb")  # 读取yuv文件
    input_data = fp.read()

    file_length = len(input_data)  # 整个文件字符长度
    y_length = img_shape[0] * img_shape[1]  # 单帧的 y 向量大小
    yuv_length = int(y_length * 3 / 2)  # 单帧的 yuv 向量的大小
    frames_num = int(file_length / yuv_length)  # 总帧数

    frame = bytearray(yuv_length)  # 用于存储单帧的 yuv 数据

    # 用于存储单帧的 y 向量数据
    # 创建一个 img_shape[0] 行 img_shape[1] 列的二维数组
    data = [[0 for x in range(img_shape[1])] for y in range(img_shape[0])]

    # 用于将 y 向量划分成 8*8 的 block
    strides = 4 * np.array([img_shape[1] * block_shape[0], block_shape[1], img_shape[1], 1])
    frame_block_shape = (img_shape[0] // block_shape[0], img_shape[1] // block_shape[1], block_shape[0], block_shape[1])

    # 初始化 block 的 index
    block_index = [(i, j) for i in range(frame_block_shape[0]) for j in range(frame_block_shape[1])]

    # 读取每一帧的数据
    for i in range(frames_num):
        for j in range(yuv_length):
            frame[j] = input_data[i * yuv_length + j]

        # 取出该帧的 y 向量数据
        y_index = 0
        for row in range(img_shape[0]):
            for col in range(img_shape[1]):
                data[row][col] = frame[y_index]
                y_index += 1

        frame_y = np.array(data)

        # 将 y 向量划分为 8 * 8 的块
        frame_block = np.lib.stride_tricks.as_strided(frame_y.astype(np.float32), frame_block_shape, strides)

        # 提取出前80 个 block 中插入的数据
        temp_list = []
        for k in range(80):
            frame_block_dct = cv2.dct(frame_block[block_index[k]])
            temp_list.append(frame_block_dct[-1][-1] % markstrength)
            frame_block[block_index[k]] = cv2.idct(frame_block_dct)

        list_bit = np.array(temp_list) > 15
        # print(list_bit)

        # 将80 转成 16 去除误差
        temp_bit = []
        for k in range(16):
            temp = int(list_bit[k]) + int(list_bit[k + 16]) + int(list_bit[k + 16 * 2]) + int(
                list_bit[k + 16 * 3]) + int(list_bit[k + 16 * 4])
            temp_bit.append(temp)

        flag_bit = np.array(temp_bit) > 2

        count_start = 0
        for k in range(16):
            if flag_bit[k] == flag_start[k]:
                count_start += 1

        if count_start > 13:
            # 则说明该帧存在水印，提取出来
            # 先提取 81 到 180 个 block 的水印属性信息
            temp_list = []
            for k in range(80, 180):
                frame_block_dct = cv2.dct(frame_block[block_index[k]])
                temp_list.append(frame_block_dct[-1][-1] % markstrength)
                frame_block[block_index[k]] = cv2.idct(frame_block_dct)

            list_bit = np.array(temp_list) > 15

            height_arr = []
            width_arr = []
            for k in range(5):
                h_tmp = list_bit[0 + k * 20] * 512 + list_bit[1 + k * 20] * 256 + list_bit[2 + k * 20] * 128 + list_bit[
                    3 + k * 20] * 64 + list_bit[4 + k * 20] * 32 + list_bit[5 + k * 20] * 16 + list_bit[
                            6 + k * 20] * 8 + \
                        list_bit[7 + k * 20] * 4 + list_bit[8 + k * 20] * 2 + list_bit[9 + k * 20]
                height_arr.append(h_tmp)

                w_tmp = list_bit[10 + k * 20] * 512 + list_bit[11 + k * 20] * 256 + list_bit[12 + k * 20] * 128 + \
                        list_bit[
                            13 + k * 20] * 64 + list_bit[14 + k * 20] * 32 + list_bit[15 + k * 20] * 16 + list_bit[
                            16 + k * 20] * 8 + list_bit[17 + k * 20] * 4 + list_bit[18 + k * 20] * 2 + list_bit[
                            19 + k * 20]
                width_arr.append(w_tmp)

            height = np.argmax(np.bincount(height_arr))
            width = np.argmax(np.bincount(width_arr))
            # print("height", height)
            # print("width:", width)

            # 提取水印信息
            watermark_size = width * height

            temp_list = []
            for k in range(180, 180 + watermark_size):
                frame_block_dct = cv2.dct(frame_block[block_index[k]])
                temp_list.append(frame_block_dct[-1][-1] % markstrength)
                frame_block[block_index[k]] = cv2.idct(frame_block_dct)

            list_bit = np.array(temp_list) > 15

            watermark_data = [[0 for x in range(width)] for y in range(height)]

            index = 0
            for row in range(height):
                for col in range(width):
                    watermark_data[row][col] = int(list_bit[index])
                    index += 1

            watermark = 255 * np.array(watermark_data)
            print(i)

            cv2.imwrite("./output/watermark_python%d.jpg" % i, watermark)


def insert_watermark(input_filepath, watermark_filepath):
    """
    1. 先判断是否存在音频流和视频流
    :param input_filepath:
    :param watermark_filepath:
    :return:
    """

    # 分离视频流和音频流存储到临时文件夹中，返回的参数用于后面判断是否有音频需要合并
    is_audio_exists = divorce_audio_video(input_filepath)

    tmp_file_directory = "./tmp_file_directory"

    # 取出视频流的多个参数不用每次都取提高速度
    # {duration, width, height, codec_name, pix_fmt, fps}
    video_param = get_video_param("%s/video.mp4" % tmp_file_directory)

    # 将视频切分成多段，传入的视频只有视频流无音频流
    gap_num = cut_video("%s/video.mp4" % tmp_file_directory, video_param['duration'])

    list_path = "%s/list.txt" % tmp_file_directory
    fp = open(list_path, "a+")

    # 将每一段 短视频当成一个单体进行操作
    # video2yuv, insert, yuv2video
    # 每一步删除使用完的文件防止文件堆积过多
    for i in range(gap_num):
        gap_video_path = "{0}/gap{1}.mp4".format(tmp_file_directory, i)
        gap_yuv_path = "{0}/gap{1}.yuv".format(tmp_file_directory, i)
        gap_yuv_embed_path = "{0}/gap{1}_embed.yuv".format(tmp_file_directory, i)
        gap_video_embed_path = "{0}/gap{1}_embed.mp4".format(tmp_file_directory, i)

        video2yuv(gap_video_path, gap_yuv_path, video_param['codec_name'], video_param['pix_fmt'])

        delete_file(gap_video_path)

        add_watermark(gap_yuv_path, gap_yuv_embed_path, watermark_filepath,
                      (video_param['height'], video_param['width']))

        delete_file(gap_yuv_path)

        yuv2video(gap_yuv_embed_path, gap_video_embed_path, (video_param['width'], video_param['height']),
                  video_param['pix_fmt'], video_param['fps'])

        delete_file(gap_yuv_embed_path)

        # 将生成的 mp4文件目录写入 list.txt 中用于连接
        fp.write("file ./gap{0}_embed.mp4\n".format(i))

    fp.close()

    # 视频合并
    gap_concat_path = "%s/gap_concat.mp4" % tmp_file_directory
    concat_gap_video(list_path, gap_concat_path)

    # 音频视频合并
    # 创建用来存储生成文件的目录
    output_directory = "./output"
    create_dir(output_directory)

    output_video_path = "%s/output_video.mp4" % output_directory

    if is_audio_exists is True:
        video_add_audio(gap_concat_path, output_video_path)
    else:
        # 将 gap_concat.mp4复制到output
        video_copy(gap_concat_path, output_video_path)

    # 最后生成output file 文件夹 和 output_video.mp4，将tmp文件夹删除
    # delete_dir(tmp_file_directory)

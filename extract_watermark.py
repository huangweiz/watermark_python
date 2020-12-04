# _*_ coding:utf-8 _*_
# 开发团队：huang
# 开发时间：2020/11/22 16:49
# 文件名称：main4.py
# 开发工具：PyCharm
from function import *


def extract_watermark(input_filepath):
    video_param = get_video_param(input_filepath)

    print(video_param)

    yuv_path = "./output/gap_yuv_path.yuv"
    video2yuv(input_filepath, yuv_path, video_param['codec_name'], video_param['pix_fmt'])

    find_flag(yuv_path, (video_param['height'], video_param['width']))


# 传入要检测的视频文件路径
input_video_path = "./output/output_video.mp4"
gap_video_path = "./output/gap_video.mp4"

# 截取一段视频
get_video(input_video_path, gap_video_path, 0, 10)
extract_watermark(gap_video_path)

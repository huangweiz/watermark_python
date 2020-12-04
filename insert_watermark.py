# _*_ coding:utf-8 _*_
# 开发团队：huang
# 开发时间：2020/11/21 16:15
# 文件名称：main3.py
# 开发工具：PyCharm
from function import *

"""只需要输入 原视频和水印路径，生成的文件存储在 ./output/output_video.mp4"""
input_file_path = "./images/mv2.mp4"
watermark_path = "./images/gray.png"
insert_watermark(input_file_path, watermark_path)



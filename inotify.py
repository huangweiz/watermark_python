from img_function import *
from video_function import *
from psnr import *
import pyinotify
import filetype
import sys
import os

WATCH_PATH = '/home/huang/data'  # 监控目录
# 水印图片固定不变
watermark_path = "/home/huang/data/watermark/watermark.png"

"""
    音视频对应表
    • jpg  –  image/jpeg 
    • png  –  image/png 
    • gif  –  image/gif 
    • webp  –  image/webp 
    • cr2  –  image/x-canon-cr2 
    • tif  –  image/tiff 
    • bmp  –  image/bmp 
    • jxr  –  image/vnd.ms-photo 
    • psd  –  image/vnd.adobe.photoshop 
    • ico  –  image/x-icon

    • mp4  –  video/mp4 
    • m4v  –  video/x-m4v 
    • mkv  –  video/x-matroska 
    • webm  –  video/webm 
    • mov  –  video/quicktime 
    • avi  –  video/x-msvideo 
    • wmv  –  video/x-ms-wmv 
    • mpg  –  video/mpeg 
    • flv  –  video/x-flv
"""

image_type = ['image/jpeg', 'image/png', ]
video_type = ['video/mp4', 'video/x-msvideo', 'video/x-flv', 'video/quicktime']

if not WATCH_PATH:
    print("The WATCH_PATH MUST BE INPUT!")
    sys.exit()
else:
    if os.path.exists(WATCH_PATH):
        print('Found watch path: path=%s.' % WATCH_PATH)
    else:
        print('The watch path NOT exists, watching stop now: path=%s.' % WATCH_PATH)
        sys.exit()


# 事件回调函数
class OnFileHandler(pyinotify.ProcessEvent):
    # 判断文件是否完全写入，完全写入后执行操作
    def process_IN_CLOSE_WRITE(self, event):
        file_path = os.path.join(event.path, event.name)
        print('文件完成写入', file_path)

        # 判断文件写入路径是否为 图片 的传入目录
        if event.path == '/home/huang/data/picture/origin':

            kind = filetype.guess(event.pathname)

            # 判断传入的文件符合 图片 类型
            if kind is not None and kind.mime in image_type:
                input_filepath = event.pathname  # 传入文件路径

                # 拼接输出文件目录
                (filename, extension) = os.path.splitext(event.name)  # 分割文件名和后缀
                output_filepath = "/home/huang/data/picture/target/{0}_mark{1}".format(filename, extension)

                # 插入数字水印
                img_insert_watermark(input_filepath, watermark_path, output_filepath)

                # 查看前后的 psnr 值
                psnr_value = psnr(input_filepath, output_filepath)
                print("psnr value is :", psnr_value)

                # 解出加入水印图片中的水印
                # output_watermark_path = "/home/huang/data/picture/target/extract.png"
                # img_extract_watermark(output_filepath, output_watermark_path)

        elif event.path == '/home/huang/data/video/origin':

            kind = filetype.guess(event.pathname)

            # 判断传入的文件符合 视频 类型
            if kind is not None and kind.mime in video_type:
                input_filepath = event.pathname

                # 分割文件名和后缀
                (filename, extension) = os.path.splitext(event.name)
                output_filepath = "/home/huang/data/video/target/{0}_mark{1}".format(filename, extension)

                video_add_visual_watermark(input_filepath, watermark_path, output_filepath)

                """插入数字水印"""
                # video_insert_watermark(compress_output_filepath, watermark_path, output_filepath)

                # output_watermark_path = "/home/huang/data/picture/target/extract.png"
                # img_extract_watermark(output_filepath, output_watermark_path)

    # 重写文件删除函数
    def process_IN_DELETE(self, event):
        print("文件删除: %s " % os.path.join(event.path, event.name))

    # 重写文件创建函数
    def process_IN_CREATE(self, event):
        print("文件创建: %s " % os.path.join(event.path, event.name))


def auto_compile(path='.'):
    wm = pyinotify.WatchManager()

    # 监控内容，监听文件创建，删除和被完成写入
    mask = pyinotify.IN_CREATE | pyinotify.IN_DELETE | pyinotify.IN_CLOSE_WRITE
    notifier = pyinotify.Notifier(wm, OnFileHandler())  # 回调函数
    wm.add_watch(path, mask, rec=True, auto_add=True)
    print('Start monitoring %s' % path)
    while True:
        try:
            notifier.process_events()
            if notifier.check_events():
                notifier.read_events()
        except KeyboardInterrupt:
            notifier.stop()
            break


if __name__ == "__main__":
    auto_compile(WATCH_PATH)
    print('monitor close')

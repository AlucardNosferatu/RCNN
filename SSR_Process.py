import os
from PIL import Image


# region 介绍
# 这个文件是公司项目相关数据的工具类
# endregion


# 把机器人采集到的大图切成小图
# 不然要识别的目标全图缩放以后连一个像素都不到
def cut_image(img):
    width, height = img.size
    item_width = int(width / 4)
    item_height = int(height / 4)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, 4):  # 两重循环，生成9张图片基于原图的位置
        for j in range(0, 4):
            box = (j * item_width, i * item_height, (j + 1) * item_width, (i + 1) * item_height)
            box_list.append(box)
    img_list = [img.crop(box) for box in box_list]
    return img_list


# 把切好的图存起来
def save_images(img_list, src_name, save_path):
    src_name = src_name.split('.')[0]
    index = 1
    for img in img_list:
        img.save(save_path + src_name.split('.')[0] + "_" + str(index) + '.png', 'PNG')
        index += 1


# 完整处理流程
# 未完待续
def full_process():
    raw_path = "ProcessedData\\SSR_RAW\\"
    split_path = "ProcessedData\\SSR_SPLIT\\"
    for e, i in enumerate(os.listdir(raw_path)):
        image = Image.open(os.path.join(raw_path, i))
        image_list = cut_image(image)
        save_images(image_list, i, split_path)


full_process()

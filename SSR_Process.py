import os
import cv2
import tensorflow as tf
from PIL import Image
from RCNN import test_model_od
from forVOC2007 import data_generator, transfer_model_build, transfer_model_train

# region 介绍
# 这个文件是公司项目相关数据的工具类
# endregion


img_path = "ProcessedData\\SSR_SPLIT"
img_path2 = "ProcessedData\\SSR_RAW"
annotation = "ProcessedData\\SSR_ANNOTATION"
pkl_path = "ProcessedData\\SSR_PKL"
labels_dict = {
    "covered": 1,
    "naked": 2,
    "used": 3
}


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


def DG4SSR():
    data_generator(
        postfix='.png',
        ld=labels_dict,
        annotation_path=annotation,
        pkl_path=pkl_path,
        img_path=img_path
    )


def BM4SSR():
    transfer_model_build(3, model_path="TrainedModels\\RCNN-SSR.h5")


def TM4SSR():
    loader_dict = [12, pkl_path, False, True, 3]
    transfer_model_train(loaderDict=loader_dict, model_path="TrainedModels\\RCNN-SSR.h5")


def TO4SSR():
    test_model_od(model_path="TrainedModels\\RCNN-SSR.h5", start_with_str="DustCap", img_path=img_path2)


TO4SSR()

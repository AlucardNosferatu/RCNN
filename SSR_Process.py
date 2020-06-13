import os
import sys

import cv2
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import xml.etree.ElementTree as ElTr
from FastRCNN import build_model, data_generator, get_slash, train, batch_test, CheckBatch

img_path = "ProcessedData\\SSR_SPLIT"
img_path2 = "ProcessedData\\SSR_RAW"
img_path3 = "ProcessedData\\SSR_OBJ_ONLY"
annotation = "ProcessedData\\SSR_ANNOTATION"
pkl_path = "ProcessedData\\SSR_PKL"
labels_dict = {
    "covered": 1,
    "naked": 2,
    "used": 3
}
slash, linux = get_slash()


def get_objects(file):
    tree = ElTr.parse(file)
    root = tree.getroot()
    objects = []
    for obj in root.iter('object'):
        objects.append(obj)
    return objects


def process_annotation_file(i):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    objects = get_objects(os.path.join(annotation, i))
    filename = i.split(".")[0] + ".png"
    print(filename)
    image_out = cv2.imread(os.path.join(img_path, filename))
    gt_values = []
    gt_labels = []
    for obj in objects:
        name = obj.find('name').text
        gt_labels.append(labels_dict[name])
        xml_box = obj.find('bndbox')
        x1 = (float(xml_box.find('xmin').text) - 1)
        y1 = (float(xml_box.find('ymin').text) - 1)
        x2 = (float(xml_box.find('xmax').text) - 1)
        y2 = (float(xml_box.find('ymax').text) - 1)
        gt_values.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
    ss.setBaseImage(image_out)
    ss.switchToSelectiveSearchFast()
    ss_results = ss.process()
    return ss_results, image_out, {"values": gt_values, "labels": gt_labels}


def data_loader(trainFast=True, shuffle=True):
    try:
        assert trainFast
    except AssertionError as e:
        print()
        print()
        print("===========================================================")
        print("SSR data only support end-to-end training currently.")
        print("Program exits now, plz check 'trainFast' in 'data_loader'.")
        print("===========================================================")
        sys.exit()
    ti_pkl = open("ProcessedData" + slash + 'train_images_fast_SSR.pkl', 'rb')
    tl_pkl = open("ProcessedData" + slash + 'train_labels_fast_SSR.pkl', 'rb')
    train_images = pickle.load(ti_pkl)
    train_labels = pickle.load(tl_pkl)
    ti_pkl.close()
    tl_pkl.close()
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(train_images)
        np.random.set_state(state)
        np.random.shuffle(train_labels)
    return train_images, train_labels


def DG4SSR():
    data_generator(
        sfs=28,
        classes_count=3,
        start_with="DustCap",
        paf=process_annotation_file,
        annotation_path=annotation,
        ti_path="ProcessedData" + slash + 'train_images_fast_SSR.pkl',
        tl_path="ProcessedData" + slash + 'train_labels_fast_SSR.pkl'
    )


def BM4SSR():
    build_model(
        classes_count=3,
        fm_layer_index=13,
        pooled_square_size=7,
        model_path="TrainedModels" + slash + "FastRCNN-SSR.h5"
    )


def TM4SSR():
    train(model_path="TrainedModels" + slash + "FastRCNN-SSR.h5", gen_data=False, dl=data_loader)


def TO4SSR():
    batch_test(
        model_path="TrainedModels" + slash + "FastRCNN-SSR.h5",
        path=img_path3,
        start_with="DustCap"
    )


def CB4SSR():
    CheckBatch(dl=data_loader)


# DG4SSR()
# BM4SSR()
TM4SSR()
# TO4SSR()
# CB4SSR()

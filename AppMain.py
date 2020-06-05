from RCNN import path, train
from forVOC2007 import transfer_model_train, data_generator
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import requests
import time
import json
import cv2
import os


def time_test(model_path="TrainedModels\\RCNN.h5", file_path=path):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    model_load_start = time.process_time()
    model_loaded = tf.keras.models.load_model(model_path)
    model_load_end = time.process_time()
    model_load_elapsed = model_load_end - model_load_start
    print("模型载入时间：" + str(model_load_elapsed))
    for e, i in enumerate(os.listdir(file_path)):
        record = ["模型载入时间：" + str(model_load_elapsed)]

        # region load an image
        pic_load_start = time.process_time()
        img = cv2.imread(os.path.join(file_path, i))
        pic_load_end = time.process_time()
        pic_load_elapsed = pic_load_end - pic_load_start
        print("图片载入时间：" + str(pic_load_elapsed))
        record.append("图片载入时间：" + str(pic_load_elapsed))
        # endregion

        # region generate rois for the image
        image_out = img.copy()
        roi_gen_start = time.process_time()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        ss_results = ss.process()
        roi_gen_end = time.process_time()
        roi_gen_elapsed = roi_gen_end - roi_gen_start
        print("选框生成时间：" + str(roi_gen_elapsed))
        record.append("选框生成时间：" + str(roi_gen_elapsed))
        # endregion

        # region make prediction for the image in rois
        predict_start = time.process_time()
        classify_elapsed = []
        for e_roi, result in enumerate(ss_results):
            if e_roi < 2000:
                # region classify target in 1 box
                x, y, w, h = result
                target_image = image_out[y:y + h, x:x + w]
                resized = cv2.resize(target_image, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                classify_start = time.process_time()
                out = model_loaded.predict(img)
                classify_end = time.process_time()
                classify_elapsed.append(classify_end - classify_start)
                print("单目标分类预测时间：" + str(classify_end - classify_start))
                # endregion
                if np.max(out) > 0.97 and np.argmax(out) != 0:
                    image_out = cv2.rectangle(image_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        record.append("单目标分类预测时间（最大值）：" + str(max(classify_elapsed)))
        record.append("单目标分类预测时间（最小值）：" + str(min(classify_elapsed)))
        record.append("单目标分类预测时间（平均值）：" + str(np.mean(classify_elapsed)))
        predict_end = time.process_time()
        predict_elapsed = predict_end - predict_start
        print("单图预测时间：" + str(predict_elapsed))
        record.append("单图预测时间：" + str(predict_elapsed))
        # endregion

        with open('TestResults\\test_result_' + str(e) + '.txt', "w") as f:
            f.writelines('\n'.join(record))
        plt.figure()
        b, g, r = cv2.split(image_out)
        image_out = cv2.merge([r, g, b])
        plt.imshow(image_out)
        plt.savefig('TestResults\\test_result_' + str(e) + '.jpg')
        plt.close()


def ExportModel():
    model_path = "TrainedModels\\RCNN.h5"
    model = tf.keras.models.load_model(model_path)
    signature = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input_param': model.input},
        outputs={'type': model.output}
    )
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder('TFServing\\00000123')
    builder.add_meta_graph_and_variables(
        sess=tf.compat.v1.keras.backend.get_session(),
        tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        }
    )
    builder.save()


def TFS_Client():
    server_url = 'http://134.134.50.135:8521/v1/models/rcnn:predict'
    image_url = 'ProcessedData\\Images\\428452.jpg'
    image_url = 'ProcessedData\\Images\\airplane_349.jpg'
    img = cv2.imread(image_url)
    plt.imshow(img)
    plt.show()
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    dic = {"instances": img.tolist()}
    dic_json = json.dumps(dic)
    response = requests.post(server_url, data=dic_json)
    response.raise_for_status()
    dic_json = json.loads(response.content)
    result = dic_json['predictions']
    if result[0][0] > result[0][1]:
        print("plane")
    else:
        print("no plane")


# train()
# ExportModel()
# TFS_Client()
# time_test(model_path="TrainedModels\\RCNN-VOC2007.h5", file_path="ProcessedData\\VOC2007_JPG")
# transfer_model_train()
# data_generator()

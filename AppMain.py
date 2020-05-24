from RCNN import path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import cv2
import os


def time_test():
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    model_load_start = time.process_time()
    model_loaded = tf.keras.models.load_model("TrainedModels\\RCNN.h5")
    model_load_end = time.process_time()
    model_load_elapsed = model_load_end - model_load_start
    print("模型载入时间：" + str(model_load_elapsed))
    for e, i in enumerate(os.listdir(path)):
        if i.startswith("4"):
            record = ["模型载入时间：" + str(model_load_elapsed)]

            # region load an image
            pic_load_start = time.process_time()
            img = cv2.imread(os.path.join(path, i))
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
                    if out[0][0] > 0.65:
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
            plt.imshow(image_out)
            plt.savefig('TestResults\\test_result_' + str(e) + '.png')
            plt.close()

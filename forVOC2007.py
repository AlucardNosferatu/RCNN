import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import cv2
import os

path = "ProcessedData\\VOC2007_JPG"
annotation = "ProcessedData\\VOC2007_XML"


def data_generator():
    train_images = []
    train_labels = []
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    for e, i in enumerate(os.listdir(annotation)):
        filename = i.split(".")[0] + ".jpg"
        print(e, filename)
        image = cv2.imread(os.path.join(path, filename))
        
        df = pd.read_csv(os.path.join(annotation, i))

        gtvalues = []

        for row in df.iterrows():
            x1 = int(row[1][0].split(" ")[0])
            y1 = int(row[1][0].split(" ")[1])
            x2 = int(row[1][0].split(" ")[2])
            y2 = int(row[1][0].split(" ")[3])
            gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
            # 把标签的坐标数据存入gtvalues

        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        # 加载SS ROI提出器

        imout = image.copy()
        counter = 0
        falsecounter = 0
        flag = 0
        fflag = 0
        bflag = 0
        for e, result in enumerate(ssresults):
            if e < 2000 and flag == 0:
                # 对SS产生的头2k个结果（坐标）进行处理
                for gtval in gtvalues:
                    # 对这张图上多个标签坐标进行处理
                    x, y, w, h = result
                    iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                    # 计算候选坐标和这一标签坐标的交并比
                    if counter < 30:
                        # 选择交并比大于阈值的头30个候选坐标
                        if iou > 0.70:
                            # 交并比阈值0.7
                            timage = imout[y:y + h, x:x + w]
                            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(1)
                            counter += 1
                    else:
                        fflag = 1
                    if falsecounter < 30:
                        # IoU低于阈值0.3，前30个坐标作为负样本（背景）
                        if iou < 0.3:
                            timage = imout[y:y + h, x:x + w]
                            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(0)
                            falsecounter += 1
                    else:
                        bflag = 1
                if fflag == 1 and bflag == 1:
                    # print("inside")
                    flag = 1
    TI_PKL = open('train_images.pkl', 'wb')
    TL_PKL = open('train_labels.pkl', 'wb')
    pickle.dump(train_images, TI_PKL)
    pickle.dump(train_labels, TL_PKL)
    TI_PKL.close()
    TL_PKL.close()
    X_new = np.array(train_images)
    y_new = np.array(train_labels)
    return X_new, y_new


def data_loader():
    TI_PKL = open('train_images.pkl', 'rb')
    TL_PKL = open('train_labels.pkl', 'rb')
    train_images = pickle.load(TI_PKL)
    train_labels = pickle.load(TL_PKL)
    TI_PKL.close()
    TL_PKL.close()
    X_new = np.array(train_images)
    y_new = np.array(train_labels)
    return X_new, y_new

def transfer_train():
    model_loaded = tf.keras.models.load_model("TrainedModels\\RCNN.h5")

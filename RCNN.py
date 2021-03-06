import cv2
import os
import pickle
import platform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# region 介绍
# 这个文件是模型训练的主要文件，目前已修改为可在Win10和Linux运行的版本
# 模型结构为单纯的图像分类+SS暴力提候选框
# endregion


# 这是用于检测运行操作系统的代码
# 根据操作系统更换路径的分隔符号
Linux = False
if platform.system() == "Linux":
    Linux = True
if Linux:
    print("Linux")
    slash = "/"
else:
    print("Windows")
    slash = "\\"

# 训练数据的图像和标签路径
path = "ProcessedData" + slash + "Images"
annotation = "ProcessedData" + slash + "Airplanes_Annotations"


# 用于把1位输出（0或1，适用于sigmoid）转为独热码（[1,0]和[0,1]，适用于softmax）
class OneHot(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == "binary":
            return np.hstack((1 - Y, Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == "binary":
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


# 测试函数，不会用于实际训练或预测
# 用于检查查看SS候选框效果
def show_SS_results():
    for e, i in enumerate(os.listdir(annotation)):
        if e < 10:
            filename = i.split(".")[0] + ".jpg"
            img = cv2.imread(os.path.join(path, filename))
            path_instance = os.path.join(annotation, i)
            f = open(path_instance)
            df = pd.read_csv(f)
            f.close()
            plt.imshow(img)
            plt.show()
            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            plt.figure()
            plt.imshow(img)
            plt.show()
            break
    cv2.setUseOptimized(True);
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    im = cv2.imread(os.path.join(path, "42850.jpg"))
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()
    rect_list = ss.process()
    image_out = im.copy()
    for i, rect in (enumerate(rect_list)):
        x, y, w, h = rect
        #     print(x,y,w,h)
        #     imOut = imOut[x:x+w,y:y+h]
        cv2.rectangle(image_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
    # plt.figure()
    plt.imshow(image_out)
    plt.show()


# 计算候选框与GroundTruth框（即标签的目标框）交并比
def get_iou(bb1, bb2):
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# 用于生成训练数据
# SS提候选框与GT求交并比
# 交并比大于0.7为正样本
# 交并比小于0.3为负样本
# 数据会保存为pkl文件
def data_generator():
    train_images = []
    train_labels = []
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    for e, i in enumerate(os.listdir(annotation)):
        # 对每一个标记文件（csv）进行操作
        try:
            if i.startswith("airplane"):
                # 只有名称带airplane才是有目标的存在的样本
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
        except Exception as e:
            print(e)
            print("error in " + filename)
            continue
    TI_PKL = open("ProcessedData" + slash + "train_images_cnn.pkl", "wb")
    TL_PKL = open("ProcessedData" + slash + "train_labels_cnn.pkl", "wb")
    pickle.dump(train_images, TI_PKL)
    pickle.dump(train_labels, TL_PKL)
    TI_PKL.close()
    TL_PKL.close()
    X_new = np.array(train_images)
    y_new = np.array(train_labels)
    return X_new, y_new


# 读取已经生成的数据文件
# 读出格式为numpy矩阵
def data_loader():
    TI_PKL = open("ProcessedData" + slash + "train_images_cnn.pkl", "rb")
    TL_PKL = open("ProcessedData" + slash + "train_labels_cnn.pkl", "rb")
    train_images = pickle.load(TI_PKL)
    train_labels = pickle.load(TL_PKL)
    TI_PKL.close()
    TL_PKL.close()
    X_new = np.array(train_images)
    y_new = np.array(train_labels)
    return X_new, y_new


# 训练模型的函数
# NewModel为True会覆盖已有训练结果生成新模型
# GenData会生成新数据包（PKL）覆盖旧数据包
def train(NewModel=False, GenData=False):
    if NewModel:
        vgg_model = tf.keras.applications.VGG16(weights="imagenet", include_top=True)
        for layers in vgg_model.layers[:15]:
            layers.trainable = False
        X = vgg_model.layers[-2].output
        predictions = Dense(2, activation="softmax")(X)
        model_final = Model(inputs=vgg_model.input, outputs=predictions)
        opt = Adam(lr=0.0001)
        model_final.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=opt,
            metrics=["accuracy"]
        )
    else:
        model_final = tf.keras.models.load_model("TrainedModels" + slash + "RCNN.h5")

    if GenData:
        x_new, y_new = data_generator()
    else:
        x_new, y_new = data_loader()

    lenc = OneHot()
    y = lenc.fit_transform(y_new)
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.5)
    trdata = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    traindata = trdata.flow(
        x=x_train,
        y=y_train
    )
    tsdata = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    testdata = tsdata.flow(
        x=x_test,
        y=y_test
    )
    checkpoint = ModelCheckpoint(
        "TrainedModels" + slash + "RCNN.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch"
    )
    early = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=100,
        verbose=1,
        mode="auto"
    )
    with tf.device("/gpu:0"):
        hist = model_final.fit_generator(
            callbacks=[checkpoint],
            validation_data=testdata,
            validation_steps=2,
            generator=traindata,
            steps_per_epoch=50,
            epochs=1000
        )
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Loss", "Validation Loss"])
    plt.show()
    plt.savefig("chart loss.png")


# 测试函数，不会用于实际训练
# 用于单独测试CNN分类器性能
# 给定已经裁切好的目标图像
# 模型输出目标图像的分类概率
def test_model_cl():
    model_final = tf.keras.models.load_model("TrainedModels" + slash + "RCNN.h5")
    x_new, y_new = data_loader()
    lenc = OneHot()
    y = lenc.fit_transform(y_new)
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.10)
    for test in x_test:
        im = test
        plt.imshow(im)
        img = np.expand_dims(im, axis=0)
        out = model_final.predict(img)

        if out[0][1] > out[0][0]:
            print(str(out) + " " + "plane")
        else:
            print(str(out) + " " + "not plane")
        plt.show()


# 测试函数，用于呈现完整的识别流程
# 分类判别采用绝对阈值判别
# 绝对阈值为0.65
def test_model_od(model_path="TrainedModels" + slash + "RCNN.h5", start_with_str="4", img_path=path):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    model_loaded = tf.keras.models.load_model(model_path)
    z = 0
    for e, i in enumerate(os.listdir(img_path)):
        if i.startswith(start_with_str):
            z += 1
            img = cv2.imread(os.path.join(img_path, i))
            # img = cv2.resize(img, (1024, 768))
            image_out = img.copy()
            # Selective Search will be replaced by ROI proposal
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            ss_results = ss.process()
            for e_roi, result in enumerate(ss_results):
                if e_roi < 2000:
                    print("当前选框编号：", e_roi)
                    x, y, w, h = result
                    target_image = image_out[y:y + h, x:x + w]
                    resized = cv2.resize(target_image, (224, 224), interpolation=cv2.INTER_AREA)
                    img = np.expand_dims(resized, axis=0)
                    out = model_loaded.predict(img)
                    # print(out)
                    if np.argmax(out[0]) != 0:
                        print("预测结果：", np.argmax(out[0]))
                        image_out = cv2.rectangle(image_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
                    # else:
                    #     image_out = cv2.rectangle(image_out, (x, y), (x + w, y + h), (255, 0, 0), 1, cv2.LINE_AA)
            plt.figure()
            plt.imshow(image_out)
            plt.show()


# 激活GPU显存增长模式
# 如果CUDA报错切换使用该函数试试
# 可能造成内存碎片
def Activate_GPU():
    gpu_list = tf.config.experimental.list_physical_devices(device_type="GPU")
    print(gpu_list)
    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)

# Activate_GPU()
# train()

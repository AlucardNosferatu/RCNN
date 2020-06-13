import datetime
import os
import cv2
import pickle
import random
import platform
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from ROI_Pooling import RoiPoolingConv
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed


# region 介绍
# 这个文件是模型训练的主要文件，目前已修改为可在Win10和Linux运行的版本
# 模型结构为带有ROI池化的CNN分类器+SS暴力提候选框
# endregion


# 用于检测运行的操作系统
def get_slash():
    is_linux = False
    if platform.system() == "Linux":
        is_linux = True
    if is_linux:
        print("Linux")
        sl = "/"
    else:
        print("Windows")
        sl = "\\"
    return sl, is_linux


slash, linux = get_slash()

# 设置训练epoch和batch_size
EP = 100
BS = 4


# 1位转独热码
class OneHot(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


# 训练数据的图片和标签文件
img_path = "ProcessedData" + slash + "Images"
annotation = "ProcessedData" + slash + "Airplanes_Annotations"


# 计算交并比
def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# 搭建并保存模型：
# ↓VGG16到展平层之前的部分+ROI输入（个数可变）
# ↓ROI池化层
# ↓展平层
# ↓4096维全连接（抄VGG16的展平后）
# ↓4096维全连接（抄VGG16的展平后）
# 2维全连接，softmax输出
def build_model(
        classes_count=1,
        model_path="TrainedModels" + slash + "FastRCNN.h5",
        fm_layer_index=17,
        pooled_square_size=7
):
    roi_input = Input(shape=(None, 4), name="input_2")
    model_cnn = tf.keras.applications.VGG16(
        include_top=True,
        weights='imagenet'
    )
    model_cnn.trainable = True
    x = model_cnn.layers[fm_layer_index].output
    x = RoiPoolingConv(pooled_square_size)([x, roi_input])
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(
        Dense(
            4096,
            activation='selu',
            kernel_initializer=RandomNormal(stddev=0.01),
            kernel_regularizer=l2(0.0005),
            bias_regularizer=l2(0.0005)
        )
    )(x)
    x = TimeDistributed(
        Dense(
            4096,
            activation='selu',
            kernel_initializer=RandomNormal(stddev=0.01),
            kernel_regularizer=l2(0.0005),
            bias_regularizer=l2(0.0005)
        )
    )(x)
    x = TimeDistributed(Dense(classes_count + 1, activation='softmax', kernel_initializer='zero'))(x)
    model_final = Model(inputs=[model_cnn.input, roi_input], outputs=x)
    opt = Adam(lr=0.0001)
    model_final.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )
    if not linux:
        tf.keras.utils.plot_model(
            model_final,
            "model.png",
            show_shapes=True,
            show_layer_names=False,
            rankdir='TB'
        )
    model_final.save(model_path)


# 测试模型（单图）
# 已注释部分用于生成并保存ROI池化前和池化后的特征图
# 会让你的测试变得非常非常慢
# 慎用
def test_model(image, roi, model, file_name):
    # fm_before_roip = Model(inputs=model.input, outputs=model.layers[13].output)
    # fm_after_roip = Model(inputs=model.input, outputs=model.layers[16].output)
    # fm_b = fm_before_roip.predict([image / 255, roi])
    # fm_a = fm_after_roip.predict([image / 255, roi])
    result = model.predict([preprocess_input(image), roi])
    # plt.figure()
    # for i in tqdm(range(fm_b.shape[3])):
    #     plt.subplot(16, 32, i + 1)
    #     plt.axis('off')
    #     plt.imshow(fm_b[0, :, :, i])
    # plt.savefig("TestResults" + slash + file_name + "fm_b.jpg")
    # plt.close()
    # for i in range(fm_a.shape[1]):
    #     plt.figure()
    #     for j in tqdm(range(fm_a.shape[4])):
    #         plt.subplot(16, 32, j + 1)
    #         plt.axis('off')
    #         plt.imshow(fm_a[0, i, :, :, j])
    #     plt.savefig("TestResults" + slash + file_name + "fm_a_" + str(i) + ".jpg")
    #     plt.close()
    image = np.array(image).astype('uint8').reshape((224, 224, 3))
    for i in range(result.shape[1]):
        x1 = int(roi[0][i][0] * 224)
        y1 = int(roi[0][i][1] * 224)
        x2 = int(roi[0][i][2] * 224)
        y2 = int(roi[0][i][3] * 224)
        if np.argmax(result[0, i, :]) != 0:
            print(str(result[0, i, :]) + " " + "plane")
            image = cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
        else:
            # print(str(result[0, i, :]) + " " + "not plane")
            # image = cv2.rectangle(
            #     image,
            #     (x1, y1),
            #     (x2, y2),
            #     (255, 0, 0),
            #     1,
            #     cv2.LINE_AA
            #
            pass
    plt.figure()
    plt.imshow(image)
    plt.savefig("TestResults" + slash + file_name + "_result.jpg")
    # plt.show()
    plt.close()


# 测试模型（批量）
# 从SS生成的候选框种选出64个合理
# （x1<x2且y1<y2）
# （四指标均不超出图片边际）
# （w和h大小均保证在特征图上大等于1个像素）
# 的候选框，若选框个数不足
# 会从已有选框复制补充
def batch_test(
        model_path="TrainedModels" + slash + "FastRCNN.h5",
        start_with="4",
        path=img_path

):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'RoiPoolingConv': RoiPoolingConv}
    )
    for e, i in enumerate(os.listdir(path)):
        if i.startswith(start_with):
            image = cv2.imread(os.path.join(path, i))
            src_w = image.shape[1]
            src_h = image.shape[0]
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            ss_results = ss.process()
            image = np.expand_dims(cv2.resize(image, (224, 224)), axis=0)
            rois_list = []
            for e_roi, result in enumerate(ss_results):
                if len(rois_list) >= 64:
                    continue
                x, y, w, h = result
                if w <= (src_w / 28) or h <= (src_h / 28):
                    continue
                else:
                    x1 = x
                    y1 = y
                    x2 = x1 + w
                    y2 = y1 + h
                    x1 = float(x1) / src_w
                    y1 = float(y1) / src_h
                    x2 = float(x2) / src_w
                    y2 = float(y2) / src_h
                rois_list.append([x1, y1, x2, y2])
            length = len(rois_list)
            while len(rois_list) < 64:
                index = random.randint(0, length - 1)
                rois_list.append(rois_list[index])
            roi = np.array(rois_list, dtype='float32').reshape((1, -1, 4))
            # try:
            test_model(image, roi, model, i)
            # except Exception as e:
            #     print(repr(e))


# 对单图采集并打包roi
# ss_results: SS生成的候选框
# gt_values: GT框list
# rois_count_per_img: 每张图要生成的roi共计个数
def rois_pack_up(
        ss_results,
        gt_values,
        gt_labels,
        rois_count_per_img,
        src_img_width=256,
        src_img_height=256,
        smallest_fm_size=14,
        classes_count=1,
        check_image=None
):
    skip = False

    th_w = src_img_width / smallest_fm_size
    th_h = src_img_height / smallest_fm_size
    rois = []
    labels = []
    count = 0
    false_count = 0
    for e, result in enumerate(ss_results):
        x, y, w, h = result
        x1 = x
        y1 = y
        x2 = x1 + w
        y2 = y1 + h

        if check_image is not None:
            # print("size_th: ", th_h, th_w)
            # print("roi_size: ", h, w)
            cv2.imshow("check 1_3", check_image[y1:y2, x1:x2])
            cv2.waitKey()
            cv2.destroyAllWindows()
        this_label = 0
        if abs(x1 - x2) <= th_w or abs(y1 - y2) <= th_h:
            # print("Lower than size threshold, discarded.")
            continue
        iou = 0
        for i in range(len(gt_values)):
            temp = get_iou(gt_values[i], {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
            if temp > iou:
                iou = temp
                if gt_labels is not None:
                    this_label = gt_labels[i]
        # if iou != 0:
        #     print("IoU: ", iou)
        if gt_labels is not None:
            n = this_label
            this_label = [0] * n
            this_label += [1]
            n = classes_count - n
            if n:
                this_label += ([0] * n)
        else:
            this_label = None
        if iou > 0.7:
            if this_label:
                labels.append(this_label)
            else:
                labels.append([1, 0])
            rois.append(
                [
                    x1 / src_img_width,
                    y1 / src_img_height,
                    x2 / src_img_width,
                    y2 / src_img_height
                ]
            )
            count += 1
        if false_count <= count:
            if iou < 0.3:
                if this_label:
                    labels.append(this_label)
                else:
                    labels.append([0, 1])
                rois.append(
                    [
                        x1 / src_img_width,
                        y1 / src_img_height,
                        x2 / src_img_width,
                        y2 / src_img_height
                    ]
                )
                false_count += 1
    length = len(rois)
    print(length)
    if len(labels) <= 1:
        skip = True
    else:
        for i in range(0, rois_count_per_img - length):
            index = random.randint(0, length - 1)
            rois.append(rois[index])
            labels.append(labels[index])
        while len(rois) > rois_count_per_img and (([1] + ([0] * classes_count)) in labels):
            index = labels.index(([1] + ([0] * classes_count)))
            del labels[index]
            del rois[index]
        labels = labels[:rois_count_per_img]
        rois = rois[:rois_count_per_img]
    return rois, labels, skip


# 处理roi，对多于批处理个数的roi切分为不同batch
# 负样本过多会跳过该图片的roi打包处理
def process_image_and_rois(
        train_images,
        train_labels,
        ss_results,
        gt_values,
        gt_labels,
        image_out,
        classes_count,
        sfs,
        checkImg=False
):
    save_dir = "TestResults" + slash + "FastRCNN" + slash + "train_batch"
    max_rois_per_batch = 64
    w = image_out.shape[1]
    h = image_out.shape[0]
    resized = cv2.resize(image_out, (224, 224), interpolation=cv2.INTER_AREA)
    resized = resized.reshape((224, 224, 3))
    check_image = None
    if checkImg:
        check_image = image_out
    rois, labels, skip = rois_pack_up(
        ss_results,
        gt_values,
        gt_labels,
        rois_count_per_img=128,
        src_img_width=w,
        src_img_height=h,
        smallest_fm_size=sfs,
        classes_count=classes_count,
        check_image=check_image
    )
    if not skip:
        for i in range(0, int(len(rois) / max_rois_per_batch)):
            train_labels.append(
                np.array(
                    labels[i * max_rois_per_batch:(i + 1) * max_rois_per_batch]
                ).reshape(max_rois_per_batch, classes_count + 1)
            )
            train_images.append(
                [
                    resized,
                    np.array(
                        rois[i * max_rois_per_batch:(i + 1) * max_rois_per_batch]
                    ).reshape(max_rois_per_batch, 4)
                ]
            )
    else:
        print("Due to too many negative samples, skip this img")
        plt.figure()
        plt.imshow(resized)
        plt.savefig(
            save_dir + slash + str(datetime.datetime.now()).replace(":", "") + ".jpg"
        )
        # plt.show()
        plt.close()
    return train_images, train_labels


# 从标签文件获取GT框列表、图片对象及SS的候选框集合
def process_annotation_file(i):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    filename = i.split(".")[0] + ".jpg"
    print(filename)
    image_out = cv2.imread(os.path.join(path, filename))
    df = pd.read_csv(os.path.join(annotation, i))
    gt_values = []
    for row in df.iterrows():
        x1 = int(row[1][0].split(" ")[0])
        y1 = int(row[1][0].split(" ")[1])
        x2 = int(row[1][0].split(" ")[2])
        y2 = int(row[1][0].split(" ")[3])
        gt_values.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
        # 把标签的坐标数据存入gt_values
    ss.setBaseImage(image_out)
    ss.switchToSelectiveSearchFast()
    ss_results = ss.process()
    return ss_results, image_out, gt_values


# 生成训练数据
def data_generator(
        sfs=14,
        classes_count=1,
        start_with="airplane",
        paf=process_annotation_file,
        annotation_path=annotation,
        ti_path="ProcessedData" + slash + 'train_images_fast.pkl',
        tl_path="ProcessedData" + slash + 'train_labels_fast.pkl'
):
    train_images = []
    train_labels = []
    for e, i in enumerate(os.listdir(annotation_path)):
        # 对每一个标记文件（csv）进行操作
        if i.startswith(start_with):
            ss_results, image_out, gt_values = paf(i)
            gt_labels = None
            if type(gt_values) == dict:
                gt_labels = gt_values['labels']
                gt_values = gt_values['values']
            train_images, train_labels = process_image_and_rois(
                train_images,
                train_labels,
                ss_results,
                gt_values,
                gt_labels,
                image_out,
                classes_count,
                sfs=sfs,
                # checkImg=("DustCap (1)_3" in i)
                checkImg=False
            )
    ti_pkl = open(ti_path, 'wb')
    tl_pkl = open(tl_path, 'wb')
    pickle.dump(train_images, ti_pkl)
    pickle.dump(train_labels, tl_pkl)
    ti_pkl.close()
    tl_pkl.close()
    return train_images, train_labels


# 读取训练数据
# trainFast: 设为True时用于端到端训练FastRCNN，设为False时用于训练VGG16做特征提取
# shuffle: 对读到的数据做随机打乱
def data_loader(trainFast=True, shuffle=True):
    if trainFast:
        ti_pkl = open("ProcessedData" + slash + 'train_images_fast.pkl', 'rb')
        tl_pkl = open("ProcessedData" + slash + 'train_labels_fast.pkl', 'rb')
    else:
        ti_pkl = open("ProcessedData" + slash + 'train_images_cnn.pkl', 'rb')
        tl_pkl = open("ProcessedData" + slash + 'train_labels_cnn.pkl', 'rb')
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


# 训练
def train(model_path="TrainedModels" + slash + "FastRCNN.h5", gen_data=False, dl=data_loader):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'RoiPoolingConv': RoiPoolingConv}
    )
    opt = Adam(lr=0.0001)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )
    for layer in model.layers:
        print(layer.name + "  " + str(layer.trainable))
    if not linux:
        tf.keras.utils.plot_model(
            model,
            "model.png",
            show_shapes=True,
            show_layer_names=False,
            rankdir='TB'
        )
    if gen_data:
        x, y = data_generator()
    else:
        x, y = dl()

    x_images = []
    x_rois = []
    for each in x:
        x_images.append(each[0])
        x_rois.append(each[1])
    x_images = np.array(x_images)
    x_images = preprocess_input(x_images)
    x_rois = np.array(x_rois)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir="TensorBoard",
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=False,
        verbose=1
    )

    with tf.device('/gpu:0'):
        model.fit(
            [x_images, x_rois],
            np.array(y),
            verbose=1,
            epochs=EP,
            batch_size=BS,
            callbacks=[cp_callback, tb_callback],
            # steps_per_epoch=100
        )


# 测试函数，不在实际训练与预测过程中使用
# 用于切分数据，检查出无效（导致ROI池化前选取面积为0）ROI
def data_divider(data):
    data = list(data)
    db = data[:int(len(data) / 2)]
    for i in range(0, int(len(data) / 2)):
        del data[int(len(data) / 2)]
    return data, db


# 测试函数，配合切分函数使用
# 细分测试数据以查找出导致零特征面积的无效ROI
def data_sampler(x, y):
    x, xb = data_divider(x)
    y, yb = data_divider(y)
    x, xc = data_divider(x)
    y, yc = data_divider(y)
    x, xd = data_divider(x)
    y, yd = data_divider(y)
    x, xe = data_divider(x)
    y, ye = data_divider(y)
    x, xf = data_divider(x)
    y, yf = data_divider(y)
    x, xg = data_divider(x)
    y, yg = data_divider(y)
    x, xh = data_divider(x)
    y, yh = data_divider(y)
    x, xi = data_divider(x)
    y, yi = data_divider(y)
    x, xj = data_divider(x)
    y, yj = data_divider(y)
    del xb, yb, xc, yc, xd, yd, xe, ye, xf, yf, xg, yg, xh, yh, xi, yi, xj, yj
    return x, y


# 清洗掉可能导致零面积的无效ROI
# 仅可用于ROI个数=1的情况
# 现已集成到ROI打包函数内，不再使用
def data_cleaner(X, y):
    th = 1 / 14
    i = 0
    length = len(X)
    while i < length:
        roi = list(X[i][1].reshape(4))
        x1 = roi[0]
        y1 = roi[1]
        x2 = roi[2]
        y2 = roi[3]
        if abs(x1 - x2) <= th or abs(y1 - y2) <= th:
            print(roi)
            del X[i], y[i]
        else:
            i += 1
        length = len(X)
    return X, y


# 激活GPU显存使用增长模式
def Activate_GPU():
    gpu_list = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(gpu_list)
    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)


# 检查生成的测试数据
# 主要用于观察正负样本比例
def CheckBatch(trainFast=True):
    x, y = data_loader(trainFast, shuffle=False)
    if trainFast:
        for i in range(len(x)):
            print(i)
            image = x[i][0].copy()
            rois = x[i][1]
            for j in range(rois.shape[0]):
                roi = rois[j, :]
                # print(roi)
                x1 = int(roi[0] * 224)
                y1 = int(roi[1] * 224)
                x2 = int(roi[2] * 224)
                y2 = int(roi[3] * 224)
                if y[i][j, 1]:
                    image = cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )
                else:
                    image = cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
            plt.figure()
            plt.imshow(image)
            plt.savefig("TestResults" + slash + "FastRCNN" + slash + "train_batch" + slash + str(i) + ".jpg")
            # plt.show()
            plt.close()
    else:
        for i in range(len(x)):
            if y[i]:
                plt.figure()
                plt.imshow(x[i])
                plt.show()
    print("Done")

# CheckBatch()
# Activate_GPU()
# batch_test()
# train()
# build_model()
# data_generator()

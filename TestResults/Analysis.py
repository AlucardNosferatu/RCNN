import os
import numpy as np

# file_path = "标准RPN效果"
file_path = "自制RPN效果"
scores = []
for e, i in enumerate(os.listdir(file_path)):
    scores.append(int(i.split('命中比例：')[1].split('%.jpg')[0]))
scores_array = np.array(scores)
mean_score = np.mean(scores_array)
std_error = np.std(scores_array)
max_score = np.max(scores_array)
min_score = np.min(scores_array)
print("平均值：" + str(mean_score))
print("标准差：" + str(std_error))
print("最大值：" + str(max_score))
print("最小值：" + str(min_score))

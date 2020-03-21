## 基础工具
import numpy as np
import pandas as pd

## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
# Train_data = pd.read_csv('datalab/used_car_train_20200313.csv', sep=' ')
# TestA_data = pd.read_csv('datalab/used_car_testA_20200313.csv', sep=' ')
## 输出数据的大小信息
# print('Train data shape:',Train_data.shape)
# print('TestA data shape:',TestA_data.shape)

# Train_data.head()
# print(Train_data.head())

# 分类准确率分数
# from sklearn.metrics import accuracy_score
# # 预测结果
# y_pred = [0, 1, 0, 1]
# # 正确结果
# y_true = [0, 1, 1, 1]
# print('ACC:',accuracy_score(y_true, y_pred))

## Precision,Recall,F1-score
# from sklearn import metrics
# y_pred = [0, 1, 0, 0]
# y_true = [0, 1, 0, 1]
# # 准确率，1为正类，关注预测为1(正类)，预测为正类且正确的/所有预测为正类
# print('Precision',metrics.precision_score(y_true, y_pred))
# # 召回率，预测为正类且正确/样本中所有正类
# print('Recall',metrics.recall_score(y_true, y_pred))
# # F1 = 2 * P * R / (P + R)
# print('F1-score:',metrics.f1_score(y_true, y_pred))

## AUC
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print('AUC socre:',roc_auc_score(y_true, y_scores))

#task-2 stash 2
#task-1 stash apply
#在贮藏上新建一个分支，当前修改将贮藏
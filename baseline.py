import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

## 定义了一个统计函数，方便后续信息统计
def Sta_inf(data):
    print('_min',np.min(data))
    print('_max:',np.max(data))
    print('_mean',np.mean(data))
    print('_ptp',np.ptp(data))
    print('_std',np.std(data))
    print('_var',np.var(data))


Train_data = pd.read_csv('datalab/used_car_train_20200313.csv', sep=' ')
TestA_data = pd.read_csv('datalab/used_car_testA_20200313.csv', sep=' ')

numerical_cols = Train_data.select_dtypes(exclude='object').columns
# print(numerical_cols)

categorical_cols = Train_data.select_dtypes(include='object').columns
# print(categorical_cols)

## 选择特征列
feature_cols = [col for col in numerical_cols if col not in ['SaleID','name','regDate','creatDate','price','model','brand','regionCode','seller']]
# print(feature_cols)
feature_cols = [col for col in feature_cols if 'Type' not in col]
print(feature_cols)

## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]
Y_data = Train_data['price']

X_test = TestA_data[feature_cols]
# 默认值填充
X_data = X_data.fillna(-1)
X_test = X_test.fillna(-1)

# print('X train shape:',X_data.shape)
# print('X test shape:',X_test.shape)

# print('Sta of "price" label:')
# Sta_inf(Y_data)

## 绘制标签的统计图，查看标签分布
# plt.hist(Y_data)
# plt.show()
# plt.close()


## xgb-Model
xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0, subsample=0.8,
                       colsample_bytree=0.9, max_depth=7)  # ,objective ='reg:squarederror'

scores_train = []
scores = []

## 5折交叉验证方式
sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_ind, val_ind in sk.split(X_data, Y_data):
    train_x = X_data.iloc[train_ind].values
    train_y = Y_data.iloc[train_ind]
    val_x = X_data.iloc[val_ind].values
    val_y = Y_data.iloc[val_ind]

    xgr.fit(train_x, train_y)
    pred_train_xgb = xgr.predict(train_x)
    pred_xgb = xgr.predict(val_x)

    score_train = mean_absolute_error(train_y, pred_train_xgb)
    scores_train.append(score_train)
    score = mean_absolute_error(val_y, pred_xgb)
    scores.append(score)

print('Train mae:', np.mean(score_train))
print('Val mae', np.mean(scores))
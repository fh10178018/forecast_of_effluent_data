# In[1]
from pandas import read_csv
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
# In[2]
# 获取数据
names = [
    "time",
    "溶解态惰性有机物质",
    "块速生物降解有机物",
    "颗粒态惰性有机物",
    "颗粒态慢速生物降解有机物",
    "氨氮",
    "硝态氮",
    "溶解态可生物降解有机氮",
    "颗粒态可生物降解有机氮",
    "异养微生物量",
    "自养微生物量",
    "由微生物衰减而产生的颗粒态惰性物质",
    "溶解氧",
    "碱度",
    "污水流量",
    "BOD",
    "COD",
    "N",
    "S_NH",
    "TSS"
]
dataset = read_csv('data_set/dry_data.csv', header=None,
                   index_col=0, names=names)
data = DataFrame(dataset)
data = data.reset_index()
# In[3]
# 数据预处理
# 数据缺失值处理
print("---------缺失值数量与处理--------")
print(data.isnull().sum())  # 显示缺失值数量
data = data.dropna()  # 缺失值直接删除

# 数据中值降噪
array = np.array(data)
arrayLength = len(array)
dataLength = len(names)
newArray = array
for y in range(1, arrayLength-2):
    for x in range(1, dataLength-1):
        for number in range(50):
            newArray[y][x] = (newArray[y-1][x] + 2 *
                              newArray[y][x] + newArray[y+1][x]) / 4
newData = DataFrame(newArray)

# 展示某个入参数据的降噪曲线图


def showNoiseReductionCurve(title):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data.loc[102:302, ["time"]], data.loc[102:302, [
            title]], data.loc[102:302, ["time"]], newData.loc[102:302, [names.index(title)]])
    ax.set_title(title + '的降噪处理')
    ax.set_xlabel('时间/s')
    ax.set_ylabel(title+"数据")
    plt.show()


showNoiseReductionCurve(title="溶解氧")

# In[4]
# 拆分数据,与数据归一化处理
train_data = data.loc[0:700]  # 选取一部分数据作为训练集
test_data = data.loc[700:]  # 选取一部分数据作为测试集


def scale(train, test):
    # 根据训练数据建立缩放器
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # 转换train data
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # 转换test data
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# 逆缩放，将数据还原回来
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


print(train_data)
# reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print ('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# %%

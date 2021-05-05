# In[1]
from pandas import read_csv
from pandas import DataFrame
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt 

from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.models import Sequential
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
dataset = read_csv('data_set/dry_data.csv',header=None, index_col=0,names=names)
data = DataFrame(dataset)
data = data.reset_index()
# In[3]
# 数据预处理
# 数据缺失值处理
print("---------缺失值数量与处理--------")
print(data.isnull().sum()) # 显示缺失值数量
data = data.dropna() # 缺失值直接删除

#数据中值降噪
array = np.array(data)
arrayLength = len(array)
dataLength = len(names)
newArray = array
for y in range(1,arrayLength-2):
  for x in range(1,dataLength-1):
    for number in range(50):
      newArray[y][x] = (newArray[y-1][x] + 2 * newArray[y][x] + newArray[y+1][x]) / 4
newData = DataFrame(newArray)

# 展示某个入参数据的降噪曲线图
def showNoiseReductionCurve(title):
  plt.rcParams['font.sans-serif']=['SimHei']
  plt.rcParams['axes.unicode_minus'] = False
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(data.loc[102:302,["time"]], data.loc[102:302,[title]], data.loc[102:302,["time"]], newData.loc[102:302,[names.index(title)]])
  ax.set_title(title + '的降噪处理')
  ax.set_xlabel('时间/s')
  ax.set_ylabel(title+"数据")
  plt.show()
showNoiseReductionCurve(title="溶解氧")

# In[4]
# 拆分数据
train_data = data.loc[0:700] # 选取一部分数据作为训练集
test_data = data.loc[700:] # 选取一部分数据作为测试集
print(train_data)
#reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print ('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# In[4]
# 训练LSTM模型
def fit_lstm(X,y, batch_size, nb_epoch, neurons):
    model = Sequential()
    # 添加LSTM层
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))  # 输出层1个node
    # 编译，损失函数mse+优化算法adam
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # 按照batch_size，一次读取batch_size个数据
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
        print("当前计算次数："+str(i))
    return model

def forcast_lstm(model, batch_size, X):
  X = np.array(X).reshape(1, 1, 1)
  print(X)
  yhat = model.predict(X, batch_size=batch_size)
  return yhat[0, 0]

X = list()
Y = list()
X = [x + 1 for x in range(20)]
Y = [y * 15 for y in X]
X = np.array(X).reshape(len(X),1,1)
Y = np.array(Y).reshape(len(Y),1,1)
lstm_model = fit_lstm(X,Y,1,100,4)
# yhat = forcast_lstm(model,1,X)

test_x = [x + 2 for x in range(20)]
test_data = np.column_stack((test_x, [y * 15 for y in test_x]))
predictions = list()
for i in range(len(test_data)):
  X,Y = test_data[i,0:-1], test_data[i,-1]
  print(X)
  yhat = forcast_lstm(lstm_model, 1,[X])
  predictions.append(yhat)
print(predictions)
Y = plt.plot(test_x, [y * 15 for y in test_x],test_x,predictions)
plt.show()
# %%

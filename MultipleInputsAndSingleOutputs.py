# 该测试采用模型设计模式为多输入单输出
# In[1]
from keras.utils.vis_utils import plot_model
from pandas import read_csv
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
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
# 数据缺失值处理和剔除重复数据的列
print("---------缺失值数量与处理--------")
print(data.isnull().sum())  # 显示缺失值数量
data = data.dropna()  # 缺失值直接删除
# 删除不变化的列，减少输入
data.drop(["溶解态惰性有机物质","硝态氮","溶解态可生物降解有机氮","颗粒态可生物降解有机氮","异养微生物量","碱度"], axis=1)

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
    plt.rcParams['font.sans-serif'] = ['simsun']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data.loc[102:302, ["time"]],data.loc[102:302, [title]],label=title+"实际值")
    ax.plot(data.loc[102:302, ["time"]], newData.loc[102:302, [names.index(title)]],'r--',label=title+"降噪值")
    # ax.set_title(title + '的降噪处理')
    ax.set_xlabel('时间/s')
    ax.set_ylabel(title+"数据")
    plt.legend()
    plt.show()


showNoiseReductionCurve(title="溶解氧")

# In[4]
# 拆分数据

train_data = data.iloc[0:700,1:]  # 选取一部分数据作为训练集,不包括时间戳
test_data = data.iloc[700:,1:]  # 选取一部分数据作为测试集，不包括时间戳
test_time = data.iloc[700:,0]
train_data = np.array(train_data) # 转成numpy对象，方便处理
test_data = np.array(test_data)

# In[5]
# 数据归一化（数据压缩）
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
scaler, train_scaled, test_scaled = scale(train_data,test_data)

# In[6]
# 建立lstm模型
# 构建模型，采取的是多输入，单输出构建模型
time_start=time.time() # 模型训练开始计算
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, Y = train[:, :-5], train[:, -5:]
    #reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model_list = list()
    for i in range(5):
      model = Sequential()
      # 添加LSTM层
      # return_sequences设置为true，预测输出输出数量才能与输入数量一致
      model.add(LSTM(neurons, batch_input_shape=(
          batch_size, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
      model.add(Dense(1))  # 输出层1个node
      # 编译，损失函数mse+优化算法adam
      model.compile(loss='mae', optimizer='adam')
      for j in range(nb_epoch):
          # 按照batch_size，一次读取batch_size个数据
          model.fit(X, Y[:,i], epochs=1, batch_size=batch_size,
                    verbose=0, shuffle=False)
          model.reset_states()
      model_list.append(model)
    return model_list
lstm_model_list = fit_lstm(train_scaled,1,100,4)

time_end=time.time() # 模型训练结束
print('模型训练结束，总耗时'+ str(time_end-time_start)+"s")

print(lstm_model_list[0].summary()) # Summarize Model
plot_model(lstm_model_list[0], to_file='model2.png',show_shapes=True)

# In[7]
# lstm模型预测
test_X, test_Y = test_scaled[:, :-5], test_scaled[:, -5:]
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

# 对5个输出进行预测
predict_y_list = list()

for i in range(len(lstm_model_list)):
  model = lstm_model_list[i]
  predictions = model.predict(test_X,batch_size=1)
  if i==0:predict_y_list = predictions
  else: predict_y_list = np.column_stack((predict_y_list,predictions))
print("--------数据预测结束----------")
# In[7]
# 数据逆缩放
# 逆缩放，将数据还原回来
test_time = data.iloc[700:,0]
def invert_scale(scaler, X, Y):
  new_row = np.column_stack((X,Y))
  inverted = scaler.inverse_transform(new_row)
  return inverted[:,-5:]
test_x = test_X.reshape((test_X.shape[0], test_X.shape[2]))
predict_y = predict_y_list.reshape((predict_y_list.shape[0], predict_y_list.shape[1]))
rmseList = []
# 在数据逆缩放之前就进行均方差计算，让数据之间有可比性
for z in range(5):
  rmseList.append(sqrt(mean_squared_error(test_Y[:, z-5], predict_y[:, z])))

yhat_p = invert_scale(scaler,test_x,predict_y)

# In[8]
# 数据图像可视化展示
for z in range(5):
  print('%s的RMSE: %.3f' %(names[15+z],rmseList[z]))
  plt.rcParams['font.sans-serif']=['SimHei']
  plt.rcParams['axes.unicode_minus']=False
  plt.plot(test_time,test_data[:, z-5],label=names[15+z]+"真实值")
  plt.plot(test_time,yhat_p[:, z],'r--',label=names[15+z]+"预测值")
  plt.legend()
  plt.show()

# %%

# 数据缩放，在建模或者测试模型之前一定要缩放到 -1 到 1 之间，不然会影响建模精度
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


def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    # 添加LSTM层
    model.add(LSTM(neurons, batch_input_shape=(
        batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))  # 输出层1个node
    # 编译，损失函数mse+优化算法adam
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # 按照batch_size，一次读取batch_size个数据
        model.fit(X, y, epochs=1, batch_size=batch_size,
                  verbose=0, shuffle=False)
        model.reset_states()
        print("当前模型计算次数："+str(i))
    return model


def forcast_lstm(model, batch_size, X):
    X = np.array(X).reshape(1, 1, 1)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


X = list()
Y = list()
X = [x + 1 for x in range(20)]
Y = [y * 15 for y in X]
train = np.column_stack((X, Y))
test_x = [x + 2.7 for x in range(20)]
test = np.column_stack((test_x, [y * 15 for y in test_x]))
print(train, test)
# 数据缩放
scaler, train_scaled, test_scaled = scale(train, test)
print(train_scaled, test_scaled)
# 建立LSTM数据，模型
lstm_model = fit_lstm(train_scaled, 1, 100, 4)

predictions = list()
for i in range(len(test_scaled)):
    X, Y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forcast_lstm(lstm_model, 1, X)
    yhat = invert_scale(scaler, X, yhat)
    predictions.append(yhat)
Y = plt.plot(test_x, [y * 15 for y in test_x], test_x, predictions)
plt.show()

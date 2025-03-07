import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 500
BATCH_SIZE = 16

# Чтение и стандартизация данных
boston_housing = keras.datasets.boston_housing
(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data()
x_mean = np.mean(raw_x_train, axis=0)
x_stddev = np.std(raw_x_train, axis=0)
x_train = (raw_x_train - x_mean) / x_stddev
x_test = (raw_x_test - x_mean) / x_stddev

# Вариант 1: Один слой, один нейрон, линейная активация
model_1 = Sequential()
model_1.add(Dense(1, activation='linear', input_shape=[x_train.shape[1]]))
model_1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model_1.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Вариант 2a: Добавление L2-регуляризации (λ = 0.1)
model_2a = Sequential()
model_2a.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1), input_shape=[x_train.shape[1]]))
model_2a.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))
model_2a.add(Dense(1, activation='linear'))
model_2a.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model_2a.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Вариант 2b: Добавление Dropout (0.2)
model_2b = Sequential()
model_2b.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1), input_shape=[x_train.shape[1]]))
model_2b.add(Dropout(0.2))
model_2b.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))
model_2b.add(Dropout(0.2))
model_2b.add(Dense(1, activation='linear'))
model_2b.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model_2b.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Вариант 2c: Увеличение количества нейронов до 128, добавление ещё одного слоя
model_2c = Sequential()
model_2c.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1), input_shape=[x_train.shape[1]]))
model_2c.add(Dropout(0.2))
model_2c.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1)))
model_2c.add(Dropout(0.2))
model_2c.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))
model_2c.add(Dense(1, activation='linear'))
model_2c.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model_2c.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Вариант 2d: Увеличение коэффициента дропаута до 0.3
model_2d = Sequential()
model_2d.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1), input_shape=[x_train.shape[1]]))
model_2d.add(Dropout(0.3))
model_2d.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1)))
model_2d.add(Dropout(0.3))
model_2d.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))
model_2d.add(Dense(1, activation='linear'))
model_2d.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model_2d.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Вывод первых 4 прогнозов для каждого варианта
models = {'Model 1': model_1, 'Model 2a': model_2a, 'Model 2b': model_2b, 'Model 2c': model_2c, 'Model 2d': model_2d}
for name, model in models.items():
    predictions = model.predict(x_test)
    print(f'Predictions for {name}:')
    for i in range(4):
        print(f'Prediction: {predictions[i]}, True value: {y_test[i]}')

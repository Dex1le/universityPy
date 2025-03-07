import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Загрузка данных
boston_housing = keras.datasets.boston_housing
(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data()

# Стандартизация данных
x_mean = np.mean(raw_x_train, axis=0)
x_stddev = np.std(raw_x_train, axis=0)
x_train = (raw_x_train - x_mean) / x_stddev
x_test = (raw_x_test - x_mean) / x_stddev

# Создание упрощенной модели (только 1 нейрон)
model = Sequential()
model.add(Dense(1, activation='linear', input_shape=[13]))

# Компиляция модели
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Обучение модели
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=500, batch_size=16, verbose=2, shuffle=True)

# Вывод предсказаний
predictions = model.predict(x_test)
for i in range(4):
    print('Prediction:', predictions[i], ', true value:', y_test[i])
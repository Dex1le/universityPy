import tensorflow as tf  # Импортируем TensorFlow
from tensorflow import keras  # Импортируем высокоуровневый API Keras
from tensorflow.keras.models import Sequential  # Импортируем последовательную модель
from tensorflow.keras.layers import Dense, Dropout  # Импортируем полносвязный слой и слой Dropout
from tensorflow.keras.regularizers import l2  # Импортируем L2-регуляризатор
import matplotlib.pyplot as plt  # Импортируем библиотеку для построения графиков
import numpy as np  # Импортируем NumPy для работы с массивами
import logging  # Импортируем модуль логирования

tf.get_logger().setLevel(logging.ERROR)  # Отключаем вывод лишней информации от TensorFlow

EPOCHS = 500  # Количество эпох обучения
BATCH_SIZE = 16  # Размер батча

# Загружаем и стандартизируем данные
boston_housing = keras.datasets.boston_housing  # Загружаем датасет Boston Housing
(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data()  # Делим на тренировочные и тестовые данные
x_mean = np.mean(raw_x_train, axis=0)  # Считаем среднее по каждому признаку
x_stddev = np.std(raw_x_train, axis=0)  # Считаем стандартное отклонение по каждому признаку
x_train = (raw_x_train - x_mean) / x_stddev  # Стандартизируем тренировочные данные
x_test = (raw_x_test - x_mean) / x_stddev  # Стандартизируем тестовые данные

# Вариант 1: Простая линейная модель с 1 нейроном
model_1 = Sequential()  # Создаём модель
model_1.add(Dense(1, activation='linear', input_shape=[x_train.shape[1]]))  # Один линейный слой
model_1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])  # Компиляция модели
model_1.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)  # Обучаем модель

# Вариант 2a: Модель с L2-регуляризацией
model_2a = Sequential()
model_2a.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1), input_shape=[x_train.shape[1]]))  # Первый слой с L2
model_2a.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))  # Второй слой с L2
model_2a.add(Dense(1, activation='linear'))  # Выходной слой
model_2a.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model_2a.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Вариант 2b: Модель с Dropout
model_2b = Sequential()
model_2b.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1), input_shape=[x_train.shape[1]]))  # Первый слой
model_2b.add(Dropout(0.2))  # Dropout после первого слоя
model_2b.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))  # Второй слой
model_2b.add(Dropout(0.2))  # Dropout после второго слоя
model_2b.add(Dense(1, activation='linear'))  # Выходной слой
model_2b.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model_2b.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Вариант 2c: Больше нейронов и дополнительный слой
model_2c = Sequential()
model_2c.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1), input_shape=[x_train.shape[1]]))  # Первый слой
model_2c.add(Dropout(0.2))  # Dropout
model_2c.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1)))  # Второй слой
model_2c.add(Dropout(0.2))  # Dropout
model_2c.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))  # Третий слой
model_2c.add(Dense(1, activation='linear'))  # Выходной слой
model_2c.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model_2c.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Вариант 2d: Dropout увеличен до 0.3
model_2d = Sequential()
model_2d.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1), input_shape=[x_train.shape[1]]))
model_2d.add(Dropout(0.3))
model_2d.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1)))
model_2d.add(Dropout(0.3))
model_2d.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))
model_2d.add(Dense(1, activation='linear'))
model_2d.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model_2d.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Предсказания и вывод первых 4 значений для каждой модели
models = {'Model 1': model_1, 'Model 2a': model_2a, 'Model 2b': model_2b, 'Model 2c': model_2c, 'Model 2d': model_2d}  # Словарь всех моделей
for name, model in models.items():
    predictions = model.predict(x_test)  # Предсказания модели
    print(f'Predictions for {name}:')  # Название модели
    for i in range(4):
        print(f'Prediction: {predictions[i]}, True value: {y_test[i]}')  # Сравнение предсказаний с истинными значениями

# Построение графиков обучения для каждой модели
    history = model.history.history  # История обучения
    plt.figure(figsize=(12, 4))  # Размер графика

    # Потери
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history['mean_absolute_error'], label='Train MAE')
    plt.plot(history['val_mean_absolute_error'], label='Val MAE')
    plt.title(f'{name} - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

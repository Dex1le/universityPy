import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Параметры обучения
EPOCHS = 20
BATCH_SIZE = 32

# Загрузка набора данных CIFAR-10
cifar_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

# Стандартизация набора данных
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# Преобразование меток в one-hot encoding
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Создание модели с двумя сверточными слоями, Dropout и одним полносвязным
model = Sequential([
    Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same',
           input_shape=(32, 32, 3), kernel_initializer='he_normal', bias_initializer='zeros'),
    Conv2D(16, (2, 2), strides=(2, 2), activation='relu', padding='same',
           kernel_initializer='he_normal', bias_initializer='zeros'),
    Dropout(0.2),
    Flatten(),
    Dense(10, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros')
])

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Построение графиков ошибки обучения и тестовой ошибки
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss (Conf3)')
plt.grid()
plt.show()
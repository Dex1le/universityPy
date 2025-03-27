import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Загрузка и нормализация данных
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
mean, std = np.mean(train_images), np.std(train_images)
train_images = (train_images - mean) / std
test_images = (test_images - mean) / std
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)


# Конфигурации
def build_model(config_name):
    model = keras.Sequential()
    if config_name == 'Conf1':
        model.add(layers.Conv2D(64, (5, 5), strides=2, activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))

    elif config_name == 'Conf2':
        model.add(layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(layers.Conv2D(16, (2, 2), strides=2, activation='relu', padding='same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))

    elif config_name == 'Conf3':
        model.add(layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(16, (2, 2), strides=2, activation='relu', padding='same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))

    elif config_name == 'Conf4':
        model.add(layers.Conv2D(64, (4, 4), strides=1, activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, (2, 2), strides=2, activation='relu', padding='same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(10, activation='softmax'))

    elif config_name == 'Conf5':
        model.add(layers.Conv2D(64, (4, 4), strides=1, activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, (2, 2), strides=2, activation='relu', padding='same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(10, activation='softmax'))

    elif config_name == 'Conf6':
        model.add(layers.Conv2D(64, (4, 4), strides=1, activation='tanh', padding='same', input_shape=(32, 32, 3)))
        model.add(layers.Conv2D(64, (2, 2), strides=2, activation='tanh', padding='same'))
        model.add(layers.Conv2D(32, (3, 3), strides=1, activation='tanh', padding='same'))
        model.add(layers.Conv2D(32, (3, 3), strides=1, activation='tanh', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='tanh'))
        model.add(layers.Dense(64, activation='tanh'))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Обучение и визуализация
configs = ['Conf1', 'Conf2', 'Conf3', 'Conf4', 'Conf5', 'Conf6']
for cfg in configs:
    print(f'🔧 Тренируем {cfg}...')
    model = build_model(cfg)
    history = model.fit(train_images, train_labels, epochs=10, batch_size=64,
                        validation_data=(test_images, test_labels), verbose=0)

    # График
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title(cfg)
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    _, acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f'✅ Точность на тесте: {acc:.4f}\n')
    #gh

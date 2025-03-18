import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import os

# Гиперпараметры
IMG_SIZE = 500
BATCH_SIZE = 128
EPOCHS = 20
FINE_TUNE_EPOCHS = 5

# Функция для загрузки данных из CIFAR-10
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch[b'labels'])
    return images, labels

# Указываем путь к распакованным данным
data_path = r"C:\Users\Public\Documents\cifar-10-batches-py"

# Загрузка тренировочных данных
train_images, train_labels = [], []
for i in range(1, 6):
    images, labels = load_cifar10_batch(os.path.join(data_path, f"data_batch_{i}"))
    train_images.append(images)
    train_labels.append(labels)

train_images = np.vstack(train_images)
train_labels = np.hstack(train_labels)

# Загрузка тестового набора
test_images, test_labels = load_cifar10_batch(os.path.join(data_path, "test_batch"))

# Нормализация данных
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-hot кодирование меток
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Функция для отображения случайных 10 изображений
def show_random_images(images, labels, class_names, num_samples=10):
    indices = random.sample(range(len(images)), num_samples)
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx])
        plt.title(class_names[np.argmax(labels[idx])])
        plt.axis("off")
    plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
show_random_images(test_images, test_labels, class_names)

# Создание модели по конфигурации 6
model = Sequential([
    Conv2D(64, (4, 4), activation='tanh', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (2, 2), activation='tanh', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='tanh', padding='same'),
    Conv2D(32, (3, 3), activation='tanh', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Размораживание слоев для дообучения (fine-tuning)
for layer in model.layers[:-3]:
    layer.trainable = True

# Компиляция после разморозки
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Переобучение модели
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=FINE_TUNE_EPOCHS, batch_size=BATCH_SIZE)

# Отображение случайных предсказаний
def show_random_predictions(model, images, labels, class_names, num_samples=10):
    indices = random.sample(range(len(images)), num_samples)
    predictions = model.predict(images[indices])
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx])
        plt.title(f"Pred: {class_names[np.argmax(predictions[i])]}\nTrue: {class_names[np.argmax(labels[idx])]}")
        plt.axis("off")
    plt.show()

show_random_predictions(model, test_images, test_labels, class_names)

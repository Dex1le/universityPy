import tensorflow as tf  # Импорт TensorFlow
from tensorflow import keras  # Импорт Keras (входит в состав TensorFlow)
from tensorflow.keras.utils import to_categorical  # Функция для кодирования меток в one-hot формат
import numpy as np  # Библиотека для работы с массивами
import logging  # Библиотека для логирования
from tensorflow.keras.layers import BatchNormalization, Activation  # Импортируем слои нормализации и активации
import matplotlib.pyplot as plt  # Библиотека для построения графиков
import random  # Для выбора случайных примеров

# Отключаем лишние предупреждения TensorFlow
tf.get_logger().setLevel(logging.ERROR)

# Фиксируем случайное зерно для воспроизводимости результатов
tf.random.set_seed(7)

# Гиперпараметры обучения
EPOCHS = 20  # Количество эпох
BATCH_SIZE = 16  # Размер пакета данных

# Загружаем датасет MNIST (рукописные цифры)
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Стандартизация изображений
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# Кодируем метки классов в формат one-hot encoding
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Создаем модель нейронной сети
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(25, kernel_initializer="he_normal", bias_initializer="zeros"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("tanh"),

    keras.layers.Dense(10, kernel_initializer="he_normal", bias_initializer="zeros"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("sigmoid"),
])

# Создаем оптимизатор SGD (градиентный спуск)
opt = keras.optimizers.SGD(learning_rate=0.01)

# Компилируем модель
model.compile(loss='MSE', optimizer=opt, metrics=['accuracy'])

# Обучение модели
history = model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    shuffle=True,
)

# Достаем историю обучения
loss = history.history["loss"]
val_loss = history.history["val_loss"]
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
epochs_range = range(1, len(loss) + 1)

# Визуализация результатов обучения
plt.figure(figsize=(12, 5))

# График потерь
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# График точности
plt.subplot(1, 2, 2)
plt.plot(epochs_range, accuracy, label="Training Accuracy")
plt.plot(epochs_range, val_accuracy, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.show()

# --- Добавляем вывод примеров распознавания ---

# Выбираем 10 случайных изображений из тестового набора
num_samples = 10
random_indices = random.sample(range(len(test_images)), num_samples)
sample_images = test_images[random_indices]
sample_labels = np.argmax(test_labels[random_indices], axis=1)  # Истинные метки

# Делаем предсказание модели
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)  # Получаем индексы предсказанных классов

# Визуализируем результаты
plt.figure(figsize=(12, 6))
for i in range(num_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i], cmap="gray")
    plt.axis("off")
    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {sample_labels[i]}")

plt.suptitle("Примеры распознавания цифр")
plt.show()
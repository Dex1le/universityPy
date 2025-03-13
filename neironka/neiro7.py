import tensorflow as tf  # Импорт TensorFlow
from tensorflow import keras  # Импорт Keras (входит в состав TensorFlow)
from tensorflow.keras.utils import to_categorical  # Функция для кодирования меток в one-hot формат
import numpy as np  # Библиотека для работы с массивами
import logging  # Библиотека для логирования
from tensorflow.keras.layers import BatchNormalization, Activation  # Импортируем слои нормализации и активации
import matplotlib.pyplot as plt  # Библиотека для построения графиков

# Отключаем лишние предупреждения TensorFlow
tf.get_logger().setLevel(logging.ERROR)

# Фиксируем случайное зерно для воспроизводимости результатов
tf.random.set_seed(7)

# Гиперпараметры обучения
EPOCHS = 20  # Количество эпох (сколько раз модель пройдет по всему набору данных)
BATCH_SIZE = 16  # Размер пакета данных при обучении (batch size)

# Загружаем датасет MNIST (рукописные цифры)
mnist = keras.datasets.mnist
(train_images, train_labels), (
test_images, test_labels) = mnist.load_data()  # Разбиваем данные на обучающую и тестовую выборку

# Стандартизация изображений (приведение значений к нормальному распределению)
mean = np.mean(train_images)  # Вычисляем среднее значение по пикселям
stddev = np.std(train_images)  # Вычисляем стандартное отклонение
train_images = (train_images - mean) / stddev  # Нормализация обучающего набора данных
test_images = (test_images - mean) / stddev  # Нормализация тестового набора данных

# Кодируем метки классов в формат one-hot encoding (пример: 3 -> [0,0,0,1,0,0,0,0,0,0])
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Создаем модель нейронной сети
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # Входной слой: превращает 28x28 изображение в одномерный массив (784 элемента)

    keras.layers.Dense(25, kernel_initializer="he_normal", bias_initializer="zeros"),
    # Первый скрытый слой с 25 нейронами
    keras.layers.BatchNormalization(),  # Пакетная нормализация (ускоряет обучение, стабилизирует градиенты)
    keras.layers.Activation("tanh"),  # Функция активации tanh (используется для нелинейности)

    keras.layers.Dense(10, kernel_initializer="he_normal", bias_initializer="zeros"),
    # Выходной слой с 10 нейронами (по количеству классов)
    keras.layers.BatchNormalization(),  # Пакетная нормализация для выхода
    keras.layers.Activation("sigmoid"),  # Функция активации sigmoid (используется для многоклассовой классификации)
])

# Создаем оптимизатор SGD (градиентный спуск) с шагом обучения 0.01
opt = keras.optimizers.SGD(learning_rate=0.01)

# Компилируем модель:
# - loss='MSE' — функция ошибки (среднеквадратичная ошибка, обычно лучше использовать categorical_crossentropy)
# - optimizer=opt — метод оптимизации
# - metrics=['accuracy'] — отслеживаем точность классификации
model.compile(loss='MSE', optimizer=opt, metrics=['accuracy'])

# Обучение модели на данных MNIST
history = model.fit(
    train_images,  # Обучающие изображения
    train_labels,  # Метки классов для обучения
    validation_data=(test_images, test_labels),  # Данные для валидации (проверки точности на тестовых данных)
    epochs=EPOCHS,  # Количество эпох
    batch_size=BATCH_SIZE,  # Размер пакета (по одному изображению за раз)
    verbose=2,  # Уровень логирования (2 — показывает краткую информацию о процессе обучения)
    shuffle=True,  # Перемешиваем данные перед каждой эпохой для лучшего обучения
)

# Достаем историю обучения
loss = history.history["loss"]  # Потери на обучающих данных
val_loss = history.history["val_loss"]  # Потери на тестовых данных
accuracy = history.history["accuracy"]  # Точность на обучающих данных
val_accuracy = history.history["val_accuracy"]  # Точность на тестовых данных
epochs_range = range(1, len(loss) + 1)  # Диапазон эпох

# Визуализация результатов обучения
plt.figure(figsize=(12, 5))

# График потерь (Loss Curve)
plt.subplot(1, 2, 1)  # Первое изображение в сетке 1x2
plt.plot(epochs_range, loss, label="Training Loss")  # Потери на обучении
plt.plot(epochs_range, val_loss, label="Validation Loss")  # Потери на валидации
plt.xlabel("Epochs")  # Подпись оси X
plt.ylabel("Loss")  # Подпись оси Y
plt.title("Loss Curve")  # Заголовок графика
plt.legend()  # Показываем легенду

# График точности (Accuracy Curve)
plt.subplot(1, 2, 2)  # Второе изображение в сетке 1x2
plt.plot(epochs_range, accuracy, label="Training Accuracy")  # Точность на обучении
plt.plot(epochs_range, val_accuracy, label="Validation Accuracy")  # Точность на валидации
plt.xlabel("Epochs")  # Подпись оси X
plt.ylabel("Accuracy")  # Подпись оси Y
plt.title("Accuracy Curve")  # Заголовок графика
plt.legend()  # Показываем легенду

# Отображаем графики
plt.show()
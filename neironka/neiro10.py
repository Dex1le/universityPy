# Импорт библиотек для построения нейросетей и визуализации
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model  # Sequential — для простой модели, Model — для кастомной
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Input  # Слои модели
from tensorflow.keras.utils import to_categorical  # Преобразование меток в one-hot
import numpy as np  # Для работы с массивами
import matplotlib.pyplot as plt  # Для построения графиков

# === CIFAR-10: Загрузка встроенного датасета из 10 классов
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Вычисляем среднее и стандартное отклонение изображений
mean = np.mean(train_images)
stddev = np.std(train_images)

# Нормализуем изображения: делаем нули средними, единицу — стандартным отклонением
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# Переводим метки в формат one-hot, например: 3 → [0,0,0,1,0,0,0,0,0,0]
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)


conf5_layers = [
    Conv2D(64, (4, 4), strides=(1, 1), activation="relu", padding="same"),  # Свёртка 64 фильтра 4x4, шаг 1
    Conv2D(64, (2, 2), strides=(2, 2), activation="relu", padding="same"),  # Свёртка 64 фильтра 2x2, шаг 2 (уменьшит размер вдвое)
    Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same"),  # Свёртка 32 фильтра
    Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same"),  # Ещё одна свёртка
    MaxPooling2D(pool_size=(2, 2)),  # Пулинг — уменьшаем размер изображения
    Dropout(0.2),  # Dropout — регуляризация, случайно отключаем 20% нейронов
    Dense(64, activation="relu"),  # Полносвязный слой на 64 нейрона
    Dropout(0.2),  # Ещё один Dropout
    Dense(64, activation="relu"),  # Ещё полносвязный слой
    Flatten(),  # Преобразуем тензор в вектор перед последним слоем
    Dropout(0.2),  # Последний Dropout
    Dense(10, activation="softmax"),  # Выходной слой: 10 классов, softmax для вероятностей
]

# Собираем модель: вход 32x32x3 + все слои из списка
cifar_model = Sequential([keras.Input(shape=(32, 32, 3))] + conf5_layers)

# Компиляция модели: оптимизатор Adam, кросс-энтропия как функция потерь, метрика — accuracy
cifar_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Обучение модели
history = cifar_model.fit(
    train_images,              # входные изображения
    train_labels,              # соответствующие метки
    validation_data=(test_images, test_labels),  # валидация на test-наборе
    epochs=20,                  # 5 эпох обучения
    batch_size=32,            # размер батча
    verbose=2                 # подробный вывод
)

# Сохраняем модель в файл
cifar_model.save("cifar_model.h5")
print("Модель успешно сохранена в 'cifar_model.h5'")

# === График потерь на обучении и валидации
plt.figure(figsize=(12, 5))
plt.plot(history.history["loss"], label="Train Loss")         # Потери на обучении
plt.plot(history.history["val_loss"], label="Val Loss")       # Потери на валидации
plt.legend()
plt.title("Потери на обучении и валидации")
plt.grid()
plt.show()

# === График точности на обучении и валидации
plt.figure(figsize=(12, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")      # Точность на обучении
plt.plot(history.history["val_accuracy"], label="Val Accuracy")    # Точность на валидации
plt.legend()
plt.title("Точность на обучении и валидации")
plt.grid()
plt.show()

# === CIFAR-100: используем как новую задачу для трансферного обучения
(cifar100_x, cifar100_y), (cifar100_val_x, cifar100_val_y) = keras.datasets.cifar100.load_data(label_mode="fine")

# Увеличиваем изображения с 32x32 до 500x500 и нормализуем их
cifar100_x = tf.image.resize(cifar100_x, (500, 500)) / 255.0
cifar100_val_x = tf.image.resize(cifar100_val_x, (500, 500)) / 255.0

# Переводим метки CIFAR-100 в формат one-hot (100 классов)
cifar100_y = to_categorical(cifar100_y, 100)
cifar100_val_y = to_categorical(cifar100_val_y, 100)

# Загружаем предобученную модель (CIFAR-10)
base_model = keras.models.load_model("cifar_model.h5")
print("Предобученная модель успешно загружена")

# Удаляем последний слой (Dense на 10 классов) и замораживаем остальные
base_model_layers = base_model.layers[:-1]
for layer in base_model_layers:
    layer.trainable = False  # Не обновляем веса этих слоёв при обучении

# Новый входной слой для изображений 500x500x3
new_input = Input(shape=(500, 500, 3))
x = new_input

# Прогоняем вход через все слои старой модели (без последнего)
for layer in base_model_layers:
    x = layer(x)

# Новый выходной слой: 100 классов вместо 10
output = Dense(100, activation="softmax")(x)

# Создаём новую модель с новым входом и выходом
lung_model = Model(inputs=new_input, outputs=output)

# Компилируем модель
lung_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Обучаем модель на CIFAR-100 (трансферное обучение)
history = lung_model.fit(
    cifar100_x,
    cifar100_y,
    validation_data=(cifar100_val_x, cifar100_val_y),
    epochs=20,      # Кол-во эпох
    verbose=2      # Подробный вывод
)

# === График потерь трансферной модели
plt.figure(figsize=(12, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Потери на обучении и валидации (CIFAR-100)")
plt.grid()
plt.show()

# === График точности трансферной модели
plt.figure(figsize=(12, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Точность на обучении и валидации (CIFAR-100)")
plt.grid()
plt.show()
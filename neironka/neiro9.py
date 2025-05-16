import tensorflow as tf  # Импортируем TensorFlow
from tensorflow import keras  # Импортируем Keras из TensorFlow
from tensorflow.keras import layers  # Импортируем слои Keras
from tensorflow.keras.utils import to_categorical  # Функция для one-hot кодирования
import numpy as np  # Импорт библиотеки NumPy
import matplotlib.pyplot as plt  # Импорт для построения графиков

# Загрузка и нормализация данных
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()  # Загружаем датасет CIFAR-10
mean, std = np.mean(train_images), np.std(train_images)  # Считаем среднее и стандартное отклонение по train-данным
train_images = (train_images - mean) / std  # Нормализуем train-данные
test_images = (test_images - mean) / std  # Нормализуем test-данные теми же значениями
train_labels = to_categorical(train_labels, 10)  # Переводим метки в one-hot формат
test_labels = to_categorical(test_labels, 10)  # Аналогично для test

# Названия классов CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',  # Список имён классов
               'dog', 'frog', 'horse', 'ship', 'truck']

# Конфигурации

def build_model(config_name):  # Функция построения модели по имени конфигурации
    model = keras.Sequential()  # Создаём последовательную модель
    if config_name == 'Conf1':
        model.add(layers.Conv2D(64, (5,5), strides=2, activation='relu', padding='same', input_shape=(32,32,3)))  # 1-й сверточный слой
        model.add(layers.Conv2D(64, (3,3), strides=2, activation='relu', padding='same'))  # 2-й сверточный слой
        model.add(layers.Flatten())  # Преобразуем feature map в вектор
        model.add(layers.Dense(10, activation='softmax'))  # Выходной слой

    elif config_name == 'Conf2':
        model.add(layers.Conv2D(64, (3,3), strides=2, activation='relu', padding='same', input_shape=(32,32,3)))  # 1-й свёрточный слой
        model.add(layers.Conv2D(16, (2,2), strides=2, activation='relu', padding='same'))  # 2-й свёрточный слой с меньшим кол-ом фильтров
        model.add(layers.Flatten())  # Flatten
        model.add(layers.Dense(10, activation='softmax'))  # Классификация

    elif config_name == 'Conf3':
        model.add(layers.Conv2D(64, (3,3), strides=2, activation='relu', padding='same', input_shape=(32,32,3)))  # 1-й свёрточный слой
        model.add(layers.Dropout(0.2))  # Dropout для регуляризации
        model.add(layers.Conv2D(16, (2,2), strides=2, activation='relu', padding='same'))  # 2-й свёрточный слой
        model.add(layers.Dropout(0.2))  # Dropout
        model.add(layers.Flatten())  # Flatten
        model.add(layers.Dense(10, activation='softmax'))  # Выход

    elif config_name == 'Conf4':
        model.add(layers.Conv2D(64, (4,4), strides=1, activation='relu', padding='same', input_shape=(32,32,3)))  # Свертка
        model.add(layers.Dropout(0.2))  # Dropout
        model.add(layers.Conv2D(64, (2,2), strides=2, activation='relu', padding='same'))  # Свертка
        model.add(layers.Dropout(0.2))  # Dropout
        model.add(layers.Conv2D(32, (3,3), strides=1, activation='relu', padding='same'))  # Свертка
        model.add(layers.Dropout(0.2))  # Dropout
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))  # Пулинг для уменьшения размера
        model.add(layers.Flatten())  # Flatten
        model.add(layers.Dense(64, activation='relu'))  # Плотный слой
        model.add(layers.Dropout(0.2))  # Dropout
        model.add(layers.Dense(10, activation='softmax'))  # Классификация

    elif config_name == 'Conf5':
        model.add(layers.Conv2D(64, (4,4), strides=1, activation='relu', padding='same', input_shape=(32,32,3)))  # Свёртка
        model.add(layers.Dropout(0.2))  # Dropout
        model.add(layers.Conv2D(64, (2,2), strides=2, activation='relu', padding='same'))  # Свёртка
        model.add(layers.Dropout(0.2))  # Dropout
        model.add(layers.Conv2D(32, (3,3), strides=1, activation='relu', padding='same'))  # Свёртка
        model.add(layers.Dropout(0.2))  # Dropout
        model.add(layers.Conv2D(32, (3,3), strides=1, activation='relu', padding='same'))  # Свёртка
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))  # Пулинг
        model.add(layers.Flatten())  # Flatten
        model.add(layers.Dense(64, activation='relu'))  # Полносвязный
        model.add(layers.Dropout(0.2))  # Dropout
        model.add(layers.Dense(64, activation='relu'))  # Полносвязный
        model.add(layers.Dropout(0.2))  # Dropout
        model.add(layers.Dense(10, activation='softmax'))  # Классификация

    elif config_name == 'Conf6':
        model.add(layers.Conv2D(64, (4,4), strides=1, activation='tanh', padding='same', input_shape=(32,32,3)))  # Свёртка с tanh
        model.add(layers.Conv2D(64, (2,2), strides=2, activation='tanh', padding='same'))  # Свёртка с tanh
        model.add(layers.Conv2D(32, (3,3), strides=1, activation='tanh', padding='same'))  # Свёртка с tanh
        model.add(layers.Conv2D(32, (3,3), strides=1, activation='tanh', padding='same'))  # Свёртка с tanh
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))  # Пулинг
        model.add(layers.Flatten())  # Flatten
        model.add(layers.Dense(64, activation='tanh'))  # Полносвязный
        model.add(layers.Dense(64, activation='tanh'))  # Полносвязный
        model.add(layers.Dense(10, activation='softmax'))  # Классификация
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])  # Компиляция модели с MSE
        return model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Компиляция модели
    return model  # Возврат модели

# Обучение и визуализация
configs = ['Conf1', 'Conf2', 'Conf3', 'Conf4', 'Conf5', 'Conf6']  # Список конфигураций

for cfg in configs:
    print(f'Тренируем {cfg}...')  # Лог в консоль
    model = build_model(cfg)  # Создаём модель
    history = model.fit(train_images, train_labels, epochs=10, batch_size=64,  # Обучаем модель
                        validation_data=(test_images, test_labels), verbose=0)

    # График
    plt.figure(figsize=(6,4))  # Размер фигуры
    plt.plot(history.history['loss'], label='train')  # Линия обучения
    plt.plot(history.history['val_loss'], label='test')  # Линия валидации
    plt.title(cfg)  # Заголовок — имя конфигурации
    plt.xlabel('Эпоха')  # Подпись оси X
    plt.ylabel('Loss')  # Подпись оси Y
    plt.legend()  # Легенда
    plt.grid(True)  # Сетка
    plt.tight_layout()  # Автоматический layout
    plt.show()  # Показ графика

    _, acc = model.evaluate(test_images, test_labels, verbose=0)  # Оцениваем точность на тесте
    print(f'Точность на тесте: {acc:.4f}\n')  # Выводим точность

    # Примеры распознавания
    sample_indexes = np.random.choice(len(test_images), 5, replace=False)  # Берём 5 случайных индексов
    sample_images = test_images[sample_indexes]  # Соответствующие картинки
    sample_labels = test_labels[sample_indexes]  # И их метки
    predictions = model.predict(sample_images, verbose=0)  # Предсказания модели

    plt.figure(figsize=(10, 2))  # Размер окна с примерами
    for i, img in enumerate(sample_images):  # Для каждой картинки
        plt.subplot(1, 5, i + 1)  # Подграф
        plt.imshow((img * std + mean).astype(np.uint8))  # Возвращаем картинке нормальные цвета
        true_label = class_names[np.argmax(sample_labels[i])]  # Истинная метка
        pred_label = class_names[np.argmax(predictions[i])]  # Предсказанная
        plt.title(f'True: {true_label}\nPred: {pred_label}')  # Подпись
        plt.axis('off')  # Без осей
    plt.tight_layout()  # Layout
    plt.show()  # Показ
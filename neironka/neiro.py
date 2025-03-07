import matplotlib.pyplot as plt
import numpy as np
import random

def show_learning(w):
    print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])

# Определяем переменные, необходимые для процесса обучения

random.seed(7)  # Чтобы обеспечить повторяемость

LEARNING_RATE = 0.1

# Определяем обучающие примеры

x_train = [
    (1.0, -1.7, 1.9), (1.0, -0.8, 1.9), (1.0, 0.7, 1.9), (1.0, 0.8, 1.9), (1.0, 1.7, 1.9),
    (1.0, -1.7, 0.5), (1.0, -0.8, 0.5), (1.0, 0.7, 0.5), (1.0, 0.8, 0.5), (1.0, 1.7, 0.5),
    (1.0, -1.7, -0.6), (1.0, -0.8, -0.6), (1.0, 0.7, -0.6), (1.0, 0.8, -0.6), (1.0, 1.7, -0.6),
    (1.0, -1.7, -0.8), (1.0, -0.8, -0.8), (1.0, 0.7, -0.8), (1.0, 0.8, -0.8), (1.0, 1.7, -0.8),
    (1.0, -1.7, -1.9), (1.0, -0.8, -1.9), (1.0, 0.7, -1.9), (1.0, 0.8, -1.9), (1.0, 1.7, -1.9)
]  # Входы

# Метки должны соответствовать числу входов (25)
y_train = [
    1.0, 1.0, 1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, -1.0, -1.0,
    1.0, -1.0, -1.0, -1.0, -1.0,  # Исправляем метку для (-1.7, -0.6)
    -1.0, -1.0, -1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0
]

index_list = list(range(len(x_train)))

# Определяем веса персептрона
w = [0.9, -0.6, -0.5]  # Инициализируем «случайными» числами

# Печатаем начальные значения весов
show_learning(w)

# Список для сохранения весов на каждом шаге
weights_history = []

# Функция для вычисления результата персептрона
def compute_output(w, x):
    z = sum([w[i] * x[i] for i in range(len(w))])  # Вычисление суммы взвешенных входов
    return 1 if z >= 0 else -1  # Применение знаковой функции

# Цикл обучения персептрона
all_correct = False
while not all_correct:
    all_correct = True
    random.shuffle(index_list)  # Сделать порядок случайным
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w, x)  # Функция персептрона

        if y != p_out:  # Обновить веса, когда неправильно
            for j in range(len(w)):
                w[j] += y * LEARNING_RATE * x[j]
            all_correct = False
            show_learning(w)  # Показать обновлённые веса
            weights_history.append(w.copy())  # Сохраняем веса после каждого изменения

# Визуализация 1: Изменение разделяющих прямых
plt.figure(figsize=(10, 6))

# Отображаем обучающие примеры
for i, (x1, x2) in enumerate([(x[1], x[2]) for x in x_train]):
    color = "green" if y_train[i] == 1 else "red"
    plt.scatter(x1, x2, color=color, marker="+" if y_train[i] == 1 else "_")

# Отображаем линии, соответствующие изменениям весов
x_vals = np.linspace(-2, 2, 100)  # возвращает равномерно распределенные числа в указанном интервале

line_colors = ["red", "blue", "purple", "yellow"]
num_colors = len(line_colors)
for idx, weights in enumerate(weights_history):
    color = line_colors[idx % num_colors]
    y_vals = [-(weights[0] + weights[1] * x) / weights[2] for x in x_vals]
    plt.plot(x_vals, y_vals, color=color, linewidth=1)

# Финальная разделяющая линия
final_y_vals = -(w[0] + w[1] * x_vals) / w[2]
plt.plot(x_vals, final_y_vals, color="green", label="Финал обучения", linewidth=3)

plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Изменение разделяющих прямых во время обучения")
plt.grid(True)
plt.legend()
plt.show()

# Визуализация 2: Разделяющая прямая
plt.figure(figsize=(10, 6))

# Отображаем обучающие примеры
for i, (x1, x2) in enumerate([(x[1], x[2]) for x in x_train]):
    color = "green" if y_train[i] == 1 else "red"
    plt.scatter(x1, x2, color=color, marker="+" if y_train[i] == 1 else "_")

# Линия разделения: w0 + w1 * x1 + w2 * x2 = 0 => x2 = -(w0 + w1 * x1) / w2
x_vals = np.linspace(-2, 2, 100)
y_vals = -(w[0] + w[1] * x_vals) / w[2]
plt.plot(x_vals, y_vals, color="green")

plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Разделение классов после обучения")
plt.grid(True)
plt.show()
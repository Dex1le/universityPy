import numpy as np
from numpy import linspace
from scipy.integrate import solve_ivp

# Исходные данные
# Гравитационная постоянная (ускорение свободного падения)
grv_const = 25.8  # [м/с^2]
# Начальная позиция объекта
init_pos = 40  # [м]
# Начальная скорость объекта
init_vel = 0.48  # [м/с]
# Начальная масса объекта
init_mass = 24  # [кг]

# Начальные условия в виде списка
init_cond = [init_pos, init_vel, init_mass]

# Временные параметры моделирования
init_time = 0  # Начальное время [с]
final_time = 15  # Конечное время [с]
num_data = 60 * 15  # Количество временных точек
# Создаём массив временных точек для расчёта
tout = linspace(init_time, final_time, num_data)

# Функция движения объекта в свободном падении
# time - текущее время
# state - вектор состояния [позиция, скорость, масса]
# grv_const - гравитационная постоянная
def free_falling_obj(time, state, grv_const):
    _, x2, x3 = state  # Разделяем вектор состояния
    # Уравнения движения
    dxdt = [x2, grv_const + (x3 - 2) * (x2 / x3), -x3 + 2]
    return dxdt

# Решаем систему дифференциальных уравнений методом solve_ivp
sol = solve_ivp(
    free_falling_obj,  # Функция с уравнениями
    (init_time, final_time),  # Начальное и конечное время
    init_cond,  # Начальные условия
    t_eval=tout,  # Временные точки для записи результатов
    args=(grv_const,)  # Дополнительные аргументы для функции
)

# Получаем массивы результатов (позиция, скорость, масса)
xout = sol.y

# Индекс для выбора данных (в данном случае, без изменений)
sliceCount = 60 * 0

# Извлекаем данные по массе, скорости и позиции объекта
mass = xout[2, sliceCount:]
velocity = xout[1, sliceCount:]
position = xout[0, sliceCount:]
fig_tout = tout[sliceCount:]

# Расчёт энергий
kinetic_energy = -(0.5 * mass * velocity**2)  # Кинетическая энергия
potential_energy = mass * grv_const * position  # Потенциальная энергия
total_energy = kinetic_energy + potential_energy  # Полная энергия

# Определяем размер графиков
fig_size = (4.5, 4)

import matplotlib.pyplot as plt

# График изменения позиции во времени
fig1 = plt.figure(figsize=fig_size)
plt.plot(fig_tout, position)
plt.ylabel("позиция [м]")
plt.xlabel("время [с]")

# График изменения скорости во времени
fig2 = plt.figure(figsize=fig_size)
plt.plot(fig_tout, velocity)
plt.ylabel("скорость [м/с]")
plt.xlabel("время [с]")

# График изменения массы объекта во времени
fig3 = plt.figure(figsize=fig_size)
plt.plot(fig_tout, mass)
plt.ylabel("m(t) [кг]")
plt.xlabel("время [с]")

# Графики энергий во времени
fig4 = plt.figure(figsize=fig_size)
plt.subplot(3, 1, 1)
plt.plot(fig_tout, kinetic_energy, label="Кинетическая энергия", color="r")
plt.ylabel("Энергия [Дж]")
plt.xlabel("Время [с]")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(fig_tout, potential_energy, label="Потенциальная энергия", color="b")
plt.ylabel("Энергия [Дж]")
plt.xlabel("Время [с]")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(fig_tout, total_energy, label="Полная энергия", color="g")
plt.ylabel("Энергия [Дж]")
plt.xlabel("Время [с]")
plt.legend()

plt.tight_layout()
plt.show()

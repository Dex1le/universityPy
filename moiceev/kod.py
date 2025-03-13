import numpy as np
from numpy import linspace
from scipy.integrate import solve_ivp

# Исходные данные
grv_const = 25.8  # [м/с^2]
init_pos = 40  # [м]
init_vel = 0.48  # [м/с]
init_mass = 24  # [кг]

init_cond = [init_pos, init_vel, init_mass]

init_time = 0  # [с]
final_time = 15  # [с]
num_data = 60 * 15
tout = linspace(init_time, final_time, num_data)

# Функция движения
def free_falling_obj(time, state, grv_const):
    _, x2, x3 = state
    dxdt = [x2, grv_const + (x3 - 2) * (x2 / x3), -x3 + 2]
    return dxdt

# Решение системы уравнений
sol = solve_ivp(
    free_falling_obj, (init_time, final_time), init_cond, t_eval=tout, args=(grv_const,)
)
xout = sol.y

sliceCount = 60 * 0

# Данные для энергии
mass = xout[2, sliceCount:]  # Масса
velocity = xout[1, sliceCount:]  # Скорость
position = xout[0, sliceCount:]  # Позиция
fig_tout = tout[sliceCount:]

kinetic_energy = -(0.5 * mass * velocity**2)  # Кинетическая энергия
potential_energy = mass * grv_const * position  # Потенциальная энергия
total_energy = kinetic_energy + potential_energy  # Полная энергия

fig_size = (4.5, 4)

import matplotlib.pyplot as plt

# Графики позиции и скорости
fig1 = plt.figure(figsize=fig_size)
plt.plot(fig_tout, position)
plt.ylabel("позиция [м]")
plt.xlabel("время [с]")

fig2 = plt.figure(figsize=fig_size)
plt.plot(fig_tout, velocity)
plt.ylabel("скорость [м/с]")
plt.xlabel("время [с]")

fig3 = plt.figure(figsize=fig_size)
plt.plot(fig_tout, mass)
plt.ylabel("m(t) [кг]")
plt.xlabel("время [с]")

# Графики энергии
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

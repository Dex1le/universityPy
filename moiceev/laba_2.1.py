import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Угловая скорость и кинематика кватерниона
def dqdt_attitude_kinematics(t, q):
    w = np.array([
        0.1 * np.sin(2 * np.pi * 0.005 * t),
        0.05 * np.cos(2 * np.pi * 0.01 * t + 0.2),
        0.02
    ])
    wx = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])
    Omega13 = np.hstack((-wx, w.reshape(3, 1)))
    Omega4 = np.hstack((-w, [0]))
    Omega = np.vstack((Omega13, Omega4))
    dqdt = 0.5 * Omega @ q
    return dqdt

# Параметры моделирования
init_time = 0
final_time = 6000
num_data = 5000
tout = np.linspace(init_time, final_time, num_data)
q0 = np.array([0, 0, 0, 1])

# Пары допусков
tols = [
    (1e-3, 1e-6),
    (1e-6, 1e-9),
    (1e-9, 1e-12)
]

colors = ['blue', 'red', 'gold']
labels = [
    'RelTol = 1e−2',
    'RelTol = 1e−5',
    'RelTol = 1e−9'
]

# Построение графика
plt.figure(figsize=(10, 5))

for (RelTol, AbsTol), color, label in zip(tols, colors, labels):
    sol = solve_ivp(dqdt_attitude_kinematics, (init_time, final_time), q0, t_eval=tout, rtol=RelTol, atol=AbsTol)
    qout = sol.y
    norms_squared = np.sum(qout**2, axis=0)
    log_error = np.log(np.abs(1 - norms_squared))
    plt.plot(tout, log_error, color=color, label=label)

plt.xlabel('Время (с)', fontsize=14)
plt.ylabel('log|qᵀq − 1|', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

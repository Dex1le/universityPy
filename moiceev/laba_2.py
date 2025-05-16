import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

init_time = 0
final_time = 60.0
num_data = 1000
tout = np.linspace(init_time, final_time, num_data)
q0 = np.array([0, 0, 0, 1])

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

    Omega13 = np.hstack((-wx, w.reshape(3,1)))
    Omega4 = np.hstack((-w, [0]))
    Omega = np.vstack((Omega13, Omega4))

    dqdt = 0.5 * Omega @ q
    return dqdt

sol = solve_ivp(dqdt_attitude_kinematics, (init_time, final_time), q0, t_eval=tout)
qout = sol.y

fig, ax = plt.subplots()
ax.plot(tout, qout[0,:], 'b-', label='q1')
ax.plot(tout, qout[1,:], 'r--', label='q2')
ax.plot(tout, qout[2,:], 'g-.', label='q3')
ax.plot(tout, qout[3,:], 'm:', label='q4')

ax.set_xlabel('время [с]', fontsize=14)
ax.set_ylabel('кватернион', fontsize=14)
ax.legend(fontsize=14, loc='upper right')
ax.set_xticks([0,10,20,30,40,50,60])
ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
ax.set_xticklabels([0,10,20,30,40,50,60], fontsize=14)
ax.set_yticklabels([-0.5, 0.0, 0.5, 1.0], fontsize=14)
ax.set_xlim(0, 60)
ax.set_ylim(-0.5, 1.0)
fig.set_figheight(6)
fig.set_figwidth(8)

plt.show()

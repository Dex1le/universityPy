import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

# Общие параметры
N_sample = 100
N_realize = 1000
dt = 0.1
time = np.linspace(0, N_sample * dt, N_sample)
mu_all = np.linspace(-2, 2, N_sample)
sigma_all = np.linspace(0.1, 1.5, N_sample)

x_rand_all = np.zeros((N_realize, N_sample))
for idx, (mu, sigma) in enumerate(zip(mu_all, sigma_all)):
    x_rand_all[:, idx] = mu + sigma * np.random.randn(N_realize)

# ФПВ
N_bin = 100
x_bin = np.linspace(-5, 5, N_bin)
dx = np.mean(np.diff(x_bin))
px_all = np.zeros((N_bin - 1, N_sample))
for j in range(N_sample):
    N_occur, _ = np.histogram(x_rand_all[:, j], bins=x_bin)
    px_all[:, j] = N_occur / (dx * N_realize)

# Область отрисовки
x_limits = (-5, 5)
t_limits = (0, 10)
z_limits = (0, 4)
short_time_idx = np.where((time >= 0) & (time <= 10))[0]
short_x_idx = np.where((x_bin[:-1] >= -5) & (x_bin[:-1] <= 5))[0]

short_time = time[short_time_idx]
short_x_bin = x_bin[:-1][short_x_idx]
short_px = px_all[np.ix_(short_x_idx, short_time_idx)]
t_mesh, x_mesh = np.meshgrid(short_time, short_x_bin)

# Функция поверхности
def create_pdf_surface(data, title, cmap='parula', edgecolor='k', zlim=z_limits):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(t_mesh, x_mesh, data, cmap=cmap, edgecolor=edgecolor)
    ax.set_xlim(t_limits)
    ax.set_ylim(x_limits)
    ax.set_zlim(*zlim)
    ax.view_init(elev=30, azim=-135)
    ax.set_xlabel("время [с]")
    ax.set_ylabel("x")
    ax.set_zlabel(r"$\hat{p}(x)$")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# === УПРАЖНЕНИЕ 1 ===
create_pdf_surface(short_px, "shading='faceted'", cmap='viridis', edgecolor='k')
create_pdf_surface(short_px, "shading='flat'", cmap='viridis', edgecolor='none')
short_px_interp = gaussian_filter(short_px, sigma=1)
create_pdf_surface(short_px_interp, "shading='interp'", cmap='viridis', edgecolor='none')

# === УПРАЖНЕНИЕ 2 ===
fig_colormap = plt.figure(figsize=(8, 6))
ax = fig_colormap.add_subplot(111, projection='3d')
ax.plot_surface(t_mesh, x_mesh, short_px, cmap='inferno', rstride=2, cstride=2)
ax.set_xlim(t_limits)
ax.set_ylim(x_limits)
ax.set_zlim(*z_limits)
ax.view_init(elev=30, azim=-135)
ax.set_xlabel("время [с]")
ax.set_ylabel("x")
ax.set_zlabel(r"$\hat{p}(x)$")
ax.set_title("График с cmap='inferno', rstride=2, cstride=2")
plt.tight_layout()
plt.show()

# === УПРАЖНЕНИЕ 3: СЛУЧАЙНОЕ БЛУЖДАНИЕ ===
def generate_noise_walk(N=1200, dt=0.1):
    sigma_x, sigma_y, sigma_z = 0.01, 0.01, 0.02
    beta_all = np.zeros((10, N, 3))
    for i in range(10):
        beta_all[i, 0] = np.random.uniform(-0.03, 0.03, size=3)
        for k in range(1, N):
            eta = np.random.randn(3)
            delta = np.array([
                sigma_x * np.sqrt(dt) * eta[0],
                sigma_y * np.sqrt(dt) * eta[1],
                sigma_z * np.sqrt(dt) * eta[2],
            ])
            beta_all[i, k] = beta_all[i, k - 1] + delta
    return beta_all

beta_walks = generate_noise_walk()
t_long = np.linspace(0, 120, 1200)

fig_walk = plt.figure(figsize=(10, 6))
colors = ['r', 'b', 'g']
labels = [r'$\beta_x(t)$', r'$\beta_y(t)$', r'$\beta_z(t)$']

for i in range(3):
    plt.subplot(3, 1, i + 1)
    for j in range(10):
        plt.plot(t_long, beta_walks[j, :, i], color=colors[i], linewidth=0.7)
    plt.ylabel(labels[i] + " (°/с)")
    plt.ylim(-0.5, 0.5)
    plt.xlim(0, 120)
    if i == 2:
        plt.xlabel("Время (с)")

plt.tight_layout()
plt.show()

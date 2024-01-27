import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def f(x):
    return np.cos(x) * np.sin(x**2 + 8)

def df(x):
    return 2 * x * np.cos(x) * np.cos(x**2 + 8) - np.sin(x) * np.sin(x**2 + 8)

def df2(x):
    return (-4 * x**2 - 1) * np.cos(x) * np.sin(x**2 + 8) + (
        2 * np.cos(x) - 4 * x * np.sin(x)
    ) * np.cos(x**2 + 8)

def tangent_equation(x, x0, y0):
    return y0 + df(x0) * (x - x0)

def normal_equation(x, x0, y0):
    return y0 - 1 / df(x0) * (x - x0)

def plot_function(x_val, y_val, label):
    plt.plot(x_val, y_val, label=label)

def plot_tangent(x_val, x0, y0):
    plt.plot(x_val, tangent_equation(x_val, x0, y0), "--", label="Касательная")

def plot_normal(x_val, x0, y0):
    plt.plot(x_val, normal_equation(x_val, x0, y0), "--", label="Нормаль")

def main():
    # Задание интервала
    x_val = np.linspace(0, 5, 1000)
    y_val = f(x_val)

    plt.figure(figsize=(8, 6))
    plot_function(x_val, y_val, "График функции")
    plt.title("График функции")

    plt.figure(figsize=(8, 6))
    plot_function(x_val, df(x_val), "1 производная")
    plt.title("1 производная")

    plt.figure(figsize=(8, 6))
    plot_function(x_val, df2(x_val), "2 производная")
    plt.title("2 производная")

    min_index = np.argmin(y_val)
    min_point = (x_val[min_index], y_val[min_index])
    x0, y0 = min_point

    plt.figure(figsize=(8, 6))
    plot_function(x_val, f(x_val), "График функции")
    plot_tangent(x_val, x0, y0)
    plot_normal(x_val, x0, y0)
    plt.plot(x0, y0, "ro")
    plt.title("Касательная и нормаль")
    plt.legend()

    plt.figure(figsize=(8, 6))
    plot_function(x_val, f(x_val), "График функции")
    for x in np.linspace(0, 5, 5):
        plot_tangent(x_val, x, f(x))
    plt.title("Касательное расслоение")
    plt.legend()

    curve_length, _ = integrate.quad(lambda x: np.sqrt(1 + df(x) ** 2), 0, 5)

    print(f"Длина кривой: {curve_length:.2f}")

    plt.show()

if __name__ == "__main__":
    main()
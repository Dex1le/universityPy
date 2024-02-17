import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button

def lissajous(a: float, b: float, delta: float, t: float) -> tuple[float, float]:
    x = np.sin(a * t + delta)
    y = np.sin(b * t)
    return x, y

styles = {
    "Сплошная": "red -",
    "Пунктир": "blue --",
    "Штрих-пунктир": "green -.",
}

def get_styles(style: str) -> tuple[str, str]:
    color_out: str = "red"
    style_out: str = "-"
    for key in styles:
        if style == key:
            color_out, style_out = styles[key].split(" ")
    return color_out, style_out

def update_graph(a, b, delta, line_style, show_tangent, slider_t, ax):
    ax.clear()
    t_values = np.linspace(0, 2 * np.pi, 1000)
    x_values, y_values = lissajous(a, b, delta, t_values)
    color_out, style_out = get_styles(line_style)
    (line,) = ax.plot(x_values, y_values, label="Лиссажу", color=color_out, linestyle=style_out)
    (point,) = ax.plot([], [], "ro", markersize=8)
    (tangent_line,) = ax.plot([], [], "k--")

    t = slider_t.val
    x, y = lissajous(a, b, delta, t)
    point.set_data(x, y)

    if show_tangent:
        slope, intercept = tangent_line_coefficients(a, b, delta, t)
        tangent_line.set_data([-1.5, 1.5], [slope * (-1.5) + intercept, slope * 1.5 + intercept])
    else:
        tangent_line.set_data([], [])

    plt.show()

def update_point_data(val, a, b, delta, line_style, show_tangent, slider_t, ax):
    update_graph(a, b, delta, line_style, show_tangent, slider_t, ax)

def tangent_line_coefficients(a, b, delta, t):
    x, y = lissajous(a, b, delta, t)
    dx_dt = a * np.cos(a * t + delta)
    dy_dt = b * np.cos(b * t)
    slope = dy_dt / dx_dt
    intercept = y - slope * x
    return slope, intercept

def reset_point(event, slider_t, ax):
    slider_t.set_val(0)
    update_graph(a, b, delta, line_style, show_tangent, slider_t, ax)

def toggle_tangent(label, show_tangent, update_graph, slider_t, ax):
    show_tangent = not show_tangent
    update_graph(a, b, delta, line_style, show_tangent, slider_t, ax)

def change_line_style(label, line_style, update_graph, slider_t, ax):
    line_style = label
    update_graph(a, b, delta, line_style, show_tangent, slider_t, ax)

if __name__ == "__main__":
    a, b, delta = 2, 3, np.pi / 3
    show_tangent = False
    line_style = ""

    fig, ax = plt.subplots(figsize=(8, 6))

    slider_ax_t = plt.axes([0.1, 0.01, 0.65, 0.03])
    slider_t = Slider(slider_ax_t, "Время - t", 0, 2 * np.pi, valinit=0, valstep=0.01)
    slider_t.on_changed(lambda val: update_point_data(val, a, b, delta, line_style, show_tangent, slider_t, ax))

    reset_button_ax = plt.axes([0.8, 0.1, 0.1, 0.04])
    reset_button = Button(reset_button_ax, "Сброс")
    reset_button.on_clicked(lambda event: reset_point(event, slider_t, ax))

    tangent_checkbox_ax = plt.axes([0.8, 0.025, 0.15, 0.04])
    tangent_checkbox = CheckButtons(tangent_checkbox_ax, ["Касательная"], actives=[False])
    tangent_checkbox.on_clicked(lambda label: toggle_tangent(label, show_tangent, update_graph, slider_t, ax))

    radio_buttons_ax_color = plt.axes([0.6, 0.1, 0.3, 0.15])
    radio_buttons_color = RadioButtons(radio_buttons_ax_color, list(styles))
    radio_buttons_color.on_clicked(lambda label: change_line_style(label, line_style, update_graph, slider_t, ax))

    update_graph(a, b, delta, line_style, show_tangent, slider_t, ax)

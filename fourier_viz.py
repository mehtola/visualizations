"""
Fourier Series Visualization Tool

Interactive tool to visualize how Fourier series approximate arbitrary functions.
Draw a function on [-π, π], then adjust the number of terms to see the approximation
converge in real time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec

# ── Constants ──────────────────────────────────────────────────────────────────
N_MAX = 50
N_GRID = 1000
X_GRID = np.linspace(-np.pi, np.pi, N_GRID)
PI_TICKS = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
PI_LABELS = ["-π", "-π/2", "0", "π/2", "π"]


def setup_pi_xaxis(ax):
    """Configure an axis with π-formatted x-tick labels."""
    ax.set_xticks(PI_TICKS)
    ax.set_xticklabels(PI_LABELS)
    ax.set_xlim(-np.pi, np.pi)


# ── Figure and axes ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.canvas.manager.set_window_title("Fourier Series Visualization")

gs = GridSpec(3, 2, figure=fig, height_ratios=[3, 3, 0.4],
              hspace=0.35, wspace=0.3)

ax_draw = fig.add_subplot(gs[0, 0])
ax_approx = fig.add_subplot(gs[0, 1])
ax_components = fig.add_subplot(gs[1, 0])
ax_bars = fig.add_subplot(gs[1, 1])
ax_slider = fig.add_subplot(gs[2, :])

# ── Drawing Canvas (top-left) ────────────────────────────────────────────────
ax_draw.set_title("Drawing Canvas (click & drag)")
ax_draw.set_ylim(-2, 2)
setup_pi_xaxis(ax_draw)
ax_draw.grid(True, alpha=0.3)
draw_line, = ax_draw.plot([], [], "b-", linewidth=2)

# ── Approximation Plot (top-right) ───────────────────────────────────────────
ax_approx.set_title("Fourier Approximation (N=1)")
ax_approx.set_ylim(-2, 2)
setup_pi_xaxis(ax_approx)
ax_approx.grid(True, alpha=0.3)
original_line, = ax_approx.plot([], [], "--", color="gray", linewidth=1.5,
                                label="Original")
approx_line, = ax_approx.plot([], [], "-", color="red", linewidth=2,
                              label="Fourier")
ax_approx.legend(loc="upper right", fontsize=8)

# ── Fourier Components (bottom-left) ─────────────────────────────────────────
ax_components.set_title("Fourier Components")
setup_pi_xaxis(ax_components)
ax_components.grid(True, alpha=0.3)

cmap = plt.cm.viridis
dc_line, = ax_components.plot([], [], "--", color="black", linewidth=1.5,
                              label="a₀/2 (DC)")
component_lines = []
for n in range(1, N_MAX + 1):
    color = cmap(n / N_MAX)
    line, = ax_components.plot([], [], "-", color=color, linewidth=1,
                               alpha=0.8)
    component_lines.append(line)

# ── Coefficient Bar Chart (bottom-right) ─────────────────────────────────────
ax_bars.set_title("Fourier Coefficients")
ax_bars.set_xlabel("n")

# ── Slider ────────────────────────────────────────────────────────────────────
slider = Slider(ax_slider, "Terms N", 1, N_MAX, valinit=1, valstep=1,
                color="steelblue")

# ── Clear Button ──────────────────────────────────────────────────────────────
ax_clear = fig.add_axes([0.46, 0.01, 0.08, 0.03])
btn_clear = Button(ax_clear, "Clear", color="lightgray", hovercolor="salmon")

# ── State ─────────────────────────────────────────────────────────────────────
state = {
    "drawing": False,
    "xs": [],
    "ys": [],
    "f_interp": None,       # interpolated function on X_GRID
    "a_coeffs": None,       # a_0, a_1, ..., a_N_MAX
    "b_coeffs": None,       # b_1, b_2, ..., b_N_MAX
}


# ── Fourier computation ──────────────────────────────────────────────────────
def compute_coefficients(f):
    """Compute Fourier coefficients a_n and b_n for f on X_GRID."""
    a = np.zeros(N_MAX + 1)
    b = np.zeros(N_MAX + 1)

    # a_0 = (1/π) ∫ f(x) dx
    a[0] = (1 / np.pi) * np.trapezoid(f, X_GRID)

    for n in range(1, N_MAX + 1):
        cos_nx = np.cos(n * X_GRID)
        sin_nx = np.sin(n * X_GRID)
        a[n] = (1 / np.pi) * np.trapezoid(f * cos_nx, X_GRID)
        b[n] = (1 / np.pi) * np.trapezoid(f * sin_nx, X_GRID)

    return a, b


def fourier_sum(a, b, N):
    """Build the Fourier partial sum with N terms on X_GRID."""
    result = np.full_like(X_GRID, a[0] / 2.0)
    for n in range(1, N + 1):
        result += a[n] * np.cos(n * X_GRID) + b[n] * np.sin(n * X_GRID)
    return result


# ── Plot update ───────────────────────────────────────────────────────────────
def update_plots(N):
    """Update approximation, components, and bar chart for N terms."""
    a = state["a_coeffs"]
    b = state["b_coeffs"]
    f = state["f_interp"]
    if a is None:
        return

    N = int(N)

    # Approximation plot
    approx = fourier_sum(a, b, N)
    original_line.set_data(X_GRID, f)
    approx_line.set_data(X_GRID, approx)
    ax_approx.set_title(f"Fourier Approximation (N={N})")

    # Auto-adjust y-limits with some padding
    all_vals = np.concatenate([f, approx])
    ymin, ymax = all_vals.min(), all_vals.max()
    margin = max(0.2, (ymax - ymin) * 0.1)
    ax_approx.set_ylim(ymin - margin, ymax + margin)

    # Components plot
    dc_val = a[0] / 2.0
    dc_line.set_data(X_GRID, np.full_like(X_GRID, dc_val))

    comp_ymin, comp_ymax = dc_val, dc_val
    for i, line in enumerate(component_lines):
        n = i + 1
        if n <= N:
            component = a[n] * np.cos(n * X_GRID) + b[n] * np.sin(n * X_GRID)
            line.set_data(X_GRID, component)
            comp_ymin = min(comp_ymin, component.min())
            comp_ymax = max(comp_ymax, component.max())
        else:
            line.set_data([], [])

    comp_margin = max(0.2, (comp_ymax - comp_ymin) * 0.1)
    ax_components.set_ylim(comp_ymin - comp_margin, comp_ymax + comp_margin)

    # Bar chart
    ax_bars.clear()
    ax_bars.set_title("Fourier Coefficients")
    ax_bars.set_xlabel("n")

    ns = np.arange(0, N + 1)
    width = 0.35
    ax_bars.bar(ns - width / 2, a[:N + 1], width, label="aₙ", color="steelblue")
    ax_bars.bar(ns + width / 2, b[:N + 1], width, label="bₙ", color="darkorange")
    ax_bars.legend(fontsize=8)
    ax_bars.axhline(0, color="black", linewidth=0.5)

    fig.canvas.draw_idle()


# ── Drawing event handlers ────────────────────────────────────────────────────
def on_press(event):
    if event.inaxes != ax_draw or event.xdata is None:
        return
    state["drawing"] = True
    state["xs"] = [event.xdata]
    state["ys"] = [np.clip(event.ydata, -2, 2)]
    draw_line.set_data(state["xs"], state["ys"])
    fig.canvas.draw_idle()


def on_motion(event):
    if not state["drawing"] or event.inaxes != ax_draw or event.xdata is None:
        return
    x = np.clip(event.xdata, -np.pi, np.pi)
    y = np.clip(event.ydata, -2, 2)
    state["xs"].append(x)
    state["ys"].append(y)
    draw_line.set_data(state["xs"], state["ys"])
    fig.canvas.draw_idle()


def on_release(event):
    if not state["drawing"]:
        return
    state["drawing"] = False
    _process_drawing()


def _process_drawing():
    """Sort, deduplicate, interpolate, compute coefficients, update plots."""
    xs = np.array(state["xs"])
    ys = np.array(state["ys"])

    if len(xs) < 2:
        return

    # Sort by x
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    # Deduplicate x-values (keep first occurrence after sort)
    unique_x, idx = np.unique(xs, return_index=True)
    unique_y = ys[idx]

    if len(unique_x) < 2:
        return

    # Interpolate onto uniform grid
    f_interp = np.interp(X_GRID, unique_x, unique_y)
    state["f_interp"] = f_interp

    # Compute Fourier coefficients
    a, b = compute_coefficients(f_interp)
    state["a_coeffs"] = a
    state["b_coeffs"] = b

    # Update all plots for current slider value
    update_plots(slider.val)


# ── Clear button handler ─────────────────────────────────────────────────────
def on_clear(event):
    state["drawing"] = False
    state["xs"] = []
    state["ys"] = []
    state["f_interp"] = None
    state["a_coeffs"] = None
    state["b_coeffs"] = None

    # Reset drawing canvas
    draw_line.set_data([], [])

    # Reset approximation plot
    original_line.set_data([], [])
    approx_line.set_data([], [])
    ax_approx.set_title("Fourier Approximation (N=1)")
    ax_approx.set_ylim(-2, 2)

    # Reset components
    dc_line.set_data([], [])
    for line in component_lines:
        line.set_data([], [])
    ax_components.set_ylim(-1, 1)

    # Reset bar chart
    ax_bars.clear()
    ax_bars.set_title("Fourier Coefficients")
    ax_bars.set_xlabel("n")

    fig.canvas.draw_idle()


# ── Connect events ────────────────────────────────────────────────────────────
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("button_release_event", on_release)
slider.on_changed(update_plots)
btn_clear.on_clicked(on_clear)

plt.show()

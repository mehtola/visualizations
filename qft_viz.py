"""
Quantum Fourier Transform Visualization Tool

Interactive tool for learning how the quantum Fourier transform maps basis
states and periodic superpositions into phase patterns. Step through a QFT
circuit, inspect amplitudes as complex phasors, and see why period-finding
creates sharp peaks in the Fourier basis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, RadioButtons, Slider


# -- Constants ----------------------------------------------------------------
MIN_QUBITS = 2
MAX_QUBITS = 6
MAX_STEPS = MAX_QUBITS + MAX_QUBITS * (MAX_QUBITS - 1) // 2 + MAX_QUBITS // 2
DEFAULT_QUBITS = 4

COLORS = {
    "bg": "#FAFAFA",
    "wire": "#616161",
    "past": "#90A4AE",
    "active": "#D84315",
    "future": "#D7D7D7",
    "bar": "#4C78A8",
    "bar_edge": "#FFFFFF",
    "target": "#E45756",
    "prob": "#54A24B",
    "phase": "#B279A2",
    "button": "lightgray",
    "button_hover": "#BBDEFB",
    "slider": "steelblue",
}


# -- State --------------------------------------------------------------------
state = {
    "mode": "basis",
    "n_qubits": DEFAULT_QUBITS,
    "basis_index": 3,
    "period": 4,
    "offset": 0,
    "step": MAX_STEPS,
    "sync_step_to_end": True,
    "updating_widgets": False,
}


# -- QFT math -----------------------------------------------------------------
def n_states():
    """Return the Hilbert-space size for the current qubit count."""
    return 2 ** state["n_qubits"]


def bit_string(index, width):
    """Return a fixed-width binary label."""
    return format(int(index), f"0{width}b")


def qft_gates(n):
    """Return the textbook QFT gate schedule with final swaps."""
    gates = []
    for target in range(n):
        gates.append({"kind": "H", "target": target})
        for control in range(target + 1, n):
            gates.append({
                "kind": "R",
                "target": target,
                "control": control,
                "power": control - target + 1,
            })
    for top in range(n // 2):
        gates.append({"kind": "SWAP", "a": top, "b": n - 1 - top})
    return gates


def basis_state(n, index):
    """Prepare |index>."""
    vec = np.zeros(2 ** n, dtype=complex)
    vec[int(index)] = 1.0
    return vec


def periodic_state(n, period, offset):
    """Prepare a uniform superposition over indices offset mod period."""
    size = 2 ** n
    period = max(1, int(period))
    offset = int(offset) % period
    indices = np.arange(offset, size, period)
    vec = np.zeros(size, dtype=complex)
    if len(indices) == 0:
        indices = np.array([offset % size])
    vec[indices] = 1.0 / np.sqrt(len(indices))
    return vec


def current_input_state():
    """Return the input state selected by the controls."""
    n = state["n_qubits"]
    if state["mode"] == "basis":
        return basis_state(n, state["basis_index"])
    return periodic_state(n, state["period"], state["offset"])


def apply_h(vec, n, qubit):
    """Apply a Hadamard gate to a big-endian qubit index."""
    out = vec.copy()
    mask = 1 << (n - 1 - qubit)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    for idx in range(len(vec)):
        if idx & mask:
            continue
        pair = idx | mask
        a0 = vec[idx]
        a1 = vec[pair]
        out[idx] = (a0 + a1) * inv_sqrt2
        out[pair] = (a0 - a1) * inv_sqrt2

    return out


def apply_controlled_phase(vec, n, control, target, power):
    """Apply controlled R_power with angle 2*pi/2**power."""
    out = vec.copy()
    control_mask = 1 << (n - 1 - control)
    target_mask = 1 << (n - 1 - target)
    angle = 2.0 * np.pi / (2 ** power)
    phase = np.exp(1j * angle)

    for idx in range(len(vec)):
        if idx & control_mask and idx & target_mask:
            out[idx] *= phase

    return out


def apply_swap(vec, n, qubit_a, qubit_b):
    """Swap two big-endian qubits."""
    out = np.empty_like(vec)
    mask_a = 1 << (n - 1 - qubit_a)
    mask_b = 1 << (n - 1 - qubit_b)

    for idx, amp in enumerate(vec):
        bit_a = bool(idx & mask_a)
        bit_b = bool(idx & mask_b)
        dest = idx
        if bit_a != bit_b:
            dest ^= mask_a | mask_b
        out[dest] = amp

    return out


def apply_gate(vec, n, gate):
    """Apply one gate descriptor."""
    kind = gate["kind"]
    if kind == "H":
        return apply_h(vec, n, gate["target"])
    if kind == "R":
        return apply_controlled_phase(
            vec, n, gate["control"], gate["target"], gate["power"]
        )
    if kind == "SWAP":
        return apply_swap(vec, n, gate["a"], gate["b"])
    return vec


def state_after_step(input_vec, n, step):
    """Apply the first step gates from the QFT schedule."""
    gates = qft_gates(n)
    vec = input_vec.copy()
    for gate in gates[:step]:
        vec = apply_gate(vec, n, gate)
    return vec


def qft_matrix_state(input_vec):
    """Apply the mathematical QFT convention F[y,x] = exp(2*pi*i*x*y/N)."""
    size = len(input_vec)
    x = np.arange(size)
    y = x[:, None]
    matrix = np.exp(2j * np.pi * y * x / size) / np.sqrt(size)
    return matrix @ input_vec


def probabilities(vec):
    """Return numerical probabilities with small noise removed."""
    probs = np.abs(vec) ** 2
    probs[probs < 1e-12] = 0.0
    return probs


def phase_colors(vec):
    """Color amplitudes by complex phase, fading near-zero entries."""
    mags = np.abs(vec)
    phases = (np.angle(vec) + np.pi) / (2 * np.pi)
    colors = plt.cm.twilight(phases)
    colors[:, 3] = np.clip(0.25 + mags / max(mags.max(), 1e-12), 0.25, 1.0)
    colors[mags < 1e-8] = (0.86, 0.86, 0.86, 0.35)
    return colors


def dominant_terms(vec, count=5):
    """Return the largest probability terms as formatted labels."""
    probs = probabilities(vec)
    n = state["n_qubits"]
    order = np.argsort(probs)[::-1]
    lines = []
    for idx in order[:count]:
        if probs[idx] <= 1e-10:
            continue
        phase = np.angle(vec[idx])
        lines.append(f"|{bit_string(idx, n)}>  p={probs[idx]:.3f}  angle={phase:+.2f}")
    return lines


def current_gate_description(step, gates):
    """Return a compact label for the selected circuit step."""
    if step <= 0:
        return "Step 0: prepare the selected input state."
    if step > len(gates):
        step = len(gates)

    gate = gates[step - 1]
    kind = gate["kind"]
    if kind == "H":
        return f"Step {step}: H on q{gate['target']} creates local superposition."
    if kind == "R":
        return (
            f"Step {step}: controlled R{gate['power']} from q{gate['control']} "
            f"to q{gate['target']} adds a conditional phase."
        )
    return f"Step {step}: swap q{gate['a']} with q{gate['b']} to restore bit order."


# -- Drawing helpers ----------------------------------------------------------
def draw_register_bars(ax, vec, title):
    """Draw probability bars for a state vector."""
    ax.clear()
    probs = probabilities(vec)
    n = state["n_qubits"]
    size = len(vec)
    x = np.arange(size)

    ax.bar(x, probs, color=phase_colors(vec), edgecolor=COLORS["bar_edge"],
           linewidth=0.6)
    ax.set_ylim(0, min(1.05, max(0.25, probs.max() * 1.25)))
    ax.set_xlim(-0.6, size - 0.4)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Probability", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(labelsize=7)

    if size <= 16:
        ax.set_xticks(x)
        ax.set_xticklabels([bit_string(i, n) for i in x], rotation=45,
                           ha="right", fontsize=7)
    else:
        ticks = np.linspace(0, size - 1, min(9, size), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(i) for i in ticks], fontsize=7)


def draw_qft_circuit(ax, gates, active_step):
    """Draw a QFT circuit and highlight progress."""
    ax.clear()
    n = state["n_qubits"]
    gate_count = len(gates)
    ax.set_title("QFT circuit", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.8, max(5.0, gate_count + 0.9))
    ax.set_ylim(-0.8, n - 0.2)
    ax.axis("off")

    y_positions = np.arange(n)[::-1]
    for q, y in enumerate(y_positions):
        ax.plot([-0.5, gate_count + 0.4], [y, y], color=COLORS["wire"],
                linewidth=1.2)
        ax.text(-0.65, y, f"q{q}", va="center", ha="right", fontsize=8,
                fontfamily="monospace")

    for idx, gate in enumerate(gates, start=1):
        if idx < active_step:
            color = COLORS["past"]
            alpha = 0.75
            linewidth = 1.0
        elif idx == active_step:
            color = COLORS["active"]
            alpha = 1.0
            linewidth = 2.2
        else:
            color = COLORS["future"]
            alpha = 0.9
            linewidth = 1.0

        x = idx - 0.2
        kind = gate["kind"]
        if kind == "H":
            y = y_positions[gate["target"]]
            rect = Rectangle((x - 0.23, y - 0.23), 0.46, 0.46,
                             facecolor="white", edgecolor=color,
                             linewidth=linewidth, alpha=alpha, zorder=3)
            ax.add_patch(rect)
            ax.text(x, y, "H", ha="center", va="center", fontsize=8,
                    fontweight="bold", color=color, zorder=4)
        elif kind == "R":
            y_control = y_positions[gate["control"]]
            y_target = y_positions[gate["target"]]
            ax.plot([x, x], [y_control, y_target], color=color,
                    linewidth=linewidth, alpha=alpha, zorder=2)
            ax.add_patch(Circle((x, y_control), 0.08, facecolor=color,
                                edgecolor=color, alpha=alpha, zorder=4))
            rect = Rectangle((x - 0.25, y_target - 0.18), 0.50, 0.36,
                             facecolor="white", edgecolor=color,
                             linewidth=linewidth, alpha=alpha, zorder=3)
            ax.add_patch(rect)
            ax.text(x, y_target, f"R{gate['power']}", ha="center",
                    va="center", fontsize=7, color=color, zorder=4)
        else:
            y_a = y_positions[gate["a"]]
            y_b = y_positions[gate["b"]]
            ax.plot([x, x], [y_a, y_b], color=color, linewidth=linewidth,
                    alpha=alpha, zorder=2)
            for y in (y_a, y_b):
                ax.plot([x - 0.10, x + 0.10], [y - 0.10, y + 0.10],
                        color=color, linewidth=linewidth, alpha=alpha,
                        zorder=4)
                ax.plot([x - 0.10, x + 0.10], [y + 0.10, y - 0.10],
                        color=color, linewidth=linewidth, alpha=alpha,
                        zorder=4)

    ax.text(0.0, -0.55, current_gate_description(active_step, gates),
            fontsize=8, color="#424242", ha="left", va="center",
            transform=ax.transData)


def draw_phasors(ax, vec):
    """Draw selected amplitudes as complex phasors."""
    ax.clear()
    ax.set_title("Complex amplitudes", fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="#BDBDBD", linewidth=0.8)
    ax.axvline(0, color="#BDBDBD", linewidth=0.8)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="#D0D0D0",
                        linewidth=0.8, linestyle="--"))
    ax.set_xticks([])
    ax.set_yticks([])

    mags = np.abs(vec)
    if mags.max() <= 1e-12:
        return

    scale = 0.95 / mags.max()
    colors = phase_colors(vec)
    n = state["n_qubits"]
    shown = np.where(mags > max(1e-8, mags.max() * 0.08))[0]
    if len(shown) > 24:
        shown = np.argsort(mags)[-24:]

    for idx in shown:
        amp = vec[idx] * scale
        ax.arrow(0, 0, amp.real, amp.imag, color=colors[idx],
                 width=0.006, head_width=0.045, length_includes_head=True,
                 alpha=0.9)
        ax.text(amp.real * 1.08, amp.imag * 1.08, bit_string(idx, n),
                fontsize=6, ha="center", va="center")

    ax.text(0.02, 0.98, "angle = phase\nlength = magnitude",
            transform=ax.transAxes, fontsize=7, color="#616161",
            ha="left", va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75})


def draw_phase_ramp(ax, final_vec):
    """Draw phase ramp information for the selected QFT output."""
    ax.clear()
    ax.set_title("Phase pattern", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    size = n_states()
    y = 0.92

    if state["mode"] == "basis":
        k = state["basis_index"]
        delta = 2 * np.pi * k / size
        ax.text(0.04, y, f"Input basis state: |{bit_string(k, state['n_qubits'])}>",
                fontsize=9, fontweight="bold", transform=ax.transAxes)
        y -= 0.12
        ax.text(0.04, y, "QFT makes all output probabilities equal.",
                fontsize=8, transform=ax.transAxes)
        y -= 0.10
        ax.text(0.04, y, f"Neighboring amplitudes rotate by {delta:.3f} rad.",
                fontsize=8, transform=ax.transAxes)
        y -= 0.13

        phases = np.angle(final_vec)
        max_rows = min(size, 8)
        for idx in range(max_rows):
            ax.text(0.08, y, f"|{bit_string(idx, state['n_qubits'])}>",
                    fontsize=7, fontfamily="monospace", transform=ax.transAxes)
            ax.plot([0.35, 0.85], [y + 0.015, y + 0.015],
                    color="#EEEEEE", linewidth=4, transform=ax.transAxes)
            marker_x = 0.35 + 0.50 * ((phases[idx] + np.pi) / (2 * np.pi))
            ax.plot(marker_x, y + 0.015, "o", color=phase_colors(final_vec)[idx],
                    markersize=5, transform=ax.transAxes)
            ax.text(0.88, y, f"{phases[idx]:+.2f}",
                    fontsize=7, transform=ax.transAxes)
            y -= 0.075

        if size > max_rows:
            ax.text(0.08, y, "...", fontsize=8, transform=ax.transAxes)
    else:
        period = state["period"]
        spacing = size / period
        ax.text(0.04, y, f"Periodic input: x = {state['offset']} mod {period}",
                fontsize=9, fontweight="bold", transform=ax.transAxes)
        y -= 0.12
        ax.text(0.04, y, "QFT converts repetition in x into peaks in frequency y.",
                fontsize=8, transform=ax.transAxes)
        y -= 0.10
        if size % period == 0:
            ax.text(0.04, y, f"Peak spacing is N / r = {int(spacing)}.",
                    fontsize=8, transform=ax.transAxes)
        else:
            ax.text(0.04, y, f"N / r = {spacing:.2f}; peaks spread across bins.",
                    fontsize=8, transform=ax.transAxes)
        y -= 0.13
        ax.text(0.04, y, "Largest output terms:",
                fontsize=8, fontweight="bold", transform=ax.transAxes)
        y -= 0.09
        for line in dominant_terms(final_vec, count=6):
            ax.text(0.08, y, line, fontsize=7, fontfamily="monospace",
                    transform=ax.transAxes)
            y -= 0.075


def draw_formula_panel(ax, input_vec, final_vec, step, gate_count):
    """Draw compact teaching text for the current interaction state."""
    ax.clear()
    ax.set_title("What to notice", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.92
    size = len(input_vec)
    ax.text(0.04, y, "QFT formula",
            fontsize=9, fontweight="bold", transform=ax.transAxes)
    y -= 0.10
    ax.text(0.04, y, "F|x> = 1/sqrt(N) sum_y exp(2*pi*i*x*y/N)|y>",
            fontsize=7.5, fontfamily="monospace", transform=ax.transAxes)
    y -= 0.15

    if step == gate_count:
        ax.text(0.04, y, "The circuit has reached the full QFT output.",
                fontsize=8, transform=ax.transAxes)
    else:
        ax.text(0.04, y, "Move the circuit-step slider to watch amplitude",
                fontsize=8, transform=ax.transAxes)
        y -= 0.08
        ax.text(0.04, y, "move from computational basis to phase basis.",
                fontsize=8, transform=ax.transAxes)
    y -= 0.14

    input_support = np.where(probabilities(input_vec) > 1e-10)[0]
    ax.text(0.04, y, f"N = {size}, nonzero input states = {len(input_support)}",
            fontsize=8, transform=ax.transAxes)
    y -= 0.10

    if state["mode"] == "basis":
        ax.text(0.04, y, "Basis inputs do not create probability peaks;",
                fontsize=8, transform=ax.transAxes)
        y -= 0.08
        ax.text(0.04, y, "they create a measurable phase gradient.",
                fontsize=8, transform=ax.transAxes)
    else:
        ax.text(0.04, y, "Periodic inputs create constructive interference",
                fontsize=8, transform=ax.transAxes)
        y -= 0.08
        ax.text(0.04, y, "at frequencies matching the hidden period.",
                fontsize=8, transform=ax.transAxes)


def draw_lesson_panel(ax):
    """Draw a compact legend and mode description."""
    ax.clear()
    ax.set_title("Lesson mode", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.90
    if state["mode"] == "basis":
        lines = [
            "Basis state mode",
            "Choose an integer x.",
            "The QFT output has flat probabilities.",
            "The useful information is in the phases.",
        ]
    else:
        lines = [
            "Periodic input mode",
            "Choose states spaced by period r.",
            "The QFT output shows frequency peaks.",
            "This is the core pattern behind period finding.",
        ]

    ax.text(0.04, y, lines[0], fontsize=9, fontweight="bold",
            transform=ax.transAxes)
    y -= 0.13
    for line in lines[1:]:
        ax.text(0.06, y, line, fontsize=8, transform=ax.transAxes)
        y -= 0.10

    y -= 0.04
    legend_items = [
        ("Past gate", COLORS["past"]),
        ("Active gate", COLORS["active"]),
        ("Future gate", COLORS["future"]),
    ]
    for label, color in legend_items:
        ax.plot([0.07, 0.18], [y, y], color=color, linewidth=3,
                transform=ax.transAxes)
        ax.text(0.23, y, label, fontsize=8, va="center",
                transform=ax.transAxes)
        y -= 0.10


def redraw():
    """Redraw all panels from current state."""
    n = state["n_qubits"]
    gates = qft_gates(n)
    gate_count = len(gates)
    step = min(int(state["step"]), gate_count)

    input_vec = current_input_state()
    current_vec = state_after_step(input_vec, n, step)
    final_vec = qft_matrix_state(input_vec)

    draw_register_bars(ax_input, input_vec, "Input register")
    draw_qft_circuit(ax_circuit, gates, step)
    draw_register_bars(ax_output, current_vec,
                       f"State after step {step} of {gate_count}")

    if state["mode"] == "periodic":
        size = len(final_vec)
        period = max(1, state["period"])
        for m in range(period):
            target = m * size / period
            if 0 <= target < size:
                ax_output.axvline(target, color=COLORS["target"],
                                  linestyle="--", linewidth=1.0, alpha=0.65)

    draw_phasors(ax_phasor, current_vec)
    draw_lesson_panel(ax_lesson)
    draw_phase_ramp(ax_phase, final_vec)
    draw_formula_panel(ax_formula, input_vec, final_vec, step, gate_count)
    fig.canvas.draw_idle()


# -- Widget plumbing ----------------------------------------------------------
def configure_slider(slider, minimum, maximum, value):
    """Update a Slider's range and value without unnecessary redraws."""
    slider.valmin = minimum
    slider.valmax = maximum
    axis_max = maximum if maximum > minimum else minimum + 1
    slider.ax.set_xlim(minimum, axis_max)
    value = int(np.clip(value, minimum, maximum))
    if int(slider.val) != value:
        slider.set_val(value)
    else:
        slider.valtext.set_text(str(value))


def configure_widgets():
    """Clamp controls to valid ranges for the selected qubit count and mode."""
    n = state["n_qubits"]
    size = 2 ** n
    gates = qft_gates(n)
    gate_count = len(gates)

    state["basis_index"] = int(np.clip(state["basis_index"], 0, size - 1))
    state["period"] = int(np.clip(state["period"], 1, size))
    state["offset"] = int(np.clip(state["offset"], 0, state["period"] - 1))
    if state["sync_step_to_end"]:
        state["step"] = gate_count
    else:
        state["step"] = int(np.clip(state["step"], 0, gate_count))

    state["updating_widgets"] = True
    configure_slider(slider_basis, 0, size - 1, state["basis_index"])
    configure_slider(slider_period, 1, size, state["period"])
    configure_slider(slider_offset, 0, max(0, state["period"] - 1),
                     state["offset"])
    configure_slider(slider_step, 0, gate_count, state["step"])
    slider_step.valtext.set_text(f"{state['step']}/{gate_count}")
    state["updating_widgets"] = False

    show_basis = state["mode"] == "basis"
    ax_basis_slider.set_visible(show_basis)
    ax_period_slider.set_visible(not show_basis)
    ax_offset_slider.set_visible(not show_basis)


def on_mode_change(label):
    state["mode"] = "basis" if label == "Basis" else "periodic"
    configure_widgets()
    redraw()


def on_qubit_change(val):
    if state["updating_widgets"]:
        return
    state["n_qubits"] = int(val)
    configure_widgets()
    redraw()


def on_basis_change(val):
    if state["updating_widgets"]:
        return
    state["basis_index"] = int(val)
    redraw()


def on_period_change(val):
    if state["updating_widgets"]:
        return
    state["period"] = int(val)
    configure_widgets()
    redraw()


def on_offset_change(val):
    if state["updating_widgets"]:
        return
    state["offset"] = int(val)
    configure_widgets()
    redraw()


def on_step_change(val):
    if state["updating_widgets"]:
        return
    gates = qft_gates(state["n_qubits"])
    gate_count = len(gates)
    state["step"] = int(np.clip(val, 0, gate_count))
    state["sync_step_to_end"] = state["step"] == gate_count
    slider_step.valtext.set_text(f"{state['step']}/{gate_count}")
    redraw()


def on_reset(event):
    state["mode"] = "basis"
    state["n_qubits"] = DEFAULT_QUBITS
    state["basis_index"] = 3
    state["period"] = 4
    state["offset"] = 0
    state["sync_step_to_end"] = True
    state["step"] = len(qft_gates(DEFAULT_QUBITS))

    state["updating_widgets"] = True
    radio_mode.set_active(0)
    slider_qubits.set_val(DEFAULT_QUBITS)
    state["updating_widgets"] = False

    configure_widgets()
    redraw()


def main():
    """Start the interactive QFT visualizer."""
    global fig, ax_input, ax_circuit, ax_output, ax_phasor
    global ax_lesson, ax_phase, ax_formula
    global ax_mode_radio, radio_mode
    global ax_qubits_slider, slider_qubits
    global ax_basis_slider, slider_basis
    global ax_period_slider, slider_period
    global ax_offset_slider, slider_offset
    global ax_step_slider, slider_step
    global ax_reset_button, btn_reset

    fig = plt.figure(figsize=(16, 10))
    fig.canvas.manager.set_window_title("Quantum Fourier Transform Visualizer")
    fig.patch.set_facecolor(COLORS["bg"])

    gs = GridSpec(
        3, 3,
        figure=fig,
        height_ratios=[2.3, 3.2, 2.0],
        width_ratios=[3.0, 4.2, 3.0],
        left=0.05,
        right=0.97,
        top=0.95,
        bottom=0.15,
        hspace=0.40,
        wspace=0.32,
    )

    ax_input = fig.add_subplot(gs[0, 0])
    ax_circuit = fig.add_subplot(gs[0, 1:])
    ax_output = fig.add_subplot(gs[1, 0:2])
    ax_phasor = fig.add_subplot(gs[1, 2])
    ax_lesson = fig.add_subplot(gs[2, 0])
    ax_phase = fig.add_subplot(gs[2, 1])
    ax_formula = fig.add_subplot(gs[2, 2])

    # Widgets
    ax_mode_radio = fig.add_axes([0.04, 0.035, 0.10, 0.075])
    radio_mode = RadioButtons(ax_mode_radio, ("Basis", "Periodic"), active=0)
    for label in radio_mode.labels:
        label.set_fontsize(8)

    ax_qubits_slider = fig.add_axes([0.19, 0.090, 0.16, 0.025])
    slider_qubits = Slider(ax_qubits_slider, "Qubits", MIN_QUBITS, MAX_QUBITS,
                           valinit=DEFAULT_QUBITS, valstep=1,
                           color=COLORS["slider"])

    ax_basis_slider = fig.add_axes([0.19, 0.045, 0.16, 0.025])
    slider_basis = Slider(ax_basis_slider, "Input x", 0, n_states() - 1,
                          valinit=state["basis_index"], valstep=1,
                          color=COLORS["slider"])

    ax_period_slider = fig.add_axes([0.42, 0.090, 0.16, 0.025])
    slider_period = Slider(ax_period_slider, "Period r", 1, n_states(),
                           valinit=state["period"], valstep=1,
                           color=COLORS["slider"])

    ax_offset_slider = fig.add_axes([0.42, 0.045, 0.16, 0.025])
    slider_offset = Slider(ax_offset_slider, "Offset", 0, state["period"] - 1,
                           valinit=state["offset"], valstep=1,
                           color=COLORS["slider"])

    ax_step_slider = fig.add_axes([0.65, 0.090, 0.22, 0.025])
    slider_step = Slider(ax_step_slider, "Circuit step", 0, MAX_STEPS,
                         valinit=len(qft_gates(DEFAULT_QUBITS)), valstep=1,
                         color="#D84315")

    ax_reset_button = fig.add_axes([0.79, 0.035, 0.08, 0.045])
    btn_reset = Button(ax_reset_button, "Reset", color=COLORS["button"],
                       hovercolor=COLORS["button_hover"])

    radio_mode.on_clicked(on_mode_change)
    slider_qubits.on_changed(on_qubit_change)
    slider_basis.on_changed(on_basis_change)
    slider_period.on_changed(on_period_change)
    slider_offset.on_changed(on_offset_change)
    slider_step.on_changed(on_step_change)
    btn_reset.on_clicked(on_reset)

    configure_widgets()
    redraw()
    plt.show()


if __name__ == "__main__":
    main()

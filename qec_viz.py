"""
Quantum Error Correction Visualization Tool

Interactive tool to visualize how quantum error correction works, focusing on the
repetition code (1D) and surface code (2D). Watch errors inject, syndromes light
up, decoding pair syndromes, and corrections applied — step by step.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.animation import FuncAnimation
from math import comb

# ── Constants ────────────────────────────────────────────────────────────────
PHASE_NAMES = [
    "Initialize",
    "Error Injection",
    "Syndrome Extraction",
    "Decoding",
    "Correction",
]

PHASE_DESCRIPTIONS = [
    "All qubits initialized to |0⟩.\nNo errors present.",
    "Random X (bit-flip) errors applied\nwith probability p per qubit.",
    "Parity checks measured.\nTriggered syndromes indicate\nneighboring qubit disagreement.",
    "Decoder pairs triggered syndromes\nto infer error locations.\nPurple lines show pairings.",
    "Corrections applied to inferred\nerror locations. Logical error\nchecked via parity/majority vote.",
]

# Colors
C_DATA_OK = "#4CAF50"       # green
C_DATA_ERR = "#F44336"      # red
C_DATA_CORR = "#2196F3"     # blue
C_SYND_OFF = "#E0E0E0"      # gray
C_SYND_ON = "#FFC107"       # amber
C_STAB_OFF = "#E3F2FD"      # light blue
C_STAB_ON = "#FF9800"       # orange
C_DECODER = "#9C27B0"       # purple
C_EDGE = "#757575"          # gray for edges
C_BG = "#FAFAFA"            # background

ANIM_INTERVAL_MS = 800

rng = np.random.default_rng()


# ── State ────────────────────────────────────────────────────────────────────
state = {
    "mode": "repetition",   # "repetition" or "surface"
    "distance": 5,
    "error_rate": 0.15,
    "phase": 0,
    "playing": False,
    # Repetition code arrays
    "rep_data": None,       # shape (d,) — 0 or 1 (error flag)
    "rep_syndromes": None,  # shape (d-1,)
    "rep_corrections": None,  # shape (d,) — proposed corrections
    "rep_pairings": [],     # list of (i, j) syndrome index pairs
    # Surface code arrays
    "surf_data": None,      # shape (d, d)
    "surf_syndromes": None, # shape (d-1, d-1)
    "surf_corrections": None,  # shape (d, d)
    "surf_pairings": [],    # list of ((r1,c1), (r2,c2)) pairs
    # Statistics
    "rounds": 0,
    "logical_errors": 0,
    "last_logical_error": False,
    # Per-distance stats for threshold plot
    "stats_by_d": {},       # {d: {"rounds": int, "errors": int}}
}


# ── Repetition Code Simulation ───────────────────────────────────────────────
def rep_initialize():
    d = state["distance"]
    state["rep_data"] = np.zeros(d, dtype=int)
    state["rep_syndromes"] = np.zeros(d - 1, dtype=int)
    state["rep_corrections"] = np.zeros(d, dtype=int)
    state["rep_pairings"] = []
    state["last_logical_error"] = False


def rep_inject_errors():
    d = state["distance"]
    p = state["error_rate"]
    state["rep_data"] = (rng.random(d) < p).astype(int)
    state["rep_syndromes"] = np.zeros(d - 1, dtype=int)
    state["rep_corrections"] = np.zeros(d, dtype=int)
    state["rep_pairings"] = []


def rep_measure_syndromes():
    data = state["rep_data"]
    d = state["distance"]
    syndromes = np.zeros(d - 1, dtype=int)
    for i in range(d - 1):
        syndromes[i] = data[i] ^ data[i + 1]
    state["rep_syndromes"] = syndromes


def rep_decode():
    """Greedy nearest-pair matching of triggered syndromes."""
    syndromes = state["rep_syndromes"]
    triggered = list(np.where(syndromes == 1)[0])
    pairings = []
    corrections = np.zeros(state["distance"], dtype=int)

    while len(triggered) >= 2:
        # Pair the closest two
        best_dist = float("inf")
        best_pair = (0, 1)
        for i in range(len(triggered) - 1):
            dist = triggered[i + 1] - triggered[i]
            if dist < best_dist:
                best_dist = dist
                best_pair = (i, i + 1)

        s1 = triggered[best_pair[0]]
        s2 = triggered[best_pair[1]]
        pairings.append((s1, s2))

        # Correct the data qubits between the paired syndromes
        for q in range(s1 + 1, s2 + 1):
            corrections[q] ^= 1

        triggered.pop(best_pair[1])
        triggered.pop(best_pair[0])

    # Unpaired syndrome at boundary — correct to nearest boundary
    if len(triggered) == 1:
        s = triggered[0]
        # Pair to nearest boundary (virtual syndrome at -1 or d-1)
        if s < state["distance"] - 1 - s:
            # Closer to left boundary
            pairings.append((-1, s))
            for q in range(0, s + 1):
                corrections[q] ^= 1
        else:
            # Closer to right boundary
            pairings.append((s, state["distance"] - 1))
            for q in range(s + 1, state["distance"]):
                corrections[q] ^= 1

    state["rep_pairings"] = pairings
    state["rep_corrections"] = corrections


def rep_apply_correction():
    """Apply corrections and check logical error."""
    data = state["rep_data"]
    corrections = state["rep_corrections"]
    corrected = data ^ corrections
    # Logical error = majority of qubits still flipped
    logical_error = np.sum(corrected) > state["distance"] // 2
    state["last_logical_error"] = logical_error
    state["rounds"] += 1
    if logical_error:
        state["logical_errors"] += 1
    # Track per-distance stats
    d = state["distance"]
    if d not in state["stats_by_d"]:
        state["stats_by_d"][d] = {"rounds": 0, "errors": 0}
    state["stats_by_d"][d]["rounds"] += 1
    if logical_error:
        state["stats_by_d"][d]["errors"] += 1


# ── Surface Code Simulation ─────────────────────────────────────────────────
def surf_initialize():
    d = state["distance"]
    state["surf_data"] = np.zeros((d, d), dtype=int)
    state["surf_syndromes"] = np.zeros((d - 1, d - 1), dtype=int)
    state["surf_corrections"] = np.zeros((d, d), dtype=int)
    state["surf_pairings"] = []
    state["last_logical_error"] = False


def surf_inject_errors():
    d = state["distance"]
    p = state["error_rate"]
    state["surf_data"] = (rng.random((d, d)) < p).astype(int)
    state["surf_syndromes"] = np.zeros((d - 1, d - 1), dtype=int)
    state["surf_corrections"] = np.zeros((d, d), dtype=int)
    state["surf_pairings"] = []


def surf_measure_z_syndromes():
    """Each Z-stabilizer plaquette checks 4 surrounding data qubits."""
    d = state["distance"]
    data = state["surf_data"]
    syndromes = np.zeros((d - 1, d - 1), dtype=int)
    for r in range(d - 1):
        for c in range(d - 1):
            syndromes[r, c] = (
                data[r, c] ^ data[r, c + 1] ^
                data[r + 1, c] ^ data[r + 1, c + 1]
            )
    state["surf_syndromes"] = syndromes


def surf_decode_greedy():
    """Greedy nearest-neighbor matching for surface code syndromes."""
    d = state["distance"]
    syndromes = state["surf_syndromes"]
    triggered = []
    for r in range(d - 1):
        for c in range(d - 1):
            if syndromes[r, c]:
                triggered.append((r, c))

    pairings = []
    corrections = np.zeros((d, d), dtype=int)
    used = set()

    # Sort triggered syndromes and greedily pair nearest neighbors
    remaining = list(triggered)
    while len(remaining) >= 2:
        best_dist = float("inf")
        best_i, best_j = 0, 1
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                dist = (abs(remaining[i][0] - remaining[j][0]) +
                        abs(remaining[i][1] - remaining[j][1]))
                if dist < best_dist:
                    best_dist = dist
                    best_i, best_j = i, j

        s1 = remaining[best_i]
        s2 = remaining[best_j]
        pairings.append((s1, s2))

        # Correct along a path between the two syndromes
        # Walk from s1 to s2 along rows first, then columns
        r, c = s1
        target_r, target_c = s2
        # Walk rows
        dr = 1 if target_r > r else -1
        while r != target_r:
            corrections[r + (1 if dr > 0 else 0), c + 1] ^= 1
            r += dr
        # Walk columns
        dc = 1 if target_c > c else -1
        while c != target_c:
            corrections[r + 1, c + (1 if dc > 0 else 0)] ^= 1
            c += dc

        remaining.pop(best_j)
        remaining.pop(best_i)

    # Unpaired syndrome — correct to nearest boundary
    if len(remaining) == 1:
        s = remaining[0]
        r, c = s
        # Find nearest boundary distance
        dists = [r, c, d - 2 - r, d - 2 - c]
        min_dir = np.argmin(dists)
        if min_dir == 0:
            # Top boundary
            pairings.append(((-1, c), s))
            for rr in range(0, r + 1):
                corrections[rr, c + 1] ^= 1
        elif min_dir == 1:
            # Left boundary
            pairings.append(((r, -1), s))
            for cc in range(0, c + 1):
                corrections[r + 1, cc] ^= 1
        elif min_dir == 2:
            # Bottom boundary
            pairings.append((s, (d - 1, c)))
            for rr in range(r + 1, d):
                corrections[rr, c + 1] ^= 1
        else:
            # Right boundary
            pairings.append((s, (r, d - 1)))
            for cc in range(c + 1, d):
                corrections[r + 1, cc] ^= 1

    state["surf_pairings"] = pairings
    state["surf_corrections"] = corrections


def surf_apply_correction():
    """Apply corrections and check logical error (parity across top row)."""
    data = state["surf_data"]
    corrections = state["surf_corrections"]
    corrected = data ^ corrections
    # Logical X error = odd parity across any row (use first row)
    logical_error = bool(np.sum(corrected[0, :]) % 2)
    state["last_logical_error"] = logical_error
    state["rounds"] += 1
    if logical_error:
        state["logical_errors"] += 1
    d = state["distance"]
    if d not in state["stats_by_d"]:
        state["stats_by_d"][d] = {"rounds": 0, "errors": 0}
    state["stats_by_d"][d]["rounds"] += 1
    if logical_error:
        state["stats_by_d"][d]["errors"] += 1


# ── Phase Logic ──────────────────────────────────────────────────────────────
def advance_phase():
    """Advance to the next phase in the 5-phase cycle."""
    phase = state["phase"]

    if state["mode"] == "repetition":
        if phase == 0:
            rep_initialize()
        elif phase == 1:
            rep_inject_errors()
        elif phase == 2:
            rep_measure_syndromes()
        elif phase == 3:
            rep_decode()
        elif phase == 4:
            rep_apply_correction()
    else:
        if phase == 0:
            surf_initialize()
        elif phase == 1:
            surf_inject_errors()
        elif phase == 2:
            surf_measure_z_syndromes()
        elif phase == 3:
            surf_decode_greedy()
        elif phase == 4:
            surf_apply_correction()

    redraw()

    # Advance phase counter (wrap after correction)
    if phase == 4:
        state["phase"] = 1  # skip re-initialize, go straight to next error round
    else:
        state["phase"] = phase + 1


# ── Drawing Functions ────────────────────────────────────────────────────────
def draw_repetition(ax):
    """Draw the repetition code chain on the given axes."""
    ax.clear()
    d = state["distance"]
    phase = state["phase"]
    data = state["rep_data"]
    syndromes = state["rep_syndromes"]
    corrections = state["rep_corrections"]
    pairings = state["rep_pairings"]

    ax.set_xlim(-1.5, d + 0.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_title(f"Repetition Code  (d={d})", fontsize=13, fontweight="bold")
    ax.axis("off")

    y_data = 0.0
    y_synd = -0.7
    qubit_r = 0.35
    synd_r = 0.22

    # Draw edges between data qubits
    for i in range(d - 1):
        ax.plot([i, i + 1], [y_data, y_data], color=C_EDGE, linewidth=1.5,
                zorder=1)

    # Draw data qubits
    if data is not None:
        for i in range(d):
            if phase == 4 and corrections is not None and corrections[i]:
                color = C_DATA_CORR
            elif data[i]:
                color = C_DATA_ERR
            else:
                color = C_DATA_OK
            circle = plt.Circle((i, y_data), qubit_r, facecolor=color,
                                edgecolor="black", linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            ax.text(i, y_data, f"q{i}", ha="center", va="center",
                    fontsize=8, fontweight="bold", zorder=4)

    # Draw syndrome qubits
    if syndromes is not None and phase >= 2:
        for i in range(d - 1):
            color = C_SYND_ON if syndromes[i] else C_SYND_OFF
            circle = plt.Circle((i + 0.5, y_synd), synd_r, facecolor=color,
                                edgecolor="black", linewidth=1, zorder=3)
            ax.add_patch(circle)
            ax.text(i + 0.5, y_synd, f"s{i}", ha="center", va="center",
                    fontsize=6, zorder=4)

    # Draw decoder pairings as arcs below the chain
    if phase >= 3 and pairings:
        for s1, s2 in pairings:
            # Convert syndrome indices to x positions
            x1 = s1 + 0.5 if s1 >= 0 else -0.5
            x2 = s2 + 0.5 if s2 < d - 1 else d - 0.5
            mid = (x1 + x2) / 2
            arc_depth = -1.5 - 0.3 * abs(s2 - s1)
            # Draw arc as a curve
            t = np.linspace(0, 1, 30)
            arc_x = x1 + (x2 - x1) * t
            arc_y = y_synd + (arc_depth - y_synd) * 4 * t * (1 - t)
            ax.plot(arc_x, arc_y, color=C_DECODER, linewidth=2,
                    linestyle="--", zorder=2)
            # Mark endpoints
            ax.plot([x1, x2], [y_synd, y_synd], "o", color=C_DECODER,
                    markersize=5, zorder=5)

    # Show correction markers
    if phase == 4 and corrections is not None:
        for i in range(d):
            if corrections[i]:
                ax.annotate("X", (i, y_data + qubit_r + 0.25),
                            ha="center", va="center", fontsize=10,
                            fontweight="bold", color=C_DATA_CORR)


def draw_surface(ax):
    """Draw the surface code grid on the given axes."""
    ax.clear()
    d = state["distance"]
    phase = state["phase"]
    data = state["surf_data"]
    syndromes = state["surf_syndromes"]
    corrections = state["surf_corrections"]
    pairings = state["surf_pairings"]

    # Scale sizes with distance
    scale = max(0.5, 3.0 / d)
    qubit_r = 0.25 * scale
    font_data = max(5, int(9 * scale))
    font_stab = max(4, int(7 * scale))

    margin = 1.5
    ax.set_xlim(-margin, d - 1 + margin)
    ax.set_ylim(-margin, d - 1 + margin)
    ax.set_aspect("equal")
    ax.set_title(f"Surface Code  (d={d})", fontsize=13, fontweight="bold")
    ax.axis("off")

    # Draw Z-stabilizer plaquettes (shaded squares on dual lattice)
    if syndromes is not None and phase >= 2:
        for r in range(d - 1):
            for c in range(d - 1):
                color = C_STAB_ON if syndromes[r, c] else C_STAB_OFF
                rect = Rectangle((c - 0.05, r - 0.05), 1.1, 1.1,
                                  facecolor=color, edgecolor="gray",
                                  linewidth=0.5, alpha=0.5, zorder=1)
                ax.add_patch(rect)
    elif phase < 2:
        for r in range(d - 1):
            for c in range(d - 1):
                rect = Rectangle((c - 0.05, r - 0.05), 1.1, 1.1,
                                  facecolor=C_STAB_OFF, edgecolor="gray",
                                  linewidth=0.5, alpha=0.3, zorder=1)
                ax.add_patch(rect)

    # Draw edges between neighboring data qubits
    for r in range(d):
        for c in range(d):
            if c < d - 1:
                ax.plot([c, c + 1], [r, r], color=C_EDGE, linewidth=1,
                        zorder=1)
            if r < d - 1:
                ax.plot([c, c], [r, r + 1], color=C_EDGE, linewidth=1,
                        zorder=1)

    # Draw data qubits
    if data is not None:
        for r in range(d):
            for c in range(d):
                if phase == 4 and corrections is not None and corrections[r, c]:
                    color = C_DATA_CORR
                elif data[r, c]:
                    color = C_DATA_ERR
                else:
                    color = C_DATA_OK
                circle = plt.Circle((c, r), qubit_r, facecolor=color,
                                    edgecolor="black", linewidth=1, zorder=3)
                ax.add_patch(circle)
                if d <= 7:
                    ax.text(c, r, f"{r},{c}", ha="center", va="center",
                            fontsize=font_data, fontweight="bold", zorder=4)

    # Draw decoder pairings
    if phase >= 3 and pairings:
        for s1, s2 in pairings:
            # Syndrome positions are at (col+0.5, row+0.5) on the dual lattice
            # but for drawing, we draw a path between them on the main lattice
            if isinstance(s1, tuple) and isinstance(s2, tuple):
                r1, c1 = s1
                r2, c2 = s2
                # Handle boundary syndromes (negative indices mean boundary)
                x1 = c1 + 0.5 if c1 >= 0 else -0.5
                y1 = r1 + 0.5 if r1 >= 0 else -0.5
                x2 = c2 + 0.5 if c2 >= 0 else -0.5
                y2 = r2 + 0.5 if r2 >= 0 else -0.5
                if x2 > d - 1:
                    x2 = d - 0.5
                if y2 > d - 1:
                    y2 = d - 0.5
                ax.plot([x1, x2], [y1, y2], color=C_DECODER, linewidth=2.5,
                        linestyle="--", zorder=5)
                ax.plot([x1, x2], [y1, y2], "o", color=C_DECODER,
                        markersize=4 * scale, zorder=6)

    # Show correction markers
    if phase == 4 and corrections is not None:
        for r in range(d):
            for c in range(d):
                if corrections[r, c]:
                    ax.annotate("X", (c, r + qubit_r + 0.15),
                                ha="center", va="center",
                                fontsize=max(7, int(10 * scale)),
                                fontweight="bold", color=C_DATA_CORR,
                                zorder=7)


def draw_info_panel(ax):
    """Draw the information panel with phase info and statistics."""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    phase = state["phase"]
    mode_label = "Repetition Code" if state["mode"] == "repetition" else "Surface Code"

    # Phase indicator
    y = 0.95
    ax.text(0.05, y, f"Phase {phase}: {PHASE_NAMES[phase]}",
            fontsize=12, fontweight="bold", va="top",
            transform=ax.transAxes)

    y -= 0.12
    ax.text(0.05, y, PHASE_DESCRIPTIONS[phase],
            fontsize=9, va="top", family="monospace",
            transform=ax.transAxes, linespacing=1.4)

    # Syndrome readout
    y -= 0.32
    ax.plot([0.05, 0.95], [y, y], color="gray", linewidth=0.5,
            transform=ax.transAxes)
    y -= 0.04

    if state["mode"] == "repetition" and state["rep_syndromes"] is not None and phase >= 2:
        synd_str = "".join(str(s) for s in state["rep_syndromes"])
        ax.text(0.05, y, f"Syndromes: [{synd_str}]",
                fontsize=9, va="top", family="monospace",
                transform=ax.transAxes)
    elif state["mode"] == "surface" and state["surf_syndromes"] is not None and phase >= 2:
        n_triggered = int(np.sum(state["surf_syndromes"]))
        ax.text(0.05, y, f"Triggered stabilizers: {n_triggered}",
                fontsize=9, va="top", family="monospace",
                transform=ax.transAxes)
    else:
        ax.text(0.05, y, "Syndromes: —",
                fontsize=9, va="top", family="monospace",
                transform=ax.transAxes)

    # Logical error status
    y -= 0.10
    if phase == 4:
        if state["last_logical_error"]:
            ax.text(0.05, y, "Logical error: YES",
                    fontsize=10, va="top", fontweight="bold", color=C_DATA_ERR,
                    transform=ax.transAxes)
        else:
            ax.text(0.05, y, "Logical error: NO",
                    fontsize=10, va="top", fontweight="bold", color=C_DATA_OK,
                    transform=ax.transAxes)
    else:
        ax.text(0.05, y, "Logical error: —",
                fontsize=9, va="top", family="monospace", color="gray",
                transform=ax.transAxes)

    # Statistics
    y -= 0.12
    ax.plot([0.05, 0.95], [y, y], color="gray", linewidth=0.5,
            transform=ax.transAxes)
    y -= 0.04
    rounds = state["rounds"]
    errors = state["logical_errors"]
    rate = errors / rounds if rounds > 0 else 0
    ax.text(0.05, y, f"Rounds: {rounds}   Logical errors: {errors}",
            fontsize=9, va="top", family="monospace",
            transform=ax.transAxes)
    y -= 0.08
    ax.text(0.05, y, f"Logical error rate: {rate:.3f}",
            fontsize=9, va="top", family="monospace",
            transform=ax.transAxes)


def draw_threshold_plot(ax):
    """Draw the threshold plot: logical error rate vs physical error rate."""
    ax.clear()
    ax.set_title("Threshold Behavior", fontsize=10, fontweight="bold")
    ax.set_xlabel("Physical error rate p", fontsize=8)
    ax.set_ylabel("Logical error rate", fontsize=8)
    ax.tick_params(labelsize=7)

    p_range = np.linspace(0.001, 0.49, 100)

    # Theoretical curves for repetition code
    for d in [3, 5, 7, 9]:
        t = (d + 1) // 2
        # P_L ~ C(d, t) * p^t for repetition code
        p_logical = comb(d, t) * p_range**t
        p_logical = np.minimum(p_logical, 1.0)
        label = f"d={d}"
        ax.semilogy(p_range, p_logical, "-", linewidth=1.2, alpha=0.7,
                    label=label)

    # Overlay empirical data points
    for d, stats in state["stats_by_d"].items():
        if stats["rounds"] >= 5:
            emp_rate = max(stats["errors"] / stats["rounds"], 1e-4)
            ax.semilogy(state["error_rate"], emp_rate, "x", markersize=8,
                        markeredgewidth=2, label=f"d={d} emp.")

    ax.set_xlim(0, 0.5)
    ax.set_ylim(1e-4, 1.0)
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3, which="both")


# ── Master Redraw ────────────────────────────────────────────────────────────
def redraw():
    """Redraw all panels."""
    if state["mode"] == "repetition":
        draw_repetition(ax_main)
    else:
        draw_surface(ax_main)
    draw_info_panel(ax_info)
    draw_threshold_plot(ax_thresh)
    fig.canvas.draw_idle()


# ── Figure and Layout ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.canvas.manager.set_window_title("Quantum Error Correction Visualization")
fig.patch.set_facecolor(C_BG)

gs = GridSpec(3, 2, figure=fig, height_ratios=[4, 4, 0.6],
              width_ratios=[6.5, 3.5], hspace=0.25, wspace=0.25)

ax_main = fig.add_subplot(gs[:2, 0])     # main visualization (spans rows 0-1)
ax_info = fig.add_subplot(gs[0, 1])      # info panel
ax_thresh = fig.add_subplot(gs[1, 1])    # threshold plot

# ── Widgets ──────────────────────────────────────────────────────────────────
# Mode selector (RadioButtons)
ax_mode = fig.add_axes([0.04, 0.02, 0.10, 0.05])
radio_mode = RadioButtons(ax_mode, ["Repetition", "Surface"], active=0)
for label in radio_mode.labels:
    label.set_fontsize(8)

# Distance slider
ax_d_slider = fig.add_axes([0.22, 0.035, 0.18, 0.025])
slider_d = Slider(ax_d_slider, "d", 3, 9, valinit=5, valstep=2,
                  color="steelblue")

# Error rate slider
ax_p_slider = fig.add_axes([0.22, 0.01, 0.18, 0.025])
slider_p = Slider(ax_p_slider, "p", 0.0, 0.5, valinit=0.15,
                  color="darkorange")

# Step button
ax_step = fig.add_axes([0.48, 0.015, 0.08, 0.04])
btn_step = Button(ax_step, "Step", color="lightgray", hovercolor="#B0BEC5")

# Play/Pause button
ax_play = fig.add_axes([0.58, 0.015, 0.10, 0.04])
btn_play = Button(ax_play, "Play", color="lightgray", hovercolor="#A5D6A7")

# Reset button
ax_reset = fig.add_axes([0.70, 0.015, 0.08, 0.04])
btn_reset = Button(ax_reset, "Reset", color="lightgray", hovercolor="#EF9A9A")


# ── Event Handlers ───────────────────────────────────────────────────────────
def on_step(event):
    advance_phase()


def on_play(event):
    state["playing"] = not state["playing"]
    btn_play.label.set_text("Pause" if state["playing"] else "Play")
    fig.canvas.draw_idle()


def on_reset(event):
    state["playing"] = False
    state["phase"] = 0
    state["rounds"] = 0
    state["logical_errors"] = 0
    state["last_logical_error"] = False
    state["stats_by_d"] = {}
    btn_play.label.set_text("Play")
    if state["mode"] == "repetition":
        rep_initialize()
    else:
        surf_initialize()
    redraw()


def on_mode_change(label):
    state["mode"] = "repetition" if label == "Repetition" else "surface"
    state["phase"] = 0
    state["playing"] = False
    btn_play.label.set_text("Play")
    if state["mode"] == "repetition":
        rep_initialize()
    else:
        surf_initialize()
    redraw()


def on_distance_change(val):
    state["distance"] = int(val)
    state["phase"] = 0
    state["playing"] = False
    btn_play.label.set_text("Play")
    if state["mode"] == "repetition":
        rep_initialize()
    else:
        surf_initialize()
    redraw()


def on_error_rate_change(val):
    state["error_rate"] = val


# ── Connect Events ───────────────────────────────────────────────────────────
btn_step.on_clicked(on_step)
btn_play.on_clicked(on_play)
btn_reset.on_clicked(on_reset)
radio_mode.on_clicked(on_mode_change)
slider_d.on_changed(on_distance_change)
slider_p.on_changed(on_error_rate_change)


# ── Animation ────────────────────────────────────────────────────────────────
def animate(frame):
    if state["playing"]:
        advance_phase()


anim = FuncAnimation(fig, animate, interval=ANIM_INTERVAL_MS, cache_frame_data=False)

# ── Initial Draw ─────────────────────────────────────────────────────────────
rep_initialize()
redraw()

plt.show()

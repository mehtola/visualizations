"""
MRF Data Explorer

Interactive tool to explore 10 Markov Random Field instances (12 binary variables,
4096 states each). Visualizes distribution shape, correlation structure, training
samples, and single-variable marginals.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.gridspec import GridSpec

# ── Constants ─────────────────────────────────────────────────────────────────
N_VARS = 12
N_STATES = 4096
N_INSTANCES = 10
DATA_DIR = "/Users/vmehtola/Documents/TCDQ/TCDQ/fbm-classical-training/data/mrf"

# Binary representations of all 4096 states: shape (4096, 12)
BIT_MATRIX = ((np.arange(N_STATES)[:, None] >> np.arange(N_VARS)) & 1).astype(float)


# ── Data loading and precomputation ──────────────────────────────────────────
def load_all_instances():
    """Load all 10 MRF instances and precompute derived quantities."""
    data = {}
    for i in range(N_INSTANCES):
        path = f"{DATA_DIR}/{i}"
        dist = np.loadtxt(f"{path}/target_dist.txt")
        samples = np.loadtxt(f"{path}/training_set.txt")[:, ::-1]  # align bit order

        # Exact marginals: P(xi=1) = sum over states where xi=1 of p(state)
        exact_marginals = BIT_MATRIX.T @ dist

        # Exact covariance: Cov(xi, xj) = E[xi*xj] - E[xi]*E[xj]
        # E[xi*xj] = sum_s p(s) * xi(s) * xj(s)
        weighted = BIT_MATRIX * dist[:, None]  # (4096, 12) weighted by prob
        exact_second = weighted.T @ BIT_MATRIX  # (12, 12) E[xi*xj]
        exact_corr = exact_second - np.outer(exact_marginals, exact_marginals)

        # Sample marginals
        sample_marginals = samples.mean(axis=0)

        # Sample covariance
        sample_second = samples.T @ samples / len(samples)
        sample_cov = sample_second - np.outer(sample_marginals, sample_marginals)

        # Entropy: -sum(p * log2(p)) for p > 0
        nonzero = dist > 0
        entropy = -np.sum(dist[nonzero] * np.log2(dist[nonzero]))

        # Sample histogram: bin each sample into its state index
        sample_indices = (samples @ (2 ** np.arange(N_VARS))).astype(int)
        sample_hist = np.bincount(sample_indices, minlength=N_STATES) / len(samples)

        data[i] = {
            "dist": dist,
            "samples": samples,
            "exact_marginals": exact_marginals,
            "exact_corr": exact_corr,
            "sample_marginals": sample_marginals,
            "sample_cov": sample_cov,
            "entropy": entropy,
            "sample_hist": sample_hist,
        }
    return data


print("Loading MRF instances...")
all_data = load_all_instances()
print("Done.")

# ── State ─────────────────────────────────────────────────────────────────────
state = {
    "instance": 0,
    "corr_source": "Exact dist",
    "n_visible": 50,
    "sample_offset": 0,
}

# ── Figure and axes ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.canvas.manager.set_window_title("MRF Data Explorer")

gs = GridSpec(3, 2, figure=fig, height_ratios=[5, 5, 0.8],
              hspace=0.35, wspace=0.3)

ax_corr = fig.add_subplot(gs[0, 0])
ax_samples = fig.add_subplot(gs[0, 1])
ax_dist = fig.add_subplot(gs[1, 0])
ax_marg = fig.add_subplot(gs[1, 1])


# ── Drawing functions ────────────────────────────────────────────────────────
def draw_correlation(ax):
    """Draw 12x12 correlation matrix heatmap."""
    ax.clear()

    # Remove old colorbar if it exists
    if hasattr(ax, "_mrf_cbar") and ax._mrf_cbar is not None:
        ax._mrf_cbar.remove()
        ax._mrf_cbar = None

    d = all_data[state["instance"]]
    if state["corr_source"] == "Exact dist":
        matrix = d["exact_corr"]
        title = "Exact Covariance"
    else:
        matrix = d["sample_cov"]
        title = "Sample Covariance"

    vmax = np.abs(matrix).max()
    if vmax == 0:
        vmax = 1e-6
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="equal", origin="upper")

    # Annotate cells
    for i in range(N_VARS):
        for j in range(N_VARS):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=5.5, color=color)

    ax.set_xticks(range(N_VARS))
    ax.set_yticks(range(N_VARS))
    ax.set_xticklabels([f"x{i}" for i in range(N_VARS)], fontsize=7)
    ax.set_yticklabels([f"x{i}" for i in range(N_VARS)], fontsize=7)
    ax.set_title(f"{title} — Instance {state['instance']}", fontsize=10)

    ax._mrf_cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_samples(ax):
    """Draw binary heatmap of visible training samples."""
    ax.clear()

    d = all_data[state["instance"]]
    samples = d["samples"]
    n_vis = state["n_visible"]
    offset = state["sample_offset"]

    # Clamp offset
    max_offset = max(0, len(samples) - n_vis)
    offset = min(offset, max_offset)
    state["sample_offset"] = offset

    chunk = samples[offset:offset + n_vis]
    ax.imshow(chunk, cmap="gray_r", aspect="auto", interpolation="nearest",
              vmin=0, vmax=1)

    ax.set_xlabel("Variable", fontsize=8)
    ax.set_ylabel("Sample index", fontsize=8)
    ax.set_xticks(range(N_VARS))
    ax.set_xticklabels([f"x{i}" for i in range(N_VARS)], fontsize=7)

    # Show a few y-tick labels
    n_rows = len(chunk)
    tick_step = max(1, n_rows // 5)
    yticks = list(range(0, n_rows, tick_step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(offset + t) for t in yticks], fontsize=7)

    ax.set_title(f"Training Samples [{offset}–{offset + n_vis - 1}]", fontsize=10)


def draw_distribution(ax):
    """Draw rank-frequency distribution profile."""
    ax.clear()

    d = all_data[state["instance"]]
    dist = d["dist"]
    sample_hist = d["sample_hist"]
    entropy = d["entropy"]

    # Sort exact distribution descending
    sort_idx = np.argsort(dist)[::-1]
    sorted_dist = dist[sort_idx]
    sorted_sample = sample_hist[sort_idx]

    ranks = np.arange(N_STATES)
    ax.semilogy(ranks, sorted_dist, "-", color="steelblue", linewidth=1.2,
                label="Exact distribution")

    # Sample overlay: only plot nonzero bins
    nonzero = sorted_sample > 0
    ax.semilogy(ranks[nonzero], sorted_sample[nonzero], ".", color="darkorange",
                markersize=3, alpha=0.7, label="Sample frequencies")

    # Uniform reference
    ax.axhline(1.0 / N_STATES, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.6, label="Uniform (1/4096)")

    ax.set_xlabel("Rank (sorted descending)", fontsize=8)
    ax.set_ylabel("Probability", fontsize=8)
    ax.set_xlim(0, N_STATES)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Distribution Profile — H = {entropy:.2f} bits (max {N_VARS})",
                 fontsize=10)
    ax.grid(True, alpha=0.2)


def draw_marginals(ax):
    """Draw grouped bar chart of exact vs sample marginals."""
    ax.clear()

    d = all_data[state["instance"]]
    exact_m = d["exact_marginals"]
    sample_m = d["sample_marginals"]

    xs = np.arange(N_VARS)
    width = 0.35
    ax.bar(xs - width / 2, exact_m, width, label="Exact P(xi=1)",
           color="steelblue")
    ax.bar(xs + width / 2, sample_m, width, label="Sample freq",
           color="darkorange")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"x{i}" for i in range(N_VARS)], fontsize=7)
    ax.set_ylabel("P(xi = 1)", fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Single-Variable Marginals — Instance {state['instance']}",
                 fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")


def redraw():
    """Redraw all four panels."""
    draw_correlation(ax_corr)
    draw_samples(ax_samples)
    draw_distribution(ax_dist)
    draw_marginals(ax_marg)
    fig.canvas.draw_idle()


# ── Widgets ──────────────────────────────────────────────────────────────────
# Instance slider
ax_inst_slider = fig.add_axes([0.06, 0.04, 0.18, 0.03])
slider_instance = Slider(ax_inst_slider, "Instance", 0, N_INSTANCES - 1,
                         valinit=0, valstep=1, color="steelblue")

# Correlation source radio buttons
ax_radio = fig.add_axes([0.30, 0.015, 0.12, 0.06])
radio_corr = RadioButtons(ax_radio, ("Exact dist", "Samples"), active=0)
for label in radio_corr.labels:
    label.set_fontsize(8)

# Visible samples slider
ax_vis_slider = fig.add_axes([0.50, 0.04, 0.16, 0.03])
slider_visible = Slider(ax_vis_slider, "Visible", 10, 100,
                        valinit=50, valstep=5, color="steelblue")

# Prev / Next buttons
ax_prev = fig.add_axes([0.74, 0.03, 0.06, 0.04])
btn_prev = Button(ax_prev, "◀ Prev", color="lightgray", hovercolor="lightblue")

ax_next = fig.add_axes([0.82, 0.03, 0.06, 0.04])
btn_next = Button(ax_next, "Next ▶", color="lightgray", hovercolor="lightblue")


# ── Event handlers ───────────────────────────────────────────────────────────
def on_instance_change(val):
    state["instance"] = int(val)
    state["sample_offset"] = 0
    redraw()


def on_corr_source(label):
    state["corr_source"] = label
    draw_correlation(ax_corr)
    fig.canvas.draw_idle()


def on_visible_change(val):
    state["n_visible"] = int(val)
    draw_samples(ax_samples)
    fig.canvas.draw_idle()


def on_prev(event):
    state["sample_offset"] = max(0, state["sample_offset"] - state["n_visible"])
    draw_samples(ax_samples)
    fig.canvas.draw_idle()


def on_next(event):
    n_total = len(all_data[state["instance"]]["samples"])
    max_offset = max(0, n_total - state["n_visible"])
    state["sample_offset"] = min(max_offset,
                                 state["sample_offset"] + state["n_visible"])
    draw_samples(ax_samples)
    fig.canvas.draw_idle()


# ── Connect events ───────────────────────────────────────────────────────────
slider_instance.on_changed(on_instance_change)
radio_corr.on_clicked(on_corr_source)
slider_visible.on_changed(on_visible_change)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)

# ── Initial draw and show ───────────────────────────────────────────────────
redraw()
plt.show()

"""
Random Graph Explorer

Interactive tool for generating and exploring random graphs across three classical
models: Erdős–Rényi G(n,p), Watts-Strogatz small-world, and Barabási–Albert
preferential attachment. Visualizes graph structure, adjacency matrix, degree
distribution, and network statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.gridspec import GridSpec

# ── Constants ────────────────────────────────────────────────────────────────
COLORS = {
    "edge": "#BDBDBD",
    "node_border": "#424242",
    "button": "lightgray",
    "button_hover": "lightblue",
    "slider": "steelblue",
}

DEFAULTS = {
    "n": 20,
    "p": 0.1,
    "k": 4,
    "beta": 0.3,
    "m": 2,
}

rng = np.random.default_rng()

# ── State ────────────────────────────────────────────────────────────────────
state = {
    "model": "Erdős–Rényi",
    "layout": "Circular",
    "n": DEFAULTS["n"],
    "p": DEFAULTS["p"],
    "k": DEFAULTS["k"],
    "beta": DEFAULTS["beta"],
    "m": DEFAULTS["m"],
    "adj": None,
    "pos": None,
    "stats": {},
}


# ── Graph generation ─────────────────────────────────────────────────────────
def generate_erdos_renyi(n, p):
    """Generate Erdős–Rényi G(n,p) random graph."""
    mask = rng.random((n, n)) < p
    upper = np.triu(mask, k=1)
    adj = (upper | upper.T).astype(int)
    return adj


def generate_watts_strogatz(n, k, beta):
    """Generate Watts-Strogatz small-world graph."""
    k = max(2, min(k, n - 1))
    if k % 2 == 1:
        k -= 1
    k = max(2, k)

    adj = np.zeros((n, n), dtype=int)
    # Build ring lattice: connect each node to k/2 neighbors on each side
    half_k = k // 2
    for i in range(n):
        for j in range(1, half_k + 1):
            neighbor = (i + j) % n
            adj[i, neighbor] = 1
            adj[neighbor, i] = 1

    # Rewire edges with probability beta
    for i in range(n):
        for j in range(1, half_k + 1):
            neighbor = (i + j) % n
            if rng.random() < beta:
                # Remove old edge
                adj[i, neighbor] = 0
                adj[neighbor, i] = 0
                # Pick new target (not self, not already connected)
                candidates = [v for v in range(n) if v != i and adj[i, v] == 0]
                if candidates:
                    new_target = rng.choice(candidates)
                    adj[i, new_target] = 1
                    adj[new_target, i] = 1
                else:
                    # No candidates; restore edge
                    adj[i, neighbor] = 1
                    adj[neighbor, i] = 1
    return adj


def generate_barabasi_albert(n, m):
    """Generate Barabási–Albert preferential attachment graph."""
    m = max(1, min(m, n - 2))
    n0 = m + 1  # Start with a fully connected clique of m+1 nodes

    adj = np.zeros((n, n), dtype=int)
    # Initial complete graph on first n0 nodes
    for i in range(n0):
        for j in range(i + 1, n0):
            adj[i, j] = 1
            adj[j, i] = 1

    degrees = adj.sum(axis=1).astype(float)

    for new_node in range(n0, n):
        # Preferential attachment: probability proportional to degree
        existing_degrees = degrees[:new_node].copy()
        total = existing_degrees.sum()
        if total == 0:
            probs = np.ones(new_node) / new_node
        else:
            probs = existing_degrees / total

        # Choose m distinct targets
        targets = set()
        while len(targets) < m:
            target = rng.choice(new_node, p=probs)
            targets.add(target)

        for t in targets:
            adj[new_node, t] = 1
            adj[t, new_node] = 1
            degrees[new_node] += 1
            degrees[t] += 1

    return adj


# ── Graph analysis ───────────────────────────────────────────────────────────
def compute_components(adj):
    """Find connected components via BFS. Returns list of sets."""
    n = len(adj)
    visited = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if visited[start]:
            continue
        comp = set()
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            comp.add(node)
            for neighbor in range(n):
                if adj[node, neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(comp)
    return components


def compute_clustering_coefficient(adj):
    """Compute average clustering coefficient."""
    n = len(adj)
    coeffs = np.zeros(n)
    for i in range(n):
        neighbors = np.where(adj[i] == 1)[0]
        k_i = len(neighbors)
        if k_i < 2:
            coeffs[i] = 0.0
            continue
        # Count edges among neighbors
        sub = adj[np.ix_(neighbors, neighbors)]
        triangles = sub.sum() / 2  # Each edge counted twice
        possible = k_i * (k_i - 1) / 2
        coeffs[i] = triangles / possible
    return coeffs.mean()


def analyze_graph(adj):
    """Compute all graph statistics."""
    n = len(adj)
    degrees = adj.sum(axis=1)
    n_edges = int(adj.sum() // 2)
    components = compute_components(adj)
    n_components = len(components)
    largest_component = max(len(c) for c in components) if components else 0
    avg_degree = degrees.mean()
    clustering = compute_clustering_coefficient(adj)

    return {
        "n": n,
        "n_edges": n_edges,
        "degrees": degrees,
        "n_components": n_components,
        "largest_component": largest_component,
        "avg_degree": avg_degree,
        "clustering": clustering,
        "components": components,
    }


# ── Layout algorithms ────────────────────────────────────────────────────────
def layout_circular(n):
    """Place nodes evenly on a unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def layout_spring(adj, iterations=80):
    """Fruchterman-Reingold force-directed layout."""
    n = len(adj)
    if n == 0:
        return np.empty((0, 2))
    if n == 1:
        return np.array([[0.0, 0.0]])

    # Optimal distance
    area = 1.0
    k = np.sqrt(area / n)

    # Initialize positions randomly
    pos = rng.uniform(-0.5, 0.5, (n, 2))
    temperature = 0.5

    for iteration in range(iterations):
        # Compute pairwise deltas: delta[i,j] = pos[i] - pos[j]
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 2)
        dist = np.sqrt((delta ** 2).sum(axis=2))  # (n, n)
        np.fill_diagonal(dist, 1e-6)  # avoid division by zero

        # Repulsive forces (all pairs): k^2 / dist, along delta direction
        repulsive_mag = k ** 2 / dist  # (n, n)
        repulsive = (delta * (repulsive_mag / dist)[:, :, np.newaxis]).sum(axis=1)

        # Attractive forces (edges only): dist^2 / k, along -delta direction
        edge_mask = adj.astype(float)  # (n, n)
        attractive_mag = dist ** 2 / k * edge_mask  # (n, n)
        attractive = -(delta * (attractive_mag / dist)[:, :, np.newaxis]).sum(axis=1)

        # Total displacement
        displacement = repulsive + attractive
        disp_norm = np.sqrt((displacement ** 2).sum(axis=1, keepdims=True))
        disp_norm = np.maximum(disp_norm, 1e-6)

        # Limit by temperature
        pos += displacement / disp_norm * np.minimum(disp_norm, temperature)
        temperature *= 0.95

    # Normalize to [-1, 1]
    pmin = pos.min(axis=0)
    pmax = pos.max(axis=0)
    span = pmax - pmin
    span[span == 0] = 1
    pos = 2 * (pos - pmin) / span - 1

    return pos


# ── Figure and axes ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.canvas.manager.set_window_title("Random Graph Explorer")

gs = GridSpec(3, 3, figure=fig, height_ratios=[5, 5, 0.8],
              width_ratios=[6, 3, 3], hspace=0.35, wspace=0.3)

ax_graph = fig.add_subplot(gs[0:2, 0])
ax_adj = fig.add_subplot(gs[0, 1:3])
ax_degree = fig.add_subplot(gs[1, 1])
ax_stats = fig.add_subplot(gs[1, 2])


# ── Drawing functions ────────────────────────────────────────────────────────
def draw_graph(ax):
    """Draw the node-link graph visualization."""
    ax.clear()

    adj = state["adj"]
    pos = state["pos"]
    stats = state["stats"]
    n = stats["n"]

    if n == 0:
        ax.set_title("Empty graph")
        ax.set_aspect("equal")
        return

    components = stats["components"]
    degrees = stats["degrees"]
    cmap = plt.cm.tab10

    # Assign colors by component
    node_colors = np.zeros(n, dtype=int)
    for idx, comp in enumerate(components):
        for node in comp:
            node_colors[node] = idx % 10

    # Draw edges
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color=COLORS["edge"], linewidth=0.6, zorder=1)

    # Draw nodes (sized by degree)
    min_size, max_size = 40, 300
    if degrees.max() > degrees.min():
        sizes = min_size + (max_size - min_size) * (degrees - degrees.min()) / (degrees.max() - degrees.min())
    else:
        sizes = np.full(n, (min_size + max_size) / 2)

    for i in range(n):
        ax.scatter(pos[i, 0], pos[i, 1], s=sizes[i],
                   c=[cmap(node_colors[i])], edgecolors=COLORS["node_border"],
                   linewidths=0.5, zorder=2)

    # Label nodes if few enough
    if n <= 20:
        for i in range(n):
            ax.text(pos[i, 0], pos[i, 1], str(i), ha="center", va="center",
                    fontsize=6, fontweight="bold", zorder=3)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title(f"{state['model']} — n={n}, {stats['n_edges']} edges, "
                 f"{state['layout']} layout", fontsize=10)
    ax.axis("off")


def draw_adjacency(ax):
    """Draw adjacency matrix heatmap."""
    ax.clear()

    adj = state["adj"]
    n = state["stats"]["n"]

    ax.imshow(adj, cmap="Blues", aspect="equal", interpolation="nearest",
              vmin=0, vmax=1)

    if n <= 30:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(range(n), fontsize=max(4, 7 - n // 10))
        ax.set_yticklabels(range(n), fontsize=max(4, 7 - n // 10))
    else:
        ax.set_xticks([0, n // 2, n - 1])
        ax.set_yticks([0, n // 2, n - 1])
        ax.set_xticklabels([0, n // 2, n - 1], fontsize=7)
        ax.set_yticklabels([0, n // 2, n - 1], fontsize=7)

    ax.set_title("Adjacency Matrix", fontsize=10)


def draw_degree_distribution(ax):
    """Draw degree distribution bar chart."""
    ax.clear()

    degrees = state["stats"]["degrees"]
    n = state["stats"]["n"]

    d_min, d_max = int(degrees.min()), int(degrees.max())
    bins = np.arange(d_min, d_max + 2) - 0.5
    ax.hist(degrees, bins=bins, color="steelblue", edgecolor="white",
            linewidth=0.5, alpha=0.85)

    # Expected degree line
    model = state["model"]
    expected = None
    label = None
    if model == "Erdős–Rényi":
        expected = (n - 1) * state["p"]
        label = f"E[deg] = (n-1)p = {expected:.1f}"
    elif model == "Watts-Strogatz":
        expected = state["k"]
        label = f"E[deg] = k = {expected}"
    elif model == "Barabási–Albert":
        expected = 2 * state["m"]
        label = f"E[deg] ≈ 2m = {expected}"

    if expected is not None:
        ax.axvline(expected, color="darkorange", linestyle="--", linewidth=1.5,
                   label=label)
        ax.legend(fontsize=7, loc="upper right")

    ax.set_xlabel("Degree", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Degree Distribution", fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")


def draw_stats(ax):
    """Draw statistics text panel."""
    ax.clear()
    ax.axis("off")

    s = state["stats"]
    n = s["n"]

    lines = [
        f"Nodes:  {n}",
        f"Edges:  {s['n_edges']}",
        f"Components:  {s['n_components']}",
        f"Largest comp.:  {s['largest_component']}",
        f"Avg degree:  {s['avg_degree']:.2f}",
        f"Clustering:  {s['clustering']:.4f}",
    ]

    # For ER, show phase transition thresholds
    if state["model"] == "Erdős–Rényi" and n > 1:
        p = state["p"]
        p_giant = 1 / n
        p_conn = np.log(n) / n

        giant_met = p >= p_giant
        conn_met = p >= p_conn

        lines.append("")
        lines.append("Phase transitions:")
        lines.append(f"  Giant comp. (p > 1/n = {p_giant:.3f})")
        lines.append(f"  Connected (p > ln(n)/n = {p_conn:.3f})")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace")

    # Color the threshold lines for ER
    if state["model"] == "Erdős–Rényi" and n > 1:
        p = state["p"]
        p_giant = 1 / n
        p_conn = np.log(n) / n

        y_giant = 0.95 - (len(lines) - 2) * 0.075
        y_conn = 0.95 - (len(lines) - 1) * 0.075

        color_giant = "green" if p >= p_giant else "red"
        color_conn = "green" if p >= p_conn else "red"

        ax.plot([0.02, 0.04], [y_giant, y_giant], color=color_giant,
                linewidth=3, transform=ax.transAxes)
        ax.plot([0.02, 0.04], [y_conn, y_conn], color=color_conn,
                linewidth=3, transform=ax.transAxes)

    ax.set_title("Statistics", fontsize=10)


def redraw():
    """Redraw all four display panels."""
    draw_graph(ax_graph)
    draw_adjacency(ax_adj)
    draw_degree_distribution(ax_degree)
    draw_stats(ax_stats)
    fig.canvas.draw_idle()


# ── Core actions ─────────────────────────────────────────────────────────────
def generate_graph():
    """Generate a new graph based on current model and parameters."""
    n = state["n"]
    model = state["model"]

    if model == "Erdős–Rényi":
        adj = generate_erdos_renyi(n, state["p"])
    elif model == "Watts-Strogatz":
        k = max(2, min(state["k"], n - 1))
        adj = generate_watts_strogatz(n, k, state["beta"])
    elif model == "Barabási–Albert":
        m = max(1, min(state["m"], n - 2))
        adj = generate_barabasi_albert(n, m)
    else:
        adj = np.zeros((n, n), dtype=int)

    state["adj"] = adj
    state["stats"] = analyze_graph(adj)

    # Compute layout
    if state["layout"] == "Circular":
        state["pos"] = layout_circular(n)
    else:
        state["pos"] = layout_spring(adj)

    redraw()


def update_slider_visibility():
    """Show/hide model-specific sliders."""
    model = state["model"]

    # ER sliders
    ax_p_slider.set_visible(model == "Erdős–Rényi")

    # WS sliders
    ax_k_slider.set_visible(model == "Watts-Strogatz")
    ax_beta_slider.set_visible(model == "Watts-Strogatz")

    # BA sliders
    ax_m_slider.set_visible(model == "Barabási–Albert")

    fig.canvas.draw_idle()


# ── Widgets ──────────────────────────────────────────────────────────────────
# Model radio buttons (left)
ax_model_radio = fig.add_axes([0.02, 0.01, 0.10, 0.07])
radio_model = RadioButtons(ax_model_radio,
                           ("Erdős–Rényi", "Watts-Strogatz", "Barabási–Albert"),
                           active=0)
for label in radio_model.labels:
    label.set_fontsize(7)

# Node count slider (always visible)
ax_n_slider = fig.add_axes([0.16, 0.045, 0.14, 0.025])
slider_n = Slider(ax_n_slider, "n", 5, 60, valinit=DEFAULTS["n"], valstep=1,
                  color=COLORS["slider"])

# ER: p slider
ax_p_slider = fig.add_axes([0.16, 0.015, 0.14, 0.025])
slider_p = Slider(ax_p_slider, "p", 0.0, 1.0, valinit=DEFAULTS["p"],
                  color=COLORS["slider"])

# WS: k slider (shares position with p)
ax_k_slider = fig.add_axes([0.16, 0.015, 0.14, 0.025])
slider_k = Slider(ax_k_slider, "k", 2, 20, valinit=DEFAULTS["k"], valstep=2,
                  color=COLORS["slider"])
ax_k_slider.set_visible(False)

# WS: beta slider
ax_beta_slider = fig.add_axes([0.34, 0.015, 0.14, 0.025])
slider_beta = Slider(ax_beta_slider, "β", 0.0, 1.0, valinit=DEFAULTS["beta"],
                     color=COLORS["slider"])
ax_beta_slider.set_visible(False)

# BA: m slider (shares position with p)
ax_m_slider = fig.add_axes([0.16, 0.015, 0.14, 0.025])
slider_m = Slider(ax_m_slider, "m", 1, 10, valinit=DEFAULTS["m"], valstep=1,
                  color=COLORS["slider"])
ax_m_slider.set_visible(False)

# New Graph button
ax_new_btn = fig.add_axes([0.52, 0.015, 0.08, 0.04])
btn_new = Button(ax_new_btn, "New Graph", color=COLORS["button"],
                 hovercolor=COLORS["button_hover"])

# Layout radio buttons (right)
ax_layout_radio = fig.add_axes([0.64, 0.01, 0.08, 0.06])
radio_layout = RadioButtons(ax_layout_radio, ("Circular", "Spring"), active=0)
for label in radio_layout.labels:
    label.set_fontsize(7)


# ── Event handlers ───────────────────────────────────────────────────────────
def on_model_change(label):
    state["model"] = label
    update_slider_visibility()
    generate_graph()


def on_n_change(val):
    state["n"] = int(val)
    generate_graph()


def on_p_change(val):
    state["p"] = val
    generate_graph()


def on_k_change(val):
    state["k"] = int(val)
    generate_graph()


def on_beta_change(val):
    state["beta"] = val
    generate_graph()


def on_m_change(val):
    state["m"] = int(val)
    generate_graph()


def on_new_graph(event):
    generate_graph()


def on_layout_change(label):
    state["layout"] = label
    # Recompute layout without regenerating graph
    adj = state["adj"]
    n = len(adj)
    if label == "Circular":
        state["pos"] = layout_circular(n)
    else:
        state["pos"] = layout_spring(adj)
    redraw()


# ── Connect events ───────────────────────────────────────────────────────────
radio_model.on_clicked(on_model_change)
slider_n.on_changed(on_n_change)
slider_p.on_changed(on_p_change)
slider_k.on_changed(on_k_change)
slider_beta.on_changed(on_beta_change)
slider_m.on_changed(on_m_change)
btn_new.on_clicked(on_new_graph)
radio_layout.on_clicked(on_layout_change)

# ── Initial draw and show ───────────────────────────────────────────────────
generate_graph()
plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python script.py <desired_time>")
    sys.exit(1)

tau_vals = np.array([350.0, 400.0])
wm_vals = np.linspace(1.28, 1.32, 100) * 2 * np.pi
desired_time = float(sys.argv[1])

pop_slices = []
t_index = np.argmin(np.abs(tlist - desired_time))

for tau in tau_vals:
    filename = f"results/pop_matrix_{tau}.npy"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found")
    pop_matrix = np.load(filename)  # shape: [len(wm_vals), len(tlist)]
    # pick one time slice for heatmap
    pop_slice = pop_matrix[:, t_index]  # shape: [len(wm_vals)]
    pop_slices.append(pop_slice)

# Stack into a 2D array: rows=tau, columns=wm
pop_matrices = np.stack(pop_slices, axis=0)  # shape: [len(tau_vals), len(wm_vals)]

# Convert pop_matrices to a 2D array if it’s a list
data = np.array(pop_matrices)  # shape: [len(tau_vals), len(wm_vals)]

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# Plot the heatmap
im = ax.imshow(data, aspect='auto', origin='lower', cmap='RdBu', interpolation='nearest')

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=16, width=2)
cbar.set_label(label=r"Probability [$P_e$]", size=18, weight='bold')

# X-axis: modulation frequencies in GHz
n_tau, n_freq = data.shape
xtick_idx = np.linspace(0, n_freq-1, 10, dtype=int)
xtick_labels = np.round(wm_vals[xtick_idx] / (2*np.pi), 4)  # convert rad/s → GHz
ax.set_xticks(xtick_idx)
ax.set_xticklabels(xtick_labels, fontsize=14)

# Y-axis: tau values
ytick_idx = np.arange(n_tau)
ytick_labels = tau_vals
ax.set_yticks(ytick_idx)
ax.set_yticklabels(ytick_labels, fontsize=14)

# Set color limits (optional)
im.set_clim(0.0, 1.0)

# Labels and title
ax.set_xlabel("Modulation Frequency [GHz]", fontsize=18, weight='bold')
ax.set_ylabel(r"$\tau$ [ns]", fontsize=18, weight='bold')
ax.set_title(f"Qubit Probability at t = {desired_time} ns", fontsize=20, weight='bold', pad=15)

plt.tight_layout()
plt.savefig("time_slice_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

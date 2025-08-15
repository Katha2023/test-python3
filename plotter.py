import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if len(sys.argv) < 2:
    print("Usage: python plotter.py <desired_time>")
    sys.exit(1)

# ---- Config ----
tau_vals = np.array(list(range(100, 201, 4)))  # ns
wm_vals = np.linspace(1.28, 1.33, 100) * 2 * np.pi  # rad/ns
tlist = np.linspace(0, 80000, 1000)  # ns
desired_time = float(sys.argv[1])

# ---- Load data ----
t_index = np.argmin(np.abs(tlist - desired_time))
pop_slices = []
for tau in tau_vals:
    filename = f"results/pop_matrix_{float(tau)}.npy"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found")
    pop_matrix = np.load(filename)  # shape: [len(wm_vals), len(tlist)]
    pop_slices.append(pop_matrix[:, t_index])

data = np.stack(pop_slices, axis=0)  # shape: [len(tau_vals), len(wm_vals)]
wm_GHz = wm_vals / (2 * np.pi)

# ---- Plot ----
if len(tau_vals) == 1:
    vals = data[0]

    fig = plt.figure(figsize=(8, 3.6))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.2, 2.0], hspace=0.35)

    # (a) Heat strip
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(vals[None, :],
                    extent=[wm_GHz.min(), wm_GHz.max(), 0, 1],
                    aspect='auto', origin='lower',
                    cmap='RdBu', interpolation='nearest')
    ax0.set_yticks([])
    ax0.set_xlabel(r'$\omega_m/2\pi$ (GHz)')
    ax0.set_title(fr'$P_e$ at $t = {desired_time:.0f}$ ns, $\tau$ = {tau_vals[0]} ns')
    cbar = plt.colorbar(im, ax=ax0, pad=0.02)
    cbar.set_label(r'$P_e$')

    # (b) Line plot
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(wm_GHz, vals, linewidth=1.8)
    ax1.set_xlabel(r'$\omega_m/2\pi$ (GHz)')
    ax1.set_ylabel(r'$P_e$')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax1.grid(True, alpha=0.3)

    plt.savefig("time_slice_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

else:
    # τ × ωm heatmap
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(data,
                   extent=[wm_GHz.min(), wm_GHz.max(), tau_vals.max(), tau_vals.min()],
                   aspect='auto', origin='upper',
                   cmap='RdBu', interpolation='nearest')
    ax.set_xlabel(r'$\omega_m/2\pi$ (GHz)')
    ax.set_ylabel(r'$\tau$ (ns)')
    ax.set_title(r'$P_e$ at $t={:.0f}$ ns'.format(desired_time))
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'$P_e$')

    plt.savefig("time_slice_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

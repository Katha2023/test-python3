import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm

wc = 3.4375 * 2 * np.pi
wa_bar = 2.137 * 2 * np.pi
A = 0.3*2* np.pi
g  = 0.05 * 2 * np.pi
kappa = 0.0004
N = 5

tlist = np.linspace(0, 80000, 1000)
wm_vals = np.linspace(1.28,1.32, 100) * 2 * np.pi
tau_vals = np.linspace(300, 400, 25)
tau_vals = np.array([350, 400])

a = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))
adag_a = a.dag() * a
sigp_sig = sm.dag() * sm
psi0 = tensor(basis(N, 0), basis(2, 1))
rho0 = ket2dm(psi0)

c_ops = []
if kappa > 0.0:
    c_ops.append(np.sqrt(kappa) * a)

H0 = wc * adag_a + g * (a.dag()*sm + a*sm.dag())
e_ops = [sigp_sig]
opts = {'nsteps': 50000}

td_expr = 'wa_bar + A*cos(wm*t) if (t % T) < tau else wa_bar'

pop_matrices = []
start_time = time.time()

desired_time = 10000
t_index = np.argmin(np.abs(tlist - desired_time))
for tau in tau_vals:
    T = 2 * tau
    H_td = [H0, [sigp_sig, td_expr]]

    def solve_for_wm(wm):
        args = {
            'tau': float(tau),
            'T': float(T),
            'wm': float(wm),
            'A': float(A),
            'wa_bar': float(wa_bar)
        }
        result = mesolve(H_td, rho0, tlist, c_ops, e_ops=e_ops, args=args, options=opts)
        return np.real(result.expect[0])

    n_cpus = max(1, cpu_count())
    print(f"Using {n_cpus} worker processes for parallel execution...")

    wm_list = [float(w) for w in wm_vals]
    with Pool(processes=n_cpus) as pool:
        results = []
        for res in tqdm(pool.imap_unordered(solve_for_wm, wm_list), 
                        total=len(wm_list), desc="Solving"):
            results.append(res[t_index])

    pop_matrix = np.array(results)
    pop_matrices.append(pop_matrix)

elapsed = time.time() - start_time
print(f"Simulation finished in {elapsed:.2f} seconds.")
data = np.array(pop_matrices)

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

im = ax.imshow(data, aspect='auto', origin='lower',
               cmap='RdBu', interpolation='nearest')

cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=16, width=2)
cbar.set_label(label=r"Probability [$P_e$]", size=18, weight='bold')

n_tau, n_freq = data.shape

xtick_idx = np.linspace(0, n_freq-1, 10, dtype=int)
xtick_labels = np.round(wm_vals[xtick_idx] / (2*np.pi), 4)
ax.set_xticks(xtick_idx)
ax.set_xticklabels(xtick_labels, fontsize=14)
ax.tick_params(axis='x', which='major', length=6, width=2)

ytick_idx = np.arange(n_tau)
ytick_labels = tau_vals
ax.set_yticks(ytick_idx)
ax.set_yticklabels(ytick_labels, fontsize=14)
ax.tick_params(axis='y', which='major', length=6, width=2)

im.set_clim(0.0, 1.0)
ax.set_xlabel("Modulation Frequency [GHz]", fontsize=18, weight='bold')
ax.set_ylabel(r"$\tau$ [ns]", fontsize=18, weight='bold')
ax.set_title(f"Qubit Probability at t = {desired_time} ns", fontsize=20, weight='bold', pad=15)

plt.tight_layout()
plt.savefig("plot.png", dpi=300, bbox_inches='tight')  # high-quality output
plt.close()

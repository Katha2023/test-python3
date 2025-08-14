import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from joblib import Parallel, delayed
import multiprocessing as mp
import time, warnings, contextlib
from tqdm.auto import tqdm
from joblib import parallel

warnings.filterwarnings("ignore", category=FutureWarning)

# ---- tqdm bridge for joblib ----
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

# ---- Parameters (same as yours) ----
wc     = 3.4375 * 2 * np.pi
wa_bar = 2.137  * 2 * np.pi
A      = 0.3    * 2 * np.pi
g      = 0.05   * 2 * np.pi
kappa  = 0.0004
gamma  = 0.0
N      = 5
#T      = 2*tau  # ns

tlist   = np.linspace(0, 80000, 1000)            # ns
t_meas  = 9700.0                                 # 10 μs = 10000 ns
i_meas  = int(np.argmin(np.abs(tlist - t_meas)))  # index to read Pe(t=10 μs)

wm_vals = np.linspace(1.28, 1.32, 100) * 2 * np.pi  # rad/ns (GHz * 2π)
wm_vals_GHz = wm_vals / (2 * np.pi)                 # for plotting

#taus = list(range(300, 401, 4))  # 300, 304, ..., 400 ns
taus = [350]

# ---- Operators & init ----
a        = tensor(destroy(N), qeye(2))
sm       = tensor(qeye(N), destroy(2))
adag_a   = a.dag() * a
sigp_sig = sm.dag() * sm

psi0 = tensor(basis(N, 0), basis(2, 1))  # |0,e>
rho0 = ket2dm(psi0)

c_ops = []
if kappa > 0.0: c_ops.append(np.sqrt(kappa) * a)
if gamma > 0.0: c_ops.append(np.sqrt(gamma) * sm)

H0 = wc * adag_a + g * (a.dag()*sm + a*sm.dag())

def pulsed_wq(t, args):
    tau   = args['tau'];  wm = args['wm']
    A     = args['A'];   wa_bar = args['wa_bar']
    return wa_bar + A * np.cos(wm * t) if (t % (2*tau)) < tau else wa_bar

opts = {"nsteps": 100000, "store_states": False, "progress_bar": None, "atol": 1e-8, "rtol": 1e-6}

def solve_one_tau(wm, tau):
    args = {'tau': float(tau), 'wm': float(wm), 'A': float(A), 'wa_bar': float(wa_bar)}
    H = [H0, [sigp_sig, pulsed_wq]]
    res = mesolve(H, rho0, tlist, c_ops, [sigp_sig], args=args, options=opts)
    return np.asarray(res.expect[0], dtype=np.float32)

# ---- Sweep (parallel in ωm for each τ) ----
n_jobs = max(1, mp.cpu_count() - 1)
print(f"Sweeping {len(taus)} τ-values × {len(wm_vals)} ωm with {n_jobs} workers...")

Pe_map = np.empty((len(taus), len(wm_vals)), dtype=np.float32)

t0 = time.perf_counter()
for itau, tau in enumerate(tqdm(taus, desc="Outer sweep τ (ns)")):
    with tqdm_joblib(tqdm(total=len(wm_vals), desc=f"  ωm sweep @ τ={int(tau)} ns", leave=False)):
        rows = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
            delayed(solve_one_tau)(wm, tau) for wm in wm_vals
        )
    # take Pe at t = 10 μs
    Pe_map[itau, :] = np.array([row[i_meas] for row in rows], dtype=np.float32)

print(f"Total sweep time: {time.perf_counter()-t0:.1f} s")

import matplotlib.ticker as mticker

wm_GHz = wm_vals_GHz
taus_np = np.array(taus, dtype=float)

if len(taus) == 1:
    vals = Pe_map[0]

    fig = plt.figure(figsize=(8, 3.6))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.2, 2.0], hspace=0.35)

    # (a) 1-row "strip" heatmap for band visualization
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(vals[None, :],
                    extent=[wm_GHz.min(), wm_GHz.max(), 0, 1],
                    aspect='auto', origin='lower',
                    cmap='RdBu', interpolation='nearest')
    ax0.set_yticks([])
    ax0.set_xlabel(r'$\omega_m/2\pi$ (GHz)')
    ax0.set_title(fr'Pe at t = {t_meas:.0f} ns,  $\tau$ = {taus[0]} ns')
    cbar = plt.colorbar(im, ax=ax0, pad=0.02)
    cbar.set_label(r'$P_e$')

    # (b) Line plot of the same data
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(wm_GHz, vals, linewidth=1.8)
    ax1.set_xlabel(r'$\omega_m/2\pi$ (GHz)')
    ax1.set_ylabel(r'$P_e(t={:.0f}\,\mathrm{{ns}})$'.format(t_meas))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()

else:
    # τ × ωm heatmap
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(Pe_map,
                   extent=[wm_GHz.min(), wm_GHz.max(), taus_np.max(), taus_np.min()],
                   aspect='auto', origin='upper', cmap='RdBu', interpolation='nearest')
    ax.set_xlabel(r'$\omega_m/2\pi$ (GHz)')
    ax.set_ylabel(r'$\tau$ (ns)')
    ax.set_title(r'$P_e$ at $t={:.0f}$ ns'.format(t_meas))
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'$P_e$')
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from joblib import Parallel, delayed
import multiprocessing as mp
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# tqdm progress bar for joblib
from tqdm.auto import tqdm
from joblib import parallel
import contextlib

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib to report into tqdm progress bar."""
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

wc = 3.4375 * 2 * np.pi
wa_bar = 2.137  * 2 * np.pi
A  = 0.02     * 2 * np.pi
g  = 0.05    * 2 * np.pi
kappa = 0.004
N = 5
tau = 350
T = 700

tlist = np.linspace(0, 80000, 300)
wm_vals = np.linspace(1.28, 1.32, 200) * 2 * np.pi  # rad/s

a = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))
adag_a = a.dag() * a
sigp_sig = sm.dag() * sm

psi0 = tensor(basis(N, 0), basis(2, 1))  # |0,e>
rho0 = ket2dm(psi0)

c_ops = []
if kappa > 0.0:
    c_ops.append(np.sqrt(kappa) * a)

H0 = wc * adag_a + g * (a.dag()*sm + a*sm.dag())

def pulsed_wq(t, args):
    tau = args['tau']; T = args['T']; wm = args['wm']; A = args['A']; wa_bar = args['wa_bar']
    return wa_bar + A * np.cos(wm * t) if (t % T) < tau else wa_bar

opts = {"nsteps": 100000, "store_states": False, "progress_bar": None, "atol": 1e-8, "rtol": 1e-6}

def solve_one(idx, wm):
    args = {'tau': float(tau), 'T': float(T), 'wm': float(wm), 'A': float(A), 'wa_bar': float(wa_bar)}
    H = [H0, [sigp_sig, pulsed_wq]]
    t0 = time.perf_counter()
    res = mesolve(H, rho0, tlist, c_ops, [sigp_sig], args=args, options=opts)
    dt = time.perf_counter() - t0
    return idx, np.asarray(res.expect[0], dtype=np.float32), dt

# -------- Parallel sweep with progress bar --------
n_jobs = max(1, mp.cpu_count())
print(f"Running {len(wm_vals)} frequencies with {n_jobs} parallel workers...")

t_start = time.perf_counter()
with tqdm_joblib(tqdm(total=len(wm_vals), desc="Sweeping ωm")):
    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        batch_size=1
    )(
        delayed(solve_one)(i, wm) for i, wm in enumerate(wm_vals)
    )
total_time = time.perf_counter() - t_start
print(f"\nTotal parallel sweep time: {total_time:.2f} s")

pop_matrix = np.empty((len(wm_vals), len(tlist)), dtype=np.float32)
task_times = np.empty(len(wm_vals), dtype=np.float64)
for i, vec, dt in results:
    pop_matrix[i, :] = vec
    task_times[i] = dt

print(f"Per-task mesolve time: min={task_times.min():.3f}s  "
      f"median={np.median(task_times):.3f}s  max={task_times.max():.3f}s")

# -------- Plot --------
Tgrid, WMgrid = np.meshgrid(tlist, wm_vals / (2 * np.pi))
plt.figure(figsize=(9, 5))
c = plt.contourf(Tgrid, WMgrid, pop_matrix, levels=100, cmap='viridis_r')
plt.xlabel("Time")
plt.ylabel("Modulation frequency $\\omega_m / 2\\pi$ (Hz)")
plt.title("Contour plot of ⟨σ⁺σ⁻⟩ vs modulation frequency and time")
plt.colorbar(c, label="⟨σ⁺σ⁻⟩")
plt.tight_layout()
plt.savefig("single_tau.png")
plt.close()

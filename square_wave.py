import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from qutip import *
from joblib import Parallel, delayed
import multiprocessing as mp
import time
import warnings
warnings.filterwarnings("ignore")

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
A  = 0.3     * 2 * np.pi
g  = 0.05    * 2 * np.pi
kappa = 0.0004
gamma  = 0.0
N = 5

tlist = np.linspace(0, 80000, 1000)
wm_vals = np.linspace(1.280, 1.320, 100) * 2 * np.pi  # rad/s

a = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))
adag_a = a.dag() * a
sigp_sig = sm.dag() * sm

psi0 = tensor(basis(N, 0), basis(2, 1))  # |0,e>
rho0 = ket2dm(psi0)

c_ops = []
if kappa > 0.0:
    c_ops.append(np.sqrt(kappa) * a)
if gamma > 0.0:
    c_ops.append(np.sqrt(gamma) * sm)

H0 = wc * adag_a + g * (a.dag()*sm + a*sm.dag())

def pulsed_wq(t, args):
    tau = args['tau']; T = args['T']; wm = args['wm']; A = args['A']; wa_bar = args['wa_bar']
    return wa_bar + A * np.cos(wm * t) if (t % T) < tau else wa_bar

opts = {"nsteps": 100000, "store_states": False, "progress_bar": None, "atol": 1e-8, "rtol": 1e-6}

n_jobs = max(1, mp.cpu_count())
print(f"Using {n_jobs} parallel workers.")

taus = np.array([350, 400])

all_pop_matrices = []
all_task_times = []

for tau_val in taus:
    print(f"\nRunning sweep for tau = {tau_val:.1f}")
    T = 2*tau_val

    def solve_one(idx, wm):
        args = {'tau': float(tau_val), 'T': float(T), 'wm': float(wm), 'A': float(A), 'wa_bar': float(wa_bar)}
        H = [H0, [sigp_sig, pulsed_wq]]
        t0 = time.perf_counter()
        res = mesolve(H, rho0, tlist, c_ops, e_ops=[sigp_sig], args=args, options=opts)
        dt = time.perf_counter() - t0
        return idx, np.asarray(res.expect[0], dtype=np.float32), dt

    t_start = time.perf_counter()
    with tqdm_joblib(tqdm(total=len(wm_vals), desc=f"Sweeping ωm (tau={tau_val:.1f})")):
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            batch_size=1
        )(
            delayed(solve_one)(i, wm) for i, wm in enumerate(wm_vals)
        )
    total_time = time.perf_counter() - t_start
    print(f"Total parallel sweep time for tau={tau_val:.1f}: {total_time:.2f} s")

    pop_matrix = np.empty((len(wm_vals), len(tlist)), dtype=np.float32)
    task_times = np.empty(len(wm_vals), dtype=np.float64)
    for i, vec, dt in results:
        pop_matrix[i, :] = vec
        task_times[i] = dt

    print(f"Per-task mesolve time for tau={tau_val:.1f}: min={task_times.min():.3f}s  "
          f"median={np.median(task_times):.3f}s  max={task_times.max():.3f}s")

    all_pop_matrices.append(pop_matrix)
    all_task_times.append(task_times)

    # Save full dataset for this tau:
    filename = f"population_vs_wm_tau_{int(tau_val)}.csv"
    np.savetxt(filename, pop_matrix, delimiter=",")
    print(f"Saved population matrix for tau={tau_val:.1f} to '{filename}'")

print("\nAll sweeps completed.")

'''
import matplotlib.pyplot as plt

def summarize_time_series(ts):
    n = len(ts)
    return np.mean(ts)

summary_matrix = np.array([
    [summarize_time_series(pop_matrix[i]) for i in range(pop_matrix.shape[0])]
    for pop_matrix in all_pop_matrices
])

plt.figure(figsize=(8, 6))
extent = [wm_vals[0]/(2*np.pi), wm_vals[-1]/(2*np.pi), taus[0], taus[-1]]
plt.imshow(summary_matrix, extent=extent, aspect='auto', origin='lower', cmap='RdBu')
plt.colorbar(label="Avg population")
plt.xlabel("Modulation frequency ωm / 2π (Hz)")
plt.ylabel("Pulse width τ")
plt.title("Population vs Modulation Frequency and Pulse Width")
plt.gca().invert_yaxis()  # invert y-axis
plt.savefig("population_vs_wm_tau.png", dpi=300)
plt.close()
'''

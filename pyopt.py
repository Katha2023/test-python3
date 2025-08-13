import numpy as np
import pyswarms as ps
import time
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import qmc

def rastrigin(X):
    X = np.atleast_2d(X)
    n_dimensions = X.shape[1]
    return np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1) + 10 * n_dimensions

n_dimensions = 5
lower_bounds = np.array([-4]*n_dimensions)
upper_bounds = np.array([5.12]*n_dimensions)
bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]
n_particles = min(2**15, max(4096, 500*n_dimensions))

sobol_sampler = qmc.Sobol(d=n_dimensions, scramble=False)
X_unit = sobol_sampler.random_base2(m=int(np.ceil(np.log2(n_particles))))
X_init = lower_bounds + X_unit * (upper_bounds - lower_bounds)

scores = np.array(Parallel(n_jobs=-1)(delayed(rastrigin)(x) for x in X_init))
n_top = max(1, int(0.01 * X_init.shape[0]))
top_indices = np.argsort(scores)[:n_top]
X_top = X_init[top_indices]

def parallel_objective(func, X):
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(func)(x) for x in X
    )
    return np.array(results)

class ParallelPSO(ps.single.GlobalBestPSO):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        super().__init__(*args, **kwargs)

    def _compute_objective(self, X):
        return parallel_objective(self.func, X)

v_max = 0.1 * (upper_bounds - lower_bounds)
v_min = -v_max
c = 1.5 + 0.5*n_dimensions/(n_dimensions+50)
w = 0.9 - 0.5*n_dimensions/(n_dimensions+50)
n_iters = 100 + 3*n_dimensions

options = {'c1': c, 'c2': c, 'w': w}

optimizer = ParallelPSO(
    n_particles=X_init.shape[0],
    func = rastrigin,
    dimensions=X_init.shape[1],
    options=options,
    bounds=(lower_bounds, upper_bounds),
    init_pos=X_init,
    velocity_clamp=(v_min, v_max)
)

start_time = time.time()
best_cost, best_pos = optimizer.optimize(rastrigin, iters=n_iters)
result = minimize(rastrigin, x0=best_pos, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
end_time = time.time()
print("Best cost:", result.fun)
print("Time taken:", end_time - start_time)

#%%
# Bayesian search over global-sim-springs-obstacle.py's parameters (mirrors
# param_optimisation.py, adapted for the always-on symmetric spring network -
# see that file's own module docstring for why it differs from
# global-sim-person.py's independently-asymmetric-per-cell K) to minimize the
# reservoir's source-decoding test NMSE (the task in data_processing.calc_nmse:
# predict the moving repulsion source's own [x, y] from the swarm's recorded
# state).
#
# Searched:
#   - beta_scale, k_scale, k_self_scale - damping / inter-robot spring
#     stiffness / homing stiffness, scaled near the calibrated formulas
#     BASE_BETA_*/BASE_K_*/BASE_K_SELF_* below (same values global-sim-
#     person.py and global-sim-springs-obstacle.py both use)
#   - k_global_scale - the moving obstacle's repulsion strength; no
#     real-robot reference point, so this gets a wider unconstrained range
#   - source_step - random_walk_wall_avoidance source's per-tick velocity
#     innovation scale
#   - Ridge alpha - free to tune per trial (no resimulation needed)
#
# NOT searched, unlike param_optimisation.py: connectivity_prob. The whole
# point of global-sim-springs-obstacle.py is an always-on, fully-connected
# spring network (see its module docstring) - every pair gets exactly one
# shared, symmetric stiffness draw (K[i,j] == K[j,i]), no Erdos-Renyi sparsity
# mask, so there's no connectivity dimension left to search.
#
# The physics functions below mirror global-sim-springs-obstacle.py and are
# duplicated rather than imported - that file's name has a hyphen so it isn't
# a valid Python module, and it's written as a notebook-style script (#%%
# cells, an IPython display() call) that isn't safe to import anyway. Only
# the random_walk_wall_avoidance source (the mode currently active there) is
# reproduced; circle/random_walk_oscillations/ball aren't part of this search.

import numpy as np
from numba import njit  # if not installed: pip install numba
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import optuna  # if not installed: pip install optuna
import importlib
import data_processing
importlib.reload(data_processing)

DT = 0.1
MAX_SPEED = 0.35

#%% PHYSICS CORE (mirrors global-sim-springs-obstacle.py)

def _wall_push(pos, size, margin, wall_strength):
    if pos < margin:
        return wall_strength * (margin - pos)
    elif pos > size - margin:
        return -wall_strength * (pos - (size - margin))
    return 0.0


def random_walk_wall_avoidance(prev_x, prev_y, prev_vx, prev_vy, size, step_scale, inertia,
                                wall_margin, wall_strength, rng):
    vx = (inertia * prev_vx + (1 - inertia) * rng.normal(0, step_scale)
          + _wall_push(prev_x, size, wall_margin, wall_strength))
    vy = (inertia * prev_vy + (1 - inertia) * rng.normal(0, step_scale)
          + _wall_push(prev_y, size, wall_margin, wall_strength))
    x_s = np.clip(prev_x + vx, 0, size)
    y_s = np.clip(prev_y + vy, 0, size)
    return x_s, y_s, vx, vy


@njit(cache=True)
def local_repulsion(x, y, x_s, y_s, K, R):
    N = x.shape[0]
    fx = np.zeros(N)
    fy = np.zeros(N)
    for i in range(N):
        dx = x[i] - x_s
        dy = y[i] - y_s
        d = (dx * dx + dy * dy) ** 0.5
        d_safe = max(d, 1e-3)
        if d < R:
            mag = K * (R - d) / d_safe
            fx[i] = mag * dx
            fy[i] = mag * dy
    return fx, fy


@njit(cache=True)
def Du(x, y, x0, y0, K, A, K_self):
    N = x.shape[0]
    sum_x = np.zeros(N)
    sum_y = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = (dx * dx + dy * dy) ** 0.5
            f = K[i, j] * (A[i, j] - d) / d
            sum_x[i] += f * dx
            sum_y[i] += f * dy
        dx0 = x[i] - x0[i]
        dy0 = y[i] - y0[i]
        d0 = (dx0 * dx0 + dy0 * dy0) ** 0.5
        if d0 > 1e-3:
            sum_x[i] += -0.5 * K_self[i] * dx0
            sum_y[i] += -0.5 * K_self[i] * dy0
    return sum_x, sum_y


@njit(cache=True)
def _step_core(x, y, theta, s, x0, y0, K, A, K_self, BETA, M, anchor,
               x_s, y_s, K_global, R_global, dt, max_speed):
    N = x.shape[0]
    Dx, Dy = Du(x, y, x0, y0, K, A, K_self)
    Gx, Gy = local_repulsion(x, y, x_s, y_s, K_global, R_global)
    x_new = np.empty(N)
    y_new = np.empty(N)
    s_new = np.empty(N)
    for i in range(N):
        energy = (Dx[i] + Gx[i]) * np.cos(theta[i]) + (Dy[i] + Gy[i]) * np.sin(theta[i])
        ds = (energy - BETA[i] * s[i]) / M
        ds = min(max(ds, -4.0), 4.0)
        sn = s[i] + ds * dt
        sn = min(max(sn, -max_speed), max_speed)
        if anchor[i]:
            x_new[i] = x[i]
            y_new[i] = y[i]
            s_new[i] = s[i]
        else:
            x_new[i] = x[i] + dt * np.cos(theta[i]) * sn
            y_new[i] = y[i] + dt * np.sin(theta[i]) * sn
            s_new[i] = sn
    return x_new, y_new, s_new


def step(state, t, rng):
    x_s, y_s, vx, vy = random_walk_wall_avoidance(
        state['source_x'], state['source_y'],
        state['source_vx'], state['source_vy'],
        state['size'], state['source_step'], state['source_inertia'],
        state['source_wall_margin'], state['source_wall_strength'], rng)
    state['source_vx'], state['source_vy'] = vx, vy
    state['source_x'], state['source_y'] = x_s, y_s

    x_new, y_new, s_new = _step_core(
        state['x'], state['y'], state['theta'], state['s'],
        state['x0'], state['y0'], state['K'], state['A'], state['K_self'],
        state['BETA'], state['M'], state['anchor'],
        x_s, y_s, state['K_global'], state['R_global'], DT, MAX_SPEED)
    state['x'] = x_new
    state['y'] = y_new
    state['s'] = s_new


def _random_positions_min_dist(N, size, min_dist, rng, max_attempts_per_point=2000):
    # rejection sampling: place robots one at a time, redrawing a candidate
    # point until it's at least min_dist from every point already placed -
    # mirrors global-sim-springs-obstacle.py's version but takes an explicit
    # rng so each trial's layout is reproducible from its seed
    xs = np.empty(N)
    ys = np.empty(N)
    for i in range(N):
        for _ in range(max_attempts_per_point):
            x = rng.uniform(size * 0.05, size * 0.95)
            y = rng.uniform(size * 0.05, size * 0.95)
            if i == 0 or np.all(np.hypot(xs[:i] - x, ys[:i] - y) >= min_dist):
                xs[i], ys[i] = x, y
                break
        else:
            raise ValueError(
                f"couldn't place robot {i + 1}/{N} with min_dist={min_dist} "
                f"after {max_attempts_per_point} attempts - try a smaller "
                f"min_dist, fewer robots, or a bigger arena")
    return xs, ys

#%% BASELINE (mirrors the current SETUP in global-sim-springs-obstacle.py -
# the "realistic, calibrated" reference point the search stays near)

N = 50
SIZE = 1
ANCHOR = np.zeros(N, dtype=bool)
M = 1.0
INIT_MIN_DIST = 0.1 * SIZE

BASE_BETA_LOW, BASE_BETA_HIGH, BASE_BETA_MULT = 1.31989, 2.830454, 5.0
BASE_K_OFFSET, BASE_K_RANGE, BASE_K_MULT = 7.8788, 5.0, 0.3
BASE_K_SELF_OFFSET, BASE_K_SELF_RANGE, BASE_K_SELF_MULT = 7.8788, 5.0, 0.3
BASE_K_GLOBAL = 50.0

SOURCE_STEP = 0.5 * SIZE
SOURCE_INERTIA = 0.995
SOURCE_WALL_MARGIN = 0.2 * SIZE
SOURCE_WALL_STRENGTH = 0.09

#%% SEARCH SPACE

# (low, high) for each parameter. beta/k/k_self_scale multiply the
# calibrated formulas above; k_global_scale sets the obstacle's strength
# directly since it has no calibrated baseline; source_step is the source's
# per-tick velocity innovation scale. No connectivity_prob - see module
# docstring.
SEARCH_RANGES = {
    'beta_scale':      (0.01, 5.0),
    'k_scale':         (0.01, 5.0),
    'k_self_scale':    (0.01, 5.0),
    'k_global_scale':  (0.3, 3.0),
    'source_step':     (0.05 * SIZE, 0.5 * SIZE),
}

ALPHA_CANDIDATES = [0.1, 1.0, 10.0, 100.0, 200.0]

N_TRIALS = 200
N_STARTUP_TRIALS = 20  # trials drawn uniformly at random before TPE starts
                        # exploiting the accumulating history
SEARCH_ITERATIONS = 20000  # short relative to the production run in
                            # global-sim-springs-obstacle.py - just enough to
                            # rank configs against each other. Re-run the
                            # winner at full length to confirm before
                            # committing to it.

#%% TRIAL MACHINERY

def build_state(params, seed):
    rng = np.random.RandomState(seed)

    x0, y0 = _random_positions_min_dist(N, SIZE, INIT_MIN_DIST, rng)
    theta = rng.uniform(0, 2 * np.pi, N)

    # resting spring length between every pair = their distance at t=0,
    # matching global-sim-springs-obstacle.py's A construction
    A = np.hypot(x0[:, None] - x0[None, :], y0[:, None] - y0[None, :])

    # symmetric, fully-connected spring matrix - one draw per unordered pair
    # (i, j), no connectivity sparsity mask (see module docstring)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            k = (BASE_K_OFFSET + rng.uniform() * BASE_K_RANGE) * BASE_K_MULT * params['k_scale']
            K[i, j] = k
            K[j, i] = k

    K_self = (BASE_K_SELF_OFFSET + rng.uniform(size=N) * BASE_K_SELF_RANGE) * BASE_K_SELF_MULT * params['k_self_scale']
    BETA = rng.uniform(BASE_BETA_LOW, BASE_BETA_HIGH, size=N) * BASE_BETA_MULT * params['beta_scale']

    K_global = BASE_K_GLOBAL * params['k_global_scale']
    R_global = 0.2 * SIZE

    source_step = params['source_step']

    return {
        'x': x0.copy(), 'y': y0.copy(), 'theta': theta, 's': np.zeros(N),
        'x0': x0, 'y0': y0, 'K': K, 'A': A, 'K_self': K_self,
        'BETA': BETA, 'M': M, 'anchor': ANCHOR,
        'size': SIZE, 'K_global': K_global, 'R_global': R_global,
        'source_step': source_step, 'source_inertia': SOURCE_INERTIA,
        'source_wall_margin': SOURCE_WALL_MARGIN, 'source_wall_strength': SOURCE_WALL_STRENGTH,
        'source_x': SIZE / 2, 'source_y': SIZE / 2,
        'source_vx': 0.0, 'source_vy': 0.0,
    }, rng


def simulate(state, rng, num_iterations):
    iterations = np.arange(0, num_iterations * DT, DT)
    n_steps = len(iterations)

    x_coords = np.empty((N, n_steps))
    y_coords = np.empty((N, n_steps))
    theta_coords = np.empty((N, n_steps))
    s_array = np.empty((N, n_steps))
    source_x = np.empty(n_steps)
    source_y = np.empty(n_steps)

    for i, t in enumerate(iterations):
        step(state, t, rng)
        x_coords[:, i] = state['x']
        y_coords[:, i] = state['y']
        theta_coords[:, i] = state['theta']
        s_array[:, i] = state['s']
        source_x[i] = state['source_x']
        source_y[i] = state['source_y']

    data = np.stack([x_coords[-int(N/2):], y_coords[-int(N/2):], theta_coords[-int(N/2):], s_array[-int(N/2):]])
    data_states = data.reshape(-1, data.shape[2]).T
    source_data = np.array([source_x, source_y])
    return data_states, source_data


def best_readout_nmse(data_states, source_data, alphas=ALPHA_CANDIDATES):
    # mirrors data_processing.calc_nmse's fit/NMSE logic, but works on
    # in-memory arrays directly (calc_nmse always reloads from
    # node-simulation.npz, which we don't want to overwrite on every one of
    # many trials) and additionally sweeps Ridge's alpha, which calc_nmse
    # leaves at its default
    best = None
    cut = int(data_states.shape[0] * 0.1)
    X = data_states[cut:, :]
    y = np.array(source_data)[:, cut:].T

    split_idx = int(0.95 * X.shape[0])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_t = scaler.fit_transform(X_train)
    X_test_t = scaler.transform(X_test)

    for alpha in alphas:
        lr = Ridge(alpha=alpha)
        lr.fit(X_train_t, y_train)
        pred_test = lr.predict(X_test_t)
        test_nmse = np.mean((y_test - pred_test) ** 2) / np.mean((y_test - np.mean(y_test, axis=0)) ** 2)
        if best is None or test_nmse < best['test_nmse']:
            best = {'test_nmse': test_nmse, 'alpha': alpha}
    return best


def run_trial(trial_idx, params):
    state, rng = build_state(params, seed=trial_idx)
    data_states, source_data = simulate(state, rng, SEARCH_ITERATIONS)
    return best_readout_nmse(data_states, source_data)


def objective(trial):
    # Optuna calls this once per trial, proposing params via its surrogate
    # model (TPE) instead of drawing them uniformly at random - it uses the
    # accumulating history of (params -> test_nmse) to concentrate later
    # trials on promising regions
    params = {name: trial.suggest_float(name, lo, hi) for name, (lo, hi) in SEARCH_RANGES.items()}
    result = run_trial(trial.number, params)
    trial.set_user_attr('alpha', result['alpha'])
    return result['test_nmse']


def run_bayesian_search(n_trials=N_TRIALS, seed=0):
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_STARTUP_TRIALS))
    study.optimize(objective, n_trials=n_trials)
    return study

#%% RUN SEARCH

study = run_bayesian_search(N_TRIALS)

results = [
    {**t.params, 'test_nmse': t.value, 'alpha': t.user_attrs['alpha'], 'trial': t.number}
    for t in study.trials
]

#%% BEST RESULT

best = min(results, key=lambda r: r['test_nmse'])
print("\nBest trial:")
for k, v in best.items():
    print(f"  {k}: {v}")
print("\nTo use these: in global-sim-springs-obstacle.py's SETUP, set")
print("BETA_SCALE = beta_scale, K_SCALE = k_scale, K_SELF_SCALE = k_self_scale,")
print("K_GLOBAL_SCALE = k_global_scale, and SOURCE_STEP = source_step * size.")
print(f"Also make sure N matches this search's N ({N}) - swarm density affects")
print("the interaction dynamics being tuned for. Then run a full-length")
print(f"confirmation before trusting the result - this search only ran")
print(f"{SEARCH_ITERATIONS} steps per trial to stay fast.")

#%% VISUALIZE: test NMSE vs each searched parameter

param_names = list(SEARCH_RANGES.keys())
_, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, name in zip(axes.ravel(), param_names):
    xs = [r[name] for r in results]
    ys = [r['test_nmse'] for r in results]
    ax.scatter(xs, ys)
    ax.set_xlabel(name)
    ax.set_ylabel("test NMSE")
plt.tight_layout()
plt.show()
# %%

#%%
# Random search over swarm/network parameters to minimize the reservoir's
# source-decoding test NMSE (the task in data_processing.calc_nmse: predict
# the moving repulsion source's own [x, y] from the swarm's recorded state).
#
# Searched:
#   - BETA, K, K_self (damping, inter-robot spring stiffness, homing
#     stiffness) - scaled near their current SETUP values in
#     global-sim-person.py, which were calibrated to match real dot-robot
#     experiments, so the search stays physically plausible
#   - connectivity_prob - random edge density on K (Erdos-Renyi sparsity):
#     what fraction of the N*(N-1) possible directed springs actually exist
#   - R_GLOBAL, K_GLOBAL - the added repulsion "person"'s reach and
#     strength, which have no real-robot reference point, so these get a
#     wider unconstrained range
#   - Ridge alpha - free to tune per trial (no resimulation needed), and
#     necessary for a fair comparison: a physical config could look
#     artificially bad just because its alpha wasn't tuned for it
#
# The physics functions below mirror global-sim-person.py and are
# duplicated rather than imported - that file's name has a hyphen so it
# isn't a valid Python module, and it's written as a notebook-style script
# (#%% cells, an IPython display() call) that isn't safe to import anyway.
# Only the random_walk_wall_avoidance source (the mode currently active
# there) is reproduced; circle mode isn't part of this search.

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

#%% PHYSICS CORE (mirrors global-sim-person.py)

def _wall_push(pos, size, margin, wall_strength):
    if pos < margin:
        return wall_strength * (margin - pos)
    elif pos > size - margin:
        return -wall_strength * (pos - (size - margin))
    return 0.0


def random_walk_pos(prev_x, prev_y, prev_vx, prev_vy, size, step_scale, inertia,
                     wall_margin, wall_strength):
    vx = (inertia * prev_vx + (1 - inertia) * np.random.normal(0, step_scale)
          + _wall_push(prev_x, size, wall_margin, wall_strength))
    vy = (inertia * prev_vy + (1 - inertia) * np.random.normal(0, step_scale)
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


def step(state, t):
    x_s, y_s, vx, vy = random_walk_pos(
        state['source_x'], state['source_y'],
        state['source_vx'], state['source_vy'],
        state['size'], state['source_step'], state['source_inertia'],
        state['source_wall_margin'], state['source_wall_strength'])
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

#%% BASELINE (mirrors the current SETUP in global-sim-person.py - the
# "realistic, calibrated" reference point the search stays near)

N = 50
SIZE = 0.1
ANCHOR = np.zeros(N, dtype=bool)
M = 1.0

BASE_BETA_LOW, BASE_BETA_HIGH, BASE_BETA_MULT = 1.31989, 2.830454, 5.0
BASE_K_OFFSET, BASE_K_RANGE, BASE_K_MULT = 7.8788, 5.0, 0.3
BASE_K_SELF_OFFSET, BASE_K_SELF_RANGE, BASE_K_SELF_MULT = 7.8788, 5.0, 0.3
BASE_K_GLOBAL = 50.0

SOURCE_STEP = 0.5 * SIZE
SOURCE_INERTIA = 0.995
SOURCE_WALL_MARGIN = 0.2 * SIZE
SOURCE_WALL_STRENGTH = 0.09

#%% SEARCH SPACE

# (low, high) for each parameter. beta/k/k_self scale multiply the
# calibrated formulas above; connectivity_prob is the fraction of K's
# off-diagonal edges kept; r_global_frac/k_global_scale set the source's
# reach and strength directly since they have no calibrated baseline.
SEARCH_RANGES = {
    'beta_scale':        (0.01, 5.0),
    'k_scale':            (0.01, 5.0),
    'k_self_scale':        (0.01, 5.0),
    'connectivity_prob':   (0.2, 1.0),
    'r_global_frac':       (0.1, 0.6),
    'k_global_scale':      (0.3, 3.0),
}

ALPHA_CANDIDATES = [0.1, 1.0, 10.0, 100.0, 200.0]

N_TRIALS = 200
SEARCH_ITERATIONS = 20000  # short relative to the 500,000-step "production"
                            # run in global-sim-person.py - just enough to
                            # rank configs against each other. Re-run the
                            # winner at full length to confirm before
                            # committing to it.

#%% TRIAL MACHINERY

def build_state(params, seed):
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(0, SIZE, N)
    y0 = rng.uniform(0, SIZE, N)
    theta = rng.uniform(0, 2 * np.pi, N)

    A = np.hypot(x0[:, None] - x0[None, :], y0[:, None] - y0[None, :])

    K = (BASE_K_OFFSET + rng.uniform(size=(N, N)) * BASE_K_RANGE) * BASE_K_MULT * params['k_scale']
    np.fill_diagonal(K, 0.0)
    # Erdos-Renyi sparsity: independently drop each directed edge with
    # probability (1 - connectivity_prob), same "not necessarily symmetric"
    # philosophy as K's stiffness values themselves (see Du's docstring in
    # global-sim-person.py)
    connectivity_mask = rng.uniform(size=(N, N)) < params['connectivity_prob']
    K = K * connectivity_mask

    K_self = (BASE_K_SELF_OFFSET + rng.uniform(size=N) * BASE_K_SELF_RANGE) * BASE_K_SELF_MULT * params['k_self_scale']
    BETA = rng.uniform(BASE_BETA_LOW, BASE_BETA_HIGH, size=N) * BASE_BETA_MULT * params['beta_scale']

    K_global = BASE_K_GLOBAL * params['k_global_scale']
    R_global = params['r_global_frac'] * SIZE

    return {
        'x': x0.copy(), 'y': y0.copy(), 'theta': theta, 's': np.zeros(N),
        'x0': x0, 'y0': y0, 'K': K, 'A': A, 'K_self': K_self,
        'BETA': BETA, 'M': M, 'anchor': ANCHOR,
        'size': SIZE, 'K_global': K_global, 'R_global': R_global,
        'source_step': SOURCE_STEP, 'source_inertia': SOURCE_INERTIA,
        'source_wall_margin': SOURCE_WALL_MARGIN, 'source_wall_strength': SOURCE_WALL_STRENGTH,
        'source_x': SIZE / 2, 'source_y': SIZE / 2,
        'source_vx': 0.0, 'source_vy': 0.0,
    }


def simulate(state, num_iterations):
    iterations = np.arange(0, num_iterations * DT, DT)
    n_steps = len(iterations)

    x_coords = np.empty((N, n_steps))
    y_coords = np.empty((N, n_steps))
    theta_coords = np.empty((N, n_steps))
    s_array = np.empty((N, n_steps))
    source_x = np.empty(n_steps)
    source_y = np.empty(n_steps)

    for i, t in enumerate(iterations):
        step(state, t)
        x_coords[:, i] = state['x']
        y_coords[:, i] = state['y']
        theta_coords[:, i] = state['theta']
        s_array[:, i] = state['s']
        source_x[i] = state['source_x']
        source_y[i] = state['source_y']

    data = np.stack([x_coords, y_coords, theta_coords, s_array])
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
    state = build_state(params, seed=trial_idx)
    data_states, source_data = simulate(state, SEARCH_ITERATIONS)
    return best_readout_nmse(data_states, source_data)


def objective(trial):
    # Optuna calls this once per trial, proposing params via its surrogate
    # model (TPE) instead of drawing them uniformly at random like the old
    # random_search - it uses the accumulating history of (params ->
    # test_nmse) to concentrate later trials on promising regions
    params = {name: trial.suggest_float(name, lo, hi) for name, (lo, hi) in SEARCH_RANGES.items()}
    result = run_trial(trial.number, params)
    trial.set_user_attr('alpha', result['alpha'])
    return result['test_nmse']


def run_bayesian_search(n_trials=N_TRIALS, seed=0):
    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study

#%% RUN SEARCH

study = run_bayesian_search(N_TRIALS)

# flatten into the same list-of-dicts shape the old random_search returned,
# so the reporting/plotting below doesn't need to change
results = [
    {**t.params, 'test_nmse': t.value, 'alpha': t.user_attrs['alpha'], 'trial': t.number}
    for t in study.trials
]

#%% BEST RESULT

best = min(results, key=lambda r: r['test_nmse'])
print("\nBest trial:")
for k, v in best.items():
    print(f"  {k}: {v}")
print("\nTo use these: scale BETA/K/K_self in global-sim-person.py's SETUP by")
print("the *_scale values above, set K_GLOBAL = 50.0 * k_global_scale, and")
print("R_GLOBAL = r_global_frac * size. For connectivity_prob, apply a random")
print("mask to K the same way build_state() does here. Then run a full-length")
print("(num_iterations=500000) confirmation before trusting the result - this")
print(f"search only ran {SEARCH_ITERATIONS} steps per trial to stay fast.")

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

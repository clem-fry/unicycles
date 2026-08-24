#%%
import numpy as np
import plots
from IPython.display import HTML
import importlib
importlib.reload(plots)
import data_processing
importlib.reload(data_processing)
from numba import njit  # if not installed: pip install numba

#%%

DT = 0.1
MAX_SPEED = 0.35


def _random_positions_min_dist(N, size, min_dist, max_attempts_per_point=2000):
    # rejection sampling: place robots one at a time, redrawing a candidate
    # point until it's at least min_dist from every point already placed.
    # Simple and fine for the N/size/min_dist ranges used here; if min_dist
    # is too large for N robots to fit, raises rather than silently
    # clumping or looping forever.
    xs = np.empty(N)
    ys = np.empty(N)
    for i in range(N):
        for _ in range(max_attempts_per_point):
            x = np.random.uniform(size * 0.05, size * 0.95)
            y = np.random.uniform(size * 0.05, size * 0.95)
            if i == 0 or np.all(np.hypot(xs[:i] - x, ys[:i] - y) >= min_dist):
                xs[i], ys[i] = x, y
                break
        else:
            raise ValueError(
                f"couldn't place robot {i + 1}/{N} with min_dist={min_dist} "
                f"after {max_attempts_per_point} attempts - try a smaller "
                f"min_dist, fewer robots, or a bigger arena")
    return xs, ys


def init_positions(layout, N, size, min_dist=None):
    # initial robot placement - 'random' (original behavior, or with a
    # minimum spacing if min_dist is given), 'grid' (evenly spaced on a
    # square grid), or 'circle' (evenly spaced on a ring around the arena
    # center)
    if layout == 'random':
        if min_dist is None:
            x0 = np.random.uniform(0, size, N)
            y0 = np.random.uniform(0, size, N)
        else:
            x0, y0 = _random_positions_min_dist(N, size, min_dist)
    elif layout == 'grid':
        n_cols = int(np.ceil(np.sqrt(N)))
        n_rows = int(np.ceil(N / n_cols))
        xs = (np.arange(n_cols) + 0.5) * (size / n_cols)
        ys = (np.arange(n_rows) + 0.5) * (size / n_rows)
        grid_x, grid_y = np.meshgrid(xs, ys)
        x0 = grid_x.ravel()[:N]
        y0 = grid_y.ravel()[:N]
    elif layout == 'circle':
        cx, cy = size / 2, size / 2
        r = size * 0.4
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x0 = cx + r * np.cos(angles)
        y0 = cy + r * np.sin(angles)
    else:
        raise ValueError(f"unknown layout: {layout!r}")
    return x0, y0


def global_input_pos(t, T, size):
    # circle trial: point orbits the arena once per period T, replacing the
    # old space-independent broadcast
    cx, cy = size / 2, size / 2
    r = size * 0.4
    omega = 2 * np.pi / T
    return cx + r * np.cos(omega * t), cy + r * np.sin(omega * t)


def _wall_push(pos, size, margin, wall_strength):
    # zero everywhere except within `margin` of an edge, where it grows
    # linearly into a push back toward the interior. Unlike a center-pull,
    # this has no attractor anywhere - it never nudges the source unless
    # it's actually near a wall, so it can't set up the resonant "ringing"
    # a global restoring force would (a damped spring pulled toward one
    # fixed point overshoots and swings back, giving the trajectory a
    # periodic signature a linear readout could exploit as a shortcut).
    if pos < margin:
        return wall_strength * (margin - pos)
    elif pos > size - margin:
        return -wall_strength * (pos - (size - margin))
    return 0.0


def random_walk_wall_avoidance(prev_x, prev_y, prev_vx, prev_vy, size, step_scale, inertia,
                     wall_margin, wall_strength):
    # random_walk trial: the random innovation drives velocity, not position
    # directly, and velocity carries over (weighted by inertia) between
    # ticks - so the source glides on a smooth curved path instead of
    # jittering in a new direction every tick. Only gets nudged when close
    # to a wall (see _wall_push); the interior is an unbiased free random
    # walk. np.clip is kept only as a rare safety net for extreme
    # excursions, not the normal edge behavior.
    vx = (inertia * prev_vx + (1 - inertia) * np.random.normal(0, step_scale)
          + _wall_push(prev_x, size, wall_margin, wall_strength))
    vy = (inertia * prev_vy + (1 - inertia) * np.random.normal(0, step_scale)
          + _wall_push(prev_y, size, wall_margin, wall_strength))

    x_s = np.clip(prev_x + vx, 0, size)
    y_s = np.clip(prev_y + vy, 0, size)

    return x_s, y_s, vx, vy

def random_walk_oscillations(prev_x, prev_y, prev_vx, prev_vy, size, step_scale, inertia, center_pull):
    cx, cy = size / 2, size / 2
    vx = (inertia * prev_vx + (1 - inertia) * np.random.normal(0, step_scale)
          - center_pull * (prev_x - cx))
    vy = (inertia * prev_vy + (1 - inertia) * np.random.normal(0, step_scale)
          - center_pull * (prev_y - cy))

    x_s = np.clip(prev_x + vx, 0, size)
    y_s = np.clip(prev_y + vy, 0, size)

    return x_s, y_s, vx, vy


def ball_pos(prev_x, prev_y, prev_vx, prev_vy, size, target_speed, speed_relax,
             heading_noise, restitution):
    # ball rolling around the arena, with a few "natural" touches instead of
    # a perfect billiard bounce:
    #   - speed relaxes toward target_speed each tick (an exponential
    #     pull, not a spring - no overshoot/ringing risk), so bounces cost
    #     energy without the ball ever grinding to a permanent stop over a
    #     long run - like a slight motor/downhill push fighting friction
    #   - heading gets a small random nudge every tick (path deviation):
    #     the direction of travel slowly wanders instead of being ruler-
    #     straight between bounces
    #   - restitution < 1 loses a fraction of speed on the axis that
    #     actually bounced (an inelastic wall collision), matching how a
    #     real ball loses energy on impact rather than a perfect reflection
    speed = np.hypot(prev_vx, prev_vy)
    heading = np.arctan2(prev_vy, prev_vx) + np.random.normal(0, heading_noise)
    speed = speed + speed_relax * (target_speed - speed)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    x_s = prev_x + vx
    y_s = prev_y + vy

    if x_s < 0:
        x_s = -x_s
        vx = -vx * restitution
    elif x_s > size:
        x_s = 2 * size - x_s
        vx = -vx * restitution

    if y_s < 0:
        y_s = -y_s
        vy = -vy * restitution
    elif y_s > size:
        y_s = 2 * size - y_s
        vy = -vy * restitution

    return x_s, y_s, vx, vy


def update_source(state, t, T):
    # advances the repulsion source and stores its new position in state -
    # random_walk needs the previous position, so this can't be a pure
    # function of (t, T) the way the circle trial is
    if state['source_mode'] == 'circle':
        x_s, y_s = global_input_pos(t, T, state['size'])
    elif state['source_mode'] == 'random_walk_wall_avoidance':
        x_s, y_s, vx, vy = random_walk_wall_avoidance(
            state['source_x'], state['source_y'],
            state['source_vx'], state['source_vy'],
            state['size'], state['source_step'], state['source_inertia'],
            state['source_wall_margin'], state['source_wall_strength'])
        state['source_vx'], state['source_vy'] = vx, vy

    elif state['source_mode'] == 'random_walk_oscillatoins':
            x_s, y_s, vx, vy = random_walk_oscillations(
                state['source_x'], state['source_y'],
                state['source_vx'], state['source_vy'],
                state['size'], state['source_step'], state['source_inertia'], state['source_centre_pull'])
            state['source_vx'], state['source_vy'] = vx, vy
    elif state['source_mode'] == 'ball':
        x_s, y_s, vx, vy = ball_pos(
            state['source_x'], state['source_y'],
            state['source_vx'], state['source_vy'],
            state['size'], state['source_ball_target_speed'],
            state['source_ball_speed_relax'], state['source_ball_heading_noise'],
            state['source_ball_restitution'])
        state['source_vx'], state['source_vy'] = vx, vy
    else:
        raise ValueError(f"unknown source_mode: {state['source_mode']!r}")
    state['source_x'], state['source_y'] = x_s, y_s
    return x_s, y_s


@njit(cache=True)
def local_repulsion(x, y, x_s, y_s, K, R):
    # Hookean, repulsion-only "spring" between each robot and the moving
    # source: robots inside radius R get pushed directly away, robots
    # outside it feel nothing - so only whoever the source is currently
    # near ("neighbouring" it) is impacted at any given t.
    # Explicit loop rather than vectorized numpy: with only N~30 robots,
    # numpy's per-call dispatch overhead dominates over the actual FLOPs, so
    # a numba-compiled loop is much faster here (see profiling in commit
    # history / conversation - ~8x on the full step() for N=30).
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
    # K[i, j] is the stiffness robot i perceives towards robot j. In the real
    # node this comes from deterministic_k(), which despite its name draws a
    # fresh random value on every call - so K is NOT symmetric here either,
    # each side of an edge can honestly disagree about how stiff it is.
    N = x.shape[0]
    sum_x = np.zeros(N)
    sum_y = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # skip self-interaction (K[i,i] is 0 anyway)
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = (dx * dx + dy * dy) ** 0.5
            f = K[i, j] * (A[i, j] - d) / d
            sum_x[i] += f * dx
            sum_y[i] += f * dy

        # spring back to this robot's own starting position
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
        # anchors don't move (real node skips its whole update() when self.anchor)
        if anchor[i]:
            x_new[i] = x[i]
            y_new[i] = y[i]
            s_new[i] = s[i]
        else:
            x_new[i] = x[i] + dt * np.cos(theta[i]) * sn
            y_new[i] = y[i] + dt * np.sin(theta[i]) * sn
            s_new[i] = sn
    return x_new, y_new, s_new


def step(state, t, T):
    # thin wrapper: unpacks state, runs the numba-compiled core, writes the
    # result back - update_source() stays plain Python since it's cheap and
    # touches numpy's global RNG (numba has its own separate RNG state)
    x_s, y_s = update_source(state, t, T)
    x_new, y_new, s_new = _step_core(
        state['x'], state['y'], state['theta'], state['s'],
        state['x0'], state['y0'], state['K'], state['A'], state['K_self'],
        state['BETA'], state['M'], state['anchor'],
        x_s, y_s, state['K_global'], state['R_global'], DT, MAX_SPEED)
    state['x'] = x_new
    state['y'] = y_new
    state['s'] = s_new

#%% SETUP

N = 20       # number of robots
size = 1
T = 70     # period of the global input signal

INIT_LAYOUT = 'random'   # 'random', 'grid', or 'circle'
INIT_MIN_DIST = 0.1 * size   # 'random' layout only: minimum spacing between
                              # robots' start positions (None = unconstrained)
x0, y0 = init_positions(INIT_LAYOUT, N, size, min_dist=INIT_MIN_DIST)
theta = np.random.uniform(0, 2*np.pi, N)  # fixed for the whole run: the real
                                           # node never publishes angular.z

# resting spring length between every pair = their distance at t=0, matching
# neighbours_rest_spring being set on the first neighbour callback
dx0 = x0[:, None] - x0[None, :]
dy0 = y0[:, None] - y0[None, :]
A = np.hypot(dx0, dy0)

# best trial from param_optimisation.py's Bayesian search (trial 86,
# test_nmse=0.00288 on a short 20,000-step search run - that's much shorter
# than num_iterations below, so treat this run as the confirmation check
# before trusting the result)

# these params give good results:
# BETA_SCALE = 0.2575741508217526
# K_SCALE = 0.13674124212076125
# K_SELF_SCALE = 2.5904874601780103
# CONNECTIVITY_PROB = 0.37219226119311855
# R_GLOBAL_FRAC = 0.5819262923311888
# K_GLOBAL_SCALE = 2.233767828569402

BETA_SCALE = 0.28425719607665745
K_SCALE = 0.44224548299162375
K_SELF_SCALE = 2.940917347454856
CONNECTIVITY_PROB = 0 #0.9083643015865849
R_GLOBAL_FRAC = 0.2 #0.5436806351031995
K_GLOBAL_SCALE = 2.4573961915776006



K = (7.8788 + np.random.uniform(size=(N, N)) * 5.0) * 0.3 * K_SCALE
np.fill_diagonal(K, 0.0)
# Erdos-Renyi sparsity from the param search: keep each directed edge with
# probability CONNECTIVITY_PROB (independently per edge, same as K's
# stiffness values not being symmetric - see Du's docstring)
connectivity_mask = np.random.uniform(size=(N, N)) < CONNECTIVITY_PROB
K = K * connectivity_mask

K_self = (7.8788 + np.random.uniform(size=N) * 5.0) * 0.3 * K_SELF_SCALE

BETA = np.random.uniform(1.31989, 2.830454, size=N) * 5 * BETA_SCALE
M = 1.0  # DotNode overrides M (and J) to 1 regardless of launch params

# moving repulsion source: pushes away whichever robots it currently comes
# within R_global of. SOURCE_MODE picks how it moves - 'circle' orbits the
# arena (see global_input_pos), 'random_walk' drifts via random_walk_pos
K_GLOBAL = 50.0 * K_GLOBAL_SCALE
R_GLOBAL = R_GLOBAL_FRAC * size

SOURCE_MODE = 'random_walk_wall_avoidance'   # 'circle' or 'random_walk' or 'random_walk_wall_avoidance
SOURCE_STEP = 0.5 * size    # random_walk only: velocity-innovation scale per tick
SOURCE_INERTIA = 0.995        # random_walk only: higher = smoother/slower-turning path
SOURCE_WALL_MARGIN = 0.2 * size   # random_walk only: distance from an edge at
                                   # which the wall push starts acting (0 outside it)
SOURCE_WALL_STRENGTH = 0.09        # random_walk_wall_avoidance only: how hard it pushes back
                                   # within that margin
SOURCE_CENTER_PULL = 0.005     # random_walk only: restoring pull toward the arena
                                # center: higher = tighter to center/fewer edge
                                # visits, 0 = old hard-bounce-at-the-wall behavior
SOURCE_BALL_SPEED = 0.02 * size   # 'ball' only: cruising speed it relaxes
                                   # toward; launched in a random direction
SOURCE_BALL_SPEED_RELAX = 0.05     # 'ball' only: how fast speed relaxes back
                                    # toward SOURCE_BALL_SPEED each tick after
                                    # a bounce (0 = no recovery -> can stall,
                                    # 1 = instant, no felt energy loss)
SOURCE_BALL_HEADING_NOISE = 0.05   # 'ball' only: per-tick random heading
                                    # perturbation (radians) - path deviation
SOURCE_BALL_RESTITUTION = 0.85     # 'ball' only: fraction of speed kept on
                                    # the bounced axis at each wall collision
                                    # (1 = perfectly elastic, old behavior)

anchor = np.zeros(N, dtype=bool)
# anchor[np.random.randint(N)] = True  # uncomment to freeze one robot in place

#%% SIMULATION

def simulation(show=False, coord_frame='global'):
    # coord_frame='local' stores each robot's displacement from its own
    # starting position (x - x0, y - y0) in data_states instead of absolute
    # arena coordinates - removes the arbitrary "where did this robot
    # happen to start" offset, so the readout sees relative motion instead.
    # Only affects what's stored/returned; the animation (if show=True)
    # always uses true absolute positions, since a "local" plot wouldn't be
    # spatially meaningful.
    if SOURCE_MODE == 'ball':
        # launch in a random direction - random_walk/circle modes start
        # from rest (0,0) and pick up velocity/position on their own, but
        # a ball with 0 initial velocity would just sit still forever
        launch_angle = np.random.uniform(0, 2 * np.pi)
        init_vx = SOURCE_BALL_SPEED * np.cos(launch_angle)
        init_vy = SOURCE_BALL_SPEED * np.sin(launch_angle)
    else:
        init_vx, init_vy = 0.0, 0.0

    state = {
        'x': x0.copy(), 'y': y0.copy(), 'theta': theta, 's': np.zeros(N),
        'x0': x0, 'y0': y0, 'K': K, 'A': A, 'K_self': K_self,
        'BETA': BETA, 'M': M, 'anchor': anchor,
        'size': size, 'K_global': K_GLOBAL, 'R_global': R_GLOBAL,
        'source_mode': SOURCE_MODE, 'source_step': SOURCE_STEP,
        'source_centre_pull': SOURCE_CENTER_PULL,
        'source_inertia': SOURCE_INERTIA,
        'source_wall_margin': SOURCE_WALL_MARGIN,
        'source_wall_strength': SOURCE_WALL_STRENGTH,
        'source_ball_target_speed': SOURCE_BALL_SPEED,
        'source_ball_speed_relax': SOURCE_BALL_SPEED_RELAX,
        'source_ball_heading_noise': SOURCE_BALL_HEADING_NOISE,
        'source_ball_restitution': SOURCE_BALL_RESTITUTION,
        'source_x': size / 2, 'source_y': size / 2,
        'source_vx': init_vx, 'source_vy': init_vy,
    }

    iterations = np.arange(0, num_iterations * DT, DT)
    n_steps = len(iterations)

    # preallocated arrays instead of Python lists built up via .append() in
    # a per-robot loop - that was 4*N scalar list.append() calls every tick
    # (120M+ over a million-iteration run), which dominated runtime far more
    # than the actual physics. A vectorized array write per tick is much
    # cheaper.
    x_coords = np.empty((N, n_steps))
    y_coords = np.empty((N, n_steps))
    theta_coords = np.empty((N, n_steps))
    s_array = np.empty((N, n_steps))
    source_x = np.empty(n_steps)
    source_y = np.empty(n_steps)

    for i, t in enumerate(iterations):
        step(state, t, T)
        x_coords[:, i] = state['x']
        y_coords[:, i] = state['y']
        theta_coords[:, i] = state['theta']
        s_array[:, i] = state['s']
        source_x[i] = state['source_x']
        source_y[i] = state['source_y']

        # keyed off the integer index rather than t itself - t drifts off
        # exact multiples of 1000 due to float accumulation in np.arange,
        # so `t % 1000 == 0` silently stops firing partway through a long run
        if i % 10000 == 0:
            print(f"{i}/{n_steps}  (t={t:.1f})")

    if coord_frame == 'local':
        stored_x = x_coords - state['x0'][:, None]
        stored_y = y_coords - state['y0'][:, None]
    elif coord_frame == 'global':
        stored_x = x_coords
        stored_y = y_coords
    else:
        raise ValueError(f"unknown coord_frame: {coord_frame!r}")

    data = np.stack([stored_x, stored_y, theta_coords, s_array])
    data_states = data.reshape(-1, data.shape[2]).T

    if show:
        ani = plots.animation(x_coords, y_coords, theta_coords,
                               source_x=source_x, source_y=source_y,
                               source_radius=state['R_global'])
    else:
        ani = None

    source_data = np.array([source_x, source_y])
    return data_states, ani, source_data

def unpack_data_states(data_states, N):
    # inverse of the x/y/theta/s -> data_states packing at the end of
    # simulation(): columns are ordered state-major (state_idx*N + node_idx),
    # so this just undoes that reshape/transpose. Note: if data_states was
    # produced with coord_frame='local', the recovered x/y are per-robot
    # displacements from their own start position, not absolute arena
    # coordinates - replay()'s animation will look wrong (robots clustered
    # near their local origins) unless you add state['x0']/state['y0'] back.
    T = data_states.shape[0]
    x_coords, y_coords, theta_coords, s_array = data_states.T.reshape(4, N, T)
    return x_coords, y_coords, theta_coords, s_array


def replay(data_states, source_data, N, R_global, max_frames=500, from_start=True):
    # re-render the animation for an already-computed run (still in memory,
    # or reloaded from node-simulation.npz) without resimulating
    x_coords, y_coords, theta_coords, _ = unpack_data_states(data_states, N)
    source_x, source_y = source_data[0], source_data[1]
    return plots.animation(x_coords, y_coords, theta_coords,
                            source_x=source_x, source_y=source_y,
                            source_radius=R_global, max_frames=max_frames,
                            from_start=from_start)

#%% RUN + ANIMATE

T = 50

num_iterations = 10000

data_states, ani, source_data = simulation(show=False, coord_frame='global')
#display(HTML(ani.to_jshtml()))

#%%
ani = replay(data_states, source_data, N, R_GLOBAL)
display(HTML(ani.to_jshtml()))

#ani.save("animation.gif", writer="pillow", fps=10)

#%%
plots.source_path(source_data[0][5000:10000], source_data[1][5000:10000], size=size)

#%%
filename = 'node-simulation.npz'

# [:, list(range(10)) + list(range(20, 30)) + list(range(40, 50)) + list(range(60, 70))]
# reuse the run from RUN + ANIMATE above rather than re-simulating
np.savez(filename, data_states=data_states,
         source_x=source_data[0], source_y=source_data[1], T=T)


#%%

lr, nmses, ys, Xs, predictions = data_processing.calc_nmse(source_data, lag=0, plot=True)
#data_processing.plot_coefficients(lr)

# %%

y_train, y_test = ys
prediction_train, prediction_test = predictions

data_processing.plot_predictions(y_test[:2000], prediction_test[:2000])
data_processing.plot_trajectory_2d(y_test[1000:1200], prediction_test[1000:1200], size=size)


#%% TRAIN PLOT

y_train, y_test = ys
prediction_train, prediction_test = predictions

data_processing.plot_predictions(y_train, prediction_train)
data_processing.plot_trajectory_2d(y_train, prediction_train, size=size)

#%% TEST PLOT

y_train, y_test = ys
prediction_train, prediction_test = predictions

data_processing.plot_predictions(y_test, prediction_test)
data_processing.plot_trajectory_2d(y_test, prediction_test, size=size)

# %% output weightings for linear readout

plots.weight_heatmap(lr, x0, y0, size=size)

data_processing.plot_weight_matrix(lr)

# %%

ani = data_processing.animate_trajectory_2d(y_test, prediction_test, size=size)
display(HTML(ani.to_jshtml()))

# %%

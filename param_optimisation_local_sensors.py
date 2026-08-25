#%%
# Bayesian search over the local-sensor swarm's parameters (mirrors
# param_optimisation.py, adapted for global-sim-local-sensors.py) to
# minimize the reservoir's source-decoding test NMSE (the task in
# data_processing.calc_nmse: predict the moving repulsion source's own
# [x, y] from the swarm's recorded state).
#
# Searched:
#   - beta_low, beta_high - damping's own sampling range (each robot's BETA
#     is drawn uniform in [beta_low, beta_high], sorted so order doesn't
#     matter). Previously a fixed calibrated range scaled by a single
#     beta_scale multiplier; now the range itself is searched directly.
#   - avoid_k_scale, avoid_r_frac - central strength/reach of the
#     repulsion-only avoidance, shared by all three "avoid something so you
#     don't collide with it" mechanisms: neighbouring robots
#     (local_neighbor_repulsion), the moving source (local_repulsion), and
#     the arena edge (wall_repulsion). One K/R dial rather than three,
#     mirroring global-sim-local-sensors.py's own choice to unify them -
#     they're the same kind of avoidance, just against different kinds of
#     "object", using the exact same linear (Hookean, repulsion-only)
#     force law mag = K*(R-d)/d for all three now that wall_repulsion has
#     dropped its old separate exponential curve (see global-sim-local-
#     sensors.py's own EXPONENTIAL_REPULSION unification) - no more
#     repulsion_steepness/lam knob to search, one less dimension. Each
#     robot then gets its own K, scaled off this central value by
#     ROBOT_PARAM_SPREAD (see below) - not searched per-robot, just the
#     shared centre they're spread around
#   - the source's "how energetic" knob, depending on SOURCE_MODE (see
#     that constant's own comment): source_ball_speed_frac (ball's cruising
#     speed, as a fraction of arena size) or source_step
#     (random_walk_wall_avoidance's per-tick velocity innovation scale) -
#     only one of the two is ever in SEARCH_RANGES/state at a time
#   - Ridge alpha - free to tune per trial (no resimulation needed), same
#     as param_optimisation.py
#
# Movement is holonomic, mirroring global-sim-local-sensors.py's own switch
# away from the unicycle constraint: each robot's combined force
# (local_neighbor_repulsion + home_spring + local_repulsion + wall_repulsion,
# i.e. Fx/Fy in _step_core) drives a plain damped-driven mass directly in
# world x/y (vx/vy) - no heading projection, so a push from any direction
# produces motion in that exact direction immediately. theta/s are no
# longer dynamical state; they're recorded purely as a polar re-encoding of
# vx/vy (theta = direction of travel, s = speed magnitude) so the
# downstream [x, y, theta, s]-per-robot feature layout data_states/
# best_readout_nmse expect stays unchanged.
#
# K_self / home_spring (each robot's pull back to its own starting
# position) is back in the search as k_self_scale, after an earlier run
# with it disabled entirely (K_self=0) to see what the swarm/decoding
# looked like without that competing pull. That run was motivated by
# diagnostics on the old calibrated SETUP showing home_spring's mean force
# (~0.13) dominating local_neighbor_repulsion's (~0.046) by ~3x, which
# looked like it was suppressing spread-out/avoidance behaviour. Given the
# physics has since changed a fair bit (holonomic motion, universal linear
# repulsion), it's worth letting the search itself decide K_self's weight
# again rather than assuming that old diagnosis still holds. BASE_K_SELF_LOW/
# HIGH set the raw per-robot sampling range; k_self_scale (searched, see
# SEARCH_RANGES) scales it, same pattern as avoid_k_scale/AVOID_K.
#
# Tried and reverted: an attract-repel spring for the neighbour-robot
# interaction (repulsive up close, attractive further out, settling
# neighbours toward an equilibrium spacing instead of only ever giving a
# transient "get away" kick). A full joint search across attract_strength_frac/
# rest_frac plus everything else (400 trials) still couldn't beat plain
# repulsion-only: best found was test_nmse~0.52 vs ~0.19, and performance
# got monotonically worse as attraction strength increased. (The revert
# left this file's local_neighbor_spring/K_attract/rest_local machinery
# still wired into _step_core/build_state/SEARCH_RANGES calling an
# undefined local_neighbor_spring - i.e. this file could not actually run -
# cleaned up along with the linear-repulsion swap above.)
#
# Not searched:
#   - ROBOT_RADIUS - ties resolve_hard_collisions's hard collision limit
#     (robots physically cannot end a tick overlapping the wall or each
#     other, regardless of the soft exponential forces above). A physical
#     robot-size constant, not something to trade off against decoding
#     accuracy, so it's fixed at the same value as
#     global-sim-local-sensors.py rather than searched.
#   - ROBOT_PARAM_SPREAD - how far each robot's own K is randomly scattered
#     around the searched central avoid_k_scale (a hardware/controller-
#     variance constant, same reasoning as ROBOT_RADIUS).
#
# The physics functions below mirror global-sim-local-sensors.py and are
# duplicated rather than imported - that file's name has a hyphen so it
# isn't a valid Python module, and it's written as a notebook-style script
# (#%% cells, an IPython display() call) that isn't safe to import anyway.
# Only random_walk_wall_avoidance and ball are reproduced (SOURCE_MODE
# below picks which); circle/random_walk_oscillations aren't part of this
# search.

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

#%% PHYSICS CORE (mirrors global-sim-local-sensors.py)

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


def ball_pos(prev_x, prev_y, prev_vx, prev_vy, size, target_speed, speed_relax,
             heading_noise, restitution, rng):
    # mirrors global-sim-local-sensors.py's ball_pos exactly, just taking an
    # explicit rng (rng.normal in place of np.random.normal) for the same
    # per-trial-reproducible-from-seed reason random_walk_wall_avoidance
    # above does
    speed = np.hypot(prev_vx, prev_vy)
    heading = np.arctan2(prev_vy, prev_vx) + rng.normal(0, heading_noise)
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


@njit(cache=True)
def local_repulsion(x, y, vx, vy, x_s, y_s, vx_s, vy_s, K, R, tau):
    # linear (Hookean), repulsion-only - see module docstring. K/tau[i] are
    # per-robot arrays. R is extended by tau[i]*closing_rate while the gap
    # to the source is shrinking (closing_rate = -d(distance)/dt > 0) - a
    # looming/time-to-collision cue, mirroring global-sim-local-sensors.py's
    # proximity_repulsion (see its docstring) - previously missing here
    # entirely (tau=0 always, i.e. no anticipation, robots only ever
    # reacted to plain static distance).
    N = x.shape[0]
    fx = np.zeros(N)
    fy = np.zeros(N)
    for i in range(N):
        dx = x[i] - x_s
        dy = y[i] - y_s
        dvx = vx[i] - vx_s
        dvy = vy[i] - vy_s
        d = (dx * dx + dy * dy) ** 0.5
        d_safe = max(d, 1e-3)
        closing_rate = -(dx * dvx + dy * dvy) / d_safe
        R_eff = R + tau[i] * max(closing_rate, 0.0)
        if d < R_eff:
            mag = K[i] * (R_eff - d) / d_safe
            fx[i] = mag * dx
            fy[i] = mag * dy
    return fx, fy


@njit(cache=True)
def local_neighbor_repulsion(x, y, vx, vy, K_local, R_local, tau):
    # linear (Hookean), repulsion-only - see module docstring. K_local[i]/
    # tau[i] are per-robot arrays. Same closing-rate R extension as
    # local_repulsion above, against neighbouring robots instead of the
    # source - see its docstring.
    N = x.shape[0]
    sum_x = np.zeros(N)
    sum_y = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dvx = vx[i] - vx[j]
            dvy = vy[i] - vy[j]
            d = (dx * dx + dy * dy) ** 0.5
            d_safe = max(d, 1e-3)
            closing_rate = -(dx * dvx + dy * dvy) / d_safe
            R_eff = R_local + tau[i] * max(closing_rate, 0.0)
            if d < R_eff:
                mag = K_local[i] * (R_eff - d) / d_safe
                sum_x[i] += mag * dx
                sum_y[i] += mag * dy
    return sum_x, sum_y


@njit(cache=True)
def wall_repulsion(x, y, vx, vy, size, K_wall, R_wall, tau):
    # linear (Hookean), repulsion-only - same law as local_repulsion/
    # local_neighbor_repulsion, see module docstring. K_wall[i]/tau[i] are
    # per-robot arrays; the push direction is fixed (+/-x or +/-y) so,
    # unlike those two, there's no distance vector to normalize, and the
    # closing rate is just -vx[i]/+vx[i] (or the y equivalent) since a wall
    # never moves.
    N = x.shape[0]
    fx = np.zeros(N)
    fy = np.zeros(N)
    for i in range(N):
        d = x[i]
        R_eff = R_wall + tau[i] * max(-vx[i], 0.0)
        if d < R_eff:
            fx[i] += K_wall[i] * (R_eff - d)
        d = size - x[i]
        R_eff = R_wall + tau[i] * max(vx[i], 0.0)
        if d < R_eff:
            fx[i] -= K_wall[i] * (R_eff - d)
        d = y[i]
        R_eff = R_wall + tau[i] * max(-vy[i], 0.0)
        if d < R_eff:
            fy[i] += K_wall[i] * (R_eff - d)
        d = size - y[i]
        R_eff = R_wall + tau[i] * max(vy[i], 0.0)
        if d < R_eff:
            fy[i] -= K_wall[i] * (R_eff - d)
    return fx, fy


@njit(cache=True)
def _clamp_to_walls(x, y, anchor, size, robot_radius):
    N = x.shape[0]
    for i in range(N):
        if anchor[i]:
            continue
        if x[i] < robot_radius:
            x[i] = robot_radius
        elif x[i] > size - robot_radius:
            x[i] = size - robot_radius
        if y[i] < robot_radius:
            y[i] = robot_radius
        elif y[i] > size - robot_radius:
            y[i] = size - robot_radius


@njit(cache=True)
def resolve_hard_collisions(x, y, anchor, size, robot_radius, n_iters=4):
    # mirrors global-sim-local-sensors.py's resolve_hard_collisions: a
    # direct position correction run after integration, on top of
    # wall_repulsion/local_neighbor_repulsion's soft exponential push. A
    # robot's centre can never end a tick closer than robot_radius to the
    # arena edge or 2*robot_radius to another robot's centre. Anchored
    # robots are immovable obstacles - only the non-anchored side of a pair
    # gets pushed. Sweeps n_iters times since resolving one overlap can
    # reopen another. Ends with one extra wall clamp on its own - without
    # it, the loop's last operation would always be a pairwise push, which
    # can walk a robot back past the edge with nothing left to catch it.
    N = x.shape[0]
    for _ in range(n_iters):
        _clamp_to_walls(x, y, anchor, size, robot_radius)

        min_dist = 2.0 * robot_radius
        for i in range(N):
            for j in range(i + 1, N):
                if anchor[i] and anchor[j]:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = (dx * dx + dy * dy) ** 0.5
                if d < min_dist:
                    if d < 1e-9:
                        dx, dy, d = 1.0, 0.0, 1e-9
                    overlap = min_dist - d
                    if anchor[i]:
                        x[j] -= overlap * dx / d
                        y[j] -= overlap * dy / d
                    elif anchor[j]:
                        x[i] += overlap * dx / d
                        y[i] += overlap * dy / d
                    else:
                        x[i] += 0.5 * overlap * dx / d
                        y[i] += 0.5 * overlap * dy / d
                        x[j] -= 0.5 * overlap * dx / d
                        y[j] -= 0.5 * overlap * dy / d
    _clamp_to_walls(x, y, anchor, size, robot_radius)
    return x, y


@njit(cache=True)
def home_spring(x, y, x0, y0, K_self):
    N = x.shape[0]
    sum_x = np.zeros(N)
    sum_y = np.zeros(N)
    for i in range(N):
        dx0 = x[i] - x0[i]
        dy0 = y[i] - y0[i]
        d0 = (dx0 * dx0 + dy0 * dy0) ** 0.5
        if d0 > 1e-3:
            sum_x[i] += -0.5 * K_self[i] * dx0
            sum_y[i] += -0.5 * K_self[i] * dy0
    return sum_x, sum_y


@njit(cache=True)
def _step_core(x, y, theta, vx, vy, x0, y0, K_local, R_local,
               K_self, BETA, M, anchor,
               x_s, y_s, vx_s, vy_s, K_global, R_global, size, K_wall, R_wall, tau,
               robot_radius, dt, max_speed):
    N = x.shape[0]
    Nx, Ny = local_neighbor_repulsion(x, y, vx, vy, K_local, R_local, tau)
    Hx, Hy = home_spring(x, y, x0, y0, K_self)
    Gx, Gy = local_repulsion(x, y, vx, vy, x_s, y_s, vx_s, vy_s, K_global, R_global, tau)
    Wx, Wy = wall_repulsion(x, y, vx, vy, size, K_wall, R_wall, tau)

    Fx = Nx + Hx + Gx + Wx
    Fy = Ny + Hy + Gy + Wy

    x_new = np.empty(N)
    y_new = np.empty(N)
    vx_new = np.empty(N)
    vy_new = np.empty(N)
    theta_new = np.empty(N)
    for i in range(N):
        # holonomic point mass in world x/y, mirroring global-sim-local-
        # sensors.py's own _step_core - no heading projection, so a push
        # from any direction moves the robot in exactly that direction.
        dvx = (Fx[i] - BETA[i] * vx[i]) / M
        dvx = min(max(dvx, -4.0), 4.0)
        vxn = vx[i] + dvx * dt

        dvy = (Fy[i] - BETA[i] * vy[i]) / M
        dvy = min(max(dvy, -4.0), 4.0)
        vyn = vy[i] + dvy * dt

        speed = (vxn * vxn + vyn * vyn) ** 0.5
        if speed > max_speed:
            scale = max_speed / speed
            vxn *= scale
            vyn *= scale

        if anchor[i]:
            x_new[i] = x[i]
            y_new[i] = y[i]
            vx_new[i] = vx[i]
            vy_new[i] = vy[i]
            theta_new[i] = theta[i]
        else:
            x_new[i] = x[i] + dt * vxn
            y_new[i] = y[i] + dt * vyn
            vx_new[i] = vxn
            vy_new[i] = vyn
            # theta is cosmetic/derived only - direction of travel, held at
            # its last value while nearly stationary rather than snapping
            # to an arbitrary angle
            if speed > 1e-6:
                theta_new[i] = np.arctan2(vyn, vxn)
            else:
                theta_new[i] = theta[i]
    x_new, y_new = resolve_hard_collisions(x_new, y_new, anchor, size, robot_radius)

    # gaps measured relative to the hard limit itself (0 = exactly at the
    # limit) - after resolve_hard_collisions these should always be >= 0;
    # logged as a sanity check that the hard limit actually held, not as
    # something the search is meant to trade off against decoding accuracy
    min_wall_gap = np.inf
    for i in range(N):
        min_wall_gap = min(min_wall_gap, x_new[i], size - x_new[i], y_new[i], size - y_new[i])
    min_wall_gap -= robot_radius
    min_neighbor_gap = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = ((x_new[i] - x_new[j]) ** 2 + (y_new[i] - y_new[j]) ** 2) ** 0.5
            min_neighbor_gap = min(min_neighbor_gap, d)
    min_neighbor_gap -= 2.0 * robot_radius
    return x_new, y_new, theta_new, vx_new, vy_new, min_neighbor_gap, min_wall_gap


def step(state, t, rng):
    if state['source_mode'] == 'ball':
        x_s, y_s, src_vx, src_vy = ball_pos(
            state['source_x'], state['source_y'],
            state['source_vx'], state['source_vy'],
            state['size'], state['source_ball_target_speed'], state['source_ball_speed_relax'],
            state['source_ball_heading_noise'], state['source_ball_restitution'], rng)
    elif state['source_mode'] == 'random_walk_wall_avoidance':
        x_s, y_s, src_vx, src_vy = random_walk_wall_avoidance(
            state['source_x'], state['source_y'],
            state['source_vx'], state['source_vy'],
            state['size'], state['source_step'], state['source_inertia'],
            state['source_wall_margin'], state['source_wall_strength'], rng)
    else:
        raise ValueError(f"unknown source_mode: {state['source_mode']!r}")
    state['source_vx'], state['source_vy'] = src_vx, src_vy
    state['source_x'], state['source_y'] = x_s, y_s

    x_new, y_new, theta_new, vx_new, vy_new, min_neighbor_gap, min_wall_gap = _step_core(
        state['x'], state['y'], state['theta'], state['vx'], state['vy'],
        state['x0'], state['y0'], state['K_local'], state['R_local'],
        state['K_self'], state['BETA'], state['M'], state['anchor'],
        x_s, y_s, state['source_vx'], state['source_vy'], state['K_global'], state['R_global'],
        state['size'], state['K_wall'], state['R_wall'], state['tau'],
        state['robot_radius'], DT, MAX_SPEED)
    state['x'] = x_new
    state['y'] = y_new
    state['theta'] = theta_new
    state['vx'] = vx_new
    state['vy'] = vy_new
    return min_neighbor_gap, min_wall_gap


def _random_positions_min_dist(N, size, min_dist, rng, max_attempts_per_point=2000):
    # rejection sampling: place robots one at a time, redrawing a candidate
    # point until it's at least min_dist from every point already placed -
    # mirrors global-sim-local-sensors.py's version but takes an explicit
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

#%% BASELINE (mirrors the current SETUP in global-sim-local-sensors.py -
# the "realistic, calibrated" reference point the search stays near)

N = 10
SIZE = 1
ANCHOR = np.zeros(N, dtype=bool)
M = 1.0
INIT_MIN_DIST = 0.1 * SIZE

# raw per-robot K_self sampling range - k_self_scale (searched, see
# SEARCH_RANGES) multiplies it, same pattern as BASE_AVOID_K/avoid_k_scale.
# Equivalent to the old offset/range/mult decomposition (LOW =
# old_offset*old_mult, HIGH = (old_offset+old_range)*old_mult).
BASE_K_SELF_LOW, BASE_K_SELF_HIGH = 2.36364, 3.86364
BASE_AVOID_K = 6.0

# which source model the search runs against - mirrors global-sim-local-
# sensors.py's own SOURCE_MODE (currently 'ball' there too). 'ball' rolls
# around bouncing off walls (ball_pos); 'random_walk_wall_avoidance' drifts
# freely and gets nudged back near an edge (random_walk_wall_avoidance) -
# only these two are implemented here, matching whichever's actually active
# in global-sim-local-sensors.py at the time.
SOURCE_MODE = 'ball'

SOURCE_STEP = 0.5 * SIZE
SOURCE_INERTIA = 0.995
SOURCE_WALL_MARGIN = 0.2 * SIZE
SOURCE_WALL_STRENGTH = 0.09

# 'ball' only - speed_relax/heading_noise/restitution fixed at global-sim-
# local-sensors.py's current values rather than searched (same "one
# primary knob searched, the rest fixed" pattern as source_step/
# SOURCE_WALL_MARGIN/SOURCE_WALL_STRENGTH for random_walk_wall_avoidance
# above) - source_ball_speed_frac (searched, see SEARCH_RANGES) is the
# "how energetic" knob for this mode, analogous to source_step's role
# there.
SOURCE_BALL_SPEED_RELAX = 0.05
SOURCE_BALL_HEADING_NOISE = 0.05
SOURCE_BALL_RESTITUTION = 0.85

# physical robot size (see resolve_hard_collisions) - fixed, not searched;
# same value as global-sim-local-sensors.py's ROBOT_RADIUS
ROBOT_RADIUS = 0.02 * SIZE

# per-robot K spread - fixed, not searched, same value and reasoning as
# global-sim-local-sensors.py's ROBOT_PARAM_SPREAD: each robot's own gain
# is a hardware/controller property, not something to trade off against
# decoding accuracy
ROBOT_PARAM_SPREAD = 0.5

#%% SEARCH SPACE

# (low, high) for each parameter. beta_low/beta_high are BETA's own sampling
# range, searched directly (see module docstring); avoid_k_scale/avoid_r_frac
# set the shared wall/neighbour/source avoidance strength/reach directly (see
# module docstring) - all three now purely linear, so there's no
# steepness/lam knob left to search; k_self_scale scales home_spring's
# strength (see BASE_K_SELF_LOW/HIGH above and the module docstring) - 0.0
# lets the search switch it off entirely if that's still what's best;
# loom_tau scales the closing-rate/looming lookahead shared by
# local_repulsion/local_neighbor_repulsion/wall_repulsion (see their
# docstrings) - how far ahead (in simulated time) a robot "leads" something
# that's closing in fast, extending its effective sensing radius by
# tau*closing_rate. 0.0 reproduces the old no-anticipation behaviour (every
# robot only ever reacts to plain static distance) if that's still best.
SEARCH_RANGES = {
    'beta_low':             (0.1, 30.0),
    'beta_high':            (1.0, 80.0),
    'avoid_k_scale':        (0.1, 10.0),
    #'avoid_r_frac':         (0.05, 0.2),
    'k_self_scale':         (0.0, 5.0),
    'loom_tau':             (0.0, 2.0),
}
if SOURCE_MODE == 'ball':
    # cruising speed the ball relaxes toward, as a fraction of the arena
    # size - centred on global-sim-local-sensors.py's current
    # SOURCE_BALL_SPEED (0.02*size) but with room either side
    SEARCH_RANGES['source_ball_speed_frac'] = (0.005, 0.08)
elif SOURCE_MODE == 'random_walk_wall_avoidance':
    SEARCH_RANGES['source_step'] = (0.05 * SIZE, 0.5 * SIZE)

ALPHA_CANDIDATES = [0.1, 1.0, 10.0, 100.0, 200.0]

N_TRIALS = 500
N_STARTUP_TRIALS = 250  # trials drawn uniformly at random before TPE starts
                        # exploiting the accumulating history. This landscape
                        # is more multimodal than it looks: with the default
                        # (10), TPE consistently locked onto a mediocre small
                        # avoid_r_frac basin (test_nmse ~0.7-0.8 across
                        # several seeds) because too few of the first trials
                        # landed anywhere near the better large-R region.
SEARCH_ITERATIONS = 20000  # short relative to the production run in
                            # global-sim-local-sensors.py - just enough to
                            # rank configs against each other. Re-run the
                            # winner at full length to confirm before
                            # committing to it.

#%% TRIAL MACHINERY

def build_state(params, seed):
    params['avoid_r_frac'] = 0.25
    rng = np.random.RandomState(seed)

    x0, y0 = _random_positions_min_dist(N, SIZE, INIT_MIN_DIST, rng)
    theta = rng.uniform(0, 2 * np.pi, N)

    K_self = rng.uniform(BASE_K_SELF_LOW, BASE_K_SELF_HIGH, size=N) * params['k_self_scale']
    beta_lo, beta_hi = sorted((params['beta_low'], params['beta_high']))
    BETA = rng.uniform(beta_lo, beta_hi, size=N)

    # wall/neighbour/source avoidance all share one K/R (mirroring
    # global-sim-local-sensors.py's K_avoid/R_avoid used identically by
    # proximity_repulsion and wall_repulsion) and stay pure repulsion -
    # being pulled toward a wall, another robot, or the source doesn't make
    # sense. Each robot gets its own K (a uniform random multiplier around
    # the tuned central value, same as K_self/BETA above) - see
    # ROBOT_PARAM_SPREAD comment for why.
    avoid_K = BASE_AVOID_K * params['avoid_k_scale']
    avoid_R = params['avoid_r_frac'] * SIZE

    K_per_robot = avoid_K * rng.uniform(1 - ROBOT_PARAM_SPREAD, 1 + ROBOT_PARAM_SPREAD, N)
    # same per-robot spread as K, shared across local_repulsion/
    # local_neighbor_repulsion/wall_repulsion (one TAU, mirroring how K/R
    # are already shared) - see SEARCH_RANGES' loom_tau comment
    tau_per_robot = params['loom_tau'] * rng.uniform(1 - ROBOT_PARAM_SPREAD, 1 + ROBOT_PARAM_SPREAD, N)

    K_local = K_global = K_wall = K_per_robot
    R_local = R_global = R_wall = avoid_R

    state = {
        'x': x0.copy(), 'y': y0.copy(), 'theta': theta,
        'vx': np.zeros(N), 'vy': np.zeros(N),
        'x0': x0, 'y0': y0, 'K_local': K_local, 'R_local': R_local,
        'K_self': K_self, 'BETA': BETA, 'M': M, 'anchor': ANCHOR,
        'size': SIZE, 'K_global': K_global, 'R_global': R_global,
        'K_wall': K_wall, 'R_wall': R_wall, 'tau': tau_per_robot, 'robot_radius': ROBOT_RADIUS,
        'source_mode': SOURCE_MODE,
        'source_x': SIZE / 2, 'source_y': SIZE / 2,
    }

    if SOURCE_MODE == 'ball':
        # launch in a random direction - starting from rest would just sit
        # still forever, since ball_pos's speed_relax pulls speed toward
        # target_speed multiplicatively off whatever speed it already has
        target_speed = params['source_ball_speed_frac'] * SIZE
        launch_angle = rng.uniform(0, 2 * np.pi)
        state['source_vx'] = target_speed * np.cos(launch_angle)
        state['source_vy'] = target_speed * np.sin(launch_angle)
        state['source_ball_target_speed'] = target_speed
        state['source_ball_speed_relax'] = SOURCE_BALL_SPEED_RELAX
        state['source_ball_heading_noise'] = SOURCE_BALL_HEADING_NOISE
        state['source_ball_restitution'] = SOURCE_BALL_RESTITUTION
    elif SOURCE_MODE == 'random_walk_wall_avoidance':
        state['source_vx'] = 0.0
        state['source_vy'] = 0.0
        state['source_step'] = params['source_step']
        state['source_inertia'] = SOURCE_INERTIA
        state['source_wall_margin'] = SOURCE_WALL_MARGIN
        state['source_wall_strength'] = SOURCE_WALL_STRENGTH
    else:
        raise ValueError(f"unknown SOURCE_MODE: {SOURCE_MODE!r}")

    return state, rng


def simulate(state, rng, num_iterations):
    iterations = np.arange(0, num_iterations * DT, DT)
    n_steps = len(iterations)

    x_coords = np.empty((N, n_steps))
    y_coords = np.empty((N, n_steps))
    theta_coords = np.empty((N, n_steps))
    s_array = np.empty((N, n_steps))
    source_x = np.empty(n_steps)
    source_y = np.empty(n_steps)

    min_neighbor_gap = np.inf
    min_wall_gap = np.inf
    for i, t in enumerate(iterations):
        step_min_neighbor, step_min_wall = step(state, t, rng)
        min_neighbor_gap = min(min_neighbor_gap, step_min_neighbor)
        min_wall_gap = min(min_wall_gap, step_min_wall)
        x_coords[:, i] = state['x']
        y_coords[:, i] = state['y']
        theta_coords[:, i] = state['theta']
        s_array[:, i] = np.hypot(state['vx'], state['vy'])  # speed magnitude, holonomic motion has no signed forward speed
        source_x[i] = state['source_x']
        source_y[i] = state['source_y']

    data = np.stack([x_coords[-int(N/2):], y_coords[-int(N/2):], theta_coords[-int(N/2):], s_array[-int(N/2):]])
    data_states = data.reshape(-1, data.shape[2]).T
    source_data = np.array([source_x, source_y])
    return data_states, source_data, min_neighbor_gap, min_wall_gap


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
    data_states, source_data, min_neighbor_gap, min_wall_gap = simulate(state, rng, SEARCH_ITERATIONS)
    result = best_readout_nmse(data_states, source_data)
    result['min_neighbor_gap'] = min_neighbor_gap
    result['min_wall_gap'] = min_wall_gap
    return result


def objective(trial):
    # Optuna calls this once per trial, proposing params via its surrogate
    # model (TPE) instead of drawing them uniformly at random - it uses the
    # accumulating history of (params -> test_nmse) to concentrate later
    # trials on promising regions. min_neighbor_gap/min_wall_gap aren't part
    # of the objective - resolve_hard_collisions makes collision-freeness a
    # hard guarantee regardless of the searched params, not something to
    # trade off against decoding accuracy - but they're logged as
    # user_attrs anyway, as a sanity check that the guarantee actually held
    # (should always be >= 0; see _step_core).
    params = {name: trial.suggest_float(name, lo, hi) for name, (lo, hi) in SEARCH_RANGES.items()}
    result = run_trial(trial.number, params)
    trial.set_user_attr('alpha', result['alpha'])
    trial.set_user_attr('min_neighbor_gap', result['min_neighbor_gap'])
    trial.set_user_attr('min_wall_gap', result['min_wall_gap'])
    return result['test_nmse']


def run_bayesian_search(n_trials=N_TRIALS, seed=0):
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_STARTUP_TRIALS))
    study.optimize(objective, n_trials=n_trials)
    return study

#%% RUN SEARCH

study = run_bayesian_search(N_TRIALS)

# flatten into a list-of-dicts, same shape param_optimisation.py produces
results = [
    {**t.params, 'test_nmse': t.value, 'alpha': t.user_attrs['alpha'],
     'min_neighbor_gap': t.user_attrs['min_neighbor_gap'],
     'min_wall_gap': t.user_attrs['min_wall_gap'], 'trial': t.number}
    for t in study.trials
]

#%% BEST RESULT

best = min(results, key=lambda r: r['test_nmse'])
print("\nBest trial:")
for k, v in best.items():
    print(f"  {k}: {v}")
if best['min_neighbor_gap'] < -1e-6 or best['min_wall_gap'] < -1e-6:
    print("\nWARNING: this config's min_neighbor_gap/min_wall_gap went negative")
    print("during the search run, which resolve_hard_collisions should make")
    print("impossible - that points at a bug in this file or in")
    print("global-sim-local-sensors.py's own resolve_hard_collisions, not a")
    print("parameter to tune around. Investigate before trusting this trial.")
print("\nTo use these: in global-sim-local-sensors.py's SETUP, set BETA_LOW/")
print("BETA_HIGH to beta_low/beta_high (sorted above), AVOID_K_SCALE to")
print("avoid_k_scale, AVOID_R_FRAC to 0.25 (fixed here, not searched - see")
print("build_state), K_SELF_SCALE to k_self_scale, and LOOM_TAU to loom_tau.")
if SOURCE_MODE == 'ball':
    print("SOURCE_MODE there must be 'ball' to match, with SOURCE_BALL_SPEED =")
    print("source_ball_speed_frac * size (SOURCE_BALL_SPEED_RELAX/")
    print("HEADING_NOISE/RESTITUTION fixed at this file's SOURCE_BALL_* values,")
    print("not searched).")
elif SOURCE_MODE == 'random_walk_wall_avoidance':
    print("SOURCE_MODE there must be 'random_walk_wall_avoidance' to match, with")
    print("SOURCE_STEP = source_step * size.")
print("Also make sure EXPONENTIAL_REPULSION is False there and N matches this")
print(f"search's N ({N}) - this search assumes the linear force law throughout,")
print(f"same as here, and swarm density (N) affects the interaction dynamics")
print(f"being tuned for. Then run a full-length confirmation before trusting the")
print(f"result - this search only ran {SEARCH_ITERATIONS} steps per trial to stay fast.")

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

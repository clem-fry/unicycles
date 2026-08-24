#%%
# Variant of global-sim-person.py: robots have no virtual springs to each
# other. Instead each robot only "senses" a nearby point once it's within a
# local radius, and reacts with a repulsion-only nonlinear potential to
# avoid it. No rest length, no attraction, no per-edge stiffness matrix:
# purely local, purely repulsive collision avoidance.
#
# proximity_repulsion is the one channel every nearby point goes through -
# other robots AND the moving "walker" source alike, same K/R/tau, no
# identity check anywhere. What lets the walker still make a "noticeable
# impression" without being labeled as special: the effective sensing
# radius grows with how fast the gap to that point is closing (a
# looming/time-to-collision cue), and the walker is normally the only thing
# around with real, sustained velocity - jostling robots mostly don't have
# that. A robot moving that fast would get identical treatment. Arena walls
# get the exact same treatment too: wall_repulsion uses the same linear
# force law and the same K/R/tau (see its docstring) - the only reason it's
# a separate function is that a wall is a line, not a point, so the
# nearest-point/direction math is simpler to write directly than to route
# through proximity_repulsion's general point-to-point form.
#
# Each robot keeps its own pull back toward its starting position
# (home_spring) - that's not a reaction to a neighbour, it's the real
# DotNode's own dead-reckoning drift correction, so it stays as-is.
#
# Motion is now fully holonomic instead of unicycle-constrained: each
# robot's combined sensed force (proximity_repulsion + home_spring +
# wall_repulsion, i.e. Fx/Fy in _step_core) drives a plain 2D damped-driven
# mass directly in world x/y - same BETA/M dynamics as before, just applied
# to vx/vy straight instead of being projected onto a heading first. A push
# from any direction produces motion in that exact direction immediately;
# there's no more "can only move/turn along the line it's currently facing"
# constraint. theta is kept only as a cosmetic facing indicator for the
# animation (see _step_core) - it tracks the direction of travel but no
# longer feeds back into it.
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
MAX_SPEED = 0.35  # caps speed magnitude (any direction), not a per-axis cap


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
        # proximity_repulsion's looming/closing-rate term needs a real
        # source velocity now (previously source_vx/vy were unused for this
        # mode) - analytic derivative of global_input_pos's circular path
        r = state['size'] * 0.4
        omega = 2 * np.pi / T
        state['source_vx'] = -r * omega * np.sin(omega * t)
        state['source_vy'] = r * omega * np.cos(omega * t)
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
def proximity_repulsion(x, y, vx, vy, other_x, other_y, other_vx, other_vy, K, R, tau, lam, exponential,
                         robot_coupling):
    # the one channel every nearby point goes through - see module
    # docstring. `other_*` is built by the caller (_step_core) as [the N
    # robots themselves] followed by [the walker source] in one array, so
    # this function has no way to tell them apart - same K/R/tau for
    # every entry, same loop, same formula.
    #
    # Force is repulsion-only, zero outside R - same as before (a robot i
    # excludes itself via the `j == i` guard below, which only ever fires
    # for the robot block of `other_*`, not the walker). R is extended by
    # tau[i] * closing_rate while the gap to that point is shrinking
    # (closing_rate = -d(distance)/dt > 0), a looming/time-to-collision
    # cue - the same one biological collision avoidance uses to react to
    # an approaching threat without recognizing what it is. closing_rate
    # <= 0 (not approaching, or receding) leaves R unextended, so a point
    # that's just standing close but not closing in gets only the plain
    # distance-based push. `exponential` picks the penetration-depth curve:
    # False = linear/Hookean (magnitude K[i]*(R_eff-d)); True = exponential
    # (K[i]*(exp((R_eff-d)/lam[i])-1), decay length lam[i]) - same shape as
    # wall_repulsion's, softer far from contact and much harder right at
    # it. Same switch, same K/R/tau/lam, for every sensed point here and
    # every wall in wall_repulsion - see module docstring.
    # robot_coupling=False drops every other-robot entry too (only the
    # walker slot, j==N, still counts) - each robot then reacts to the
    # walker exactly as before but never to its neighbours, i.e. N
    # independent sensors instead of an interacting swarm. Baseline for
    # checking whether inter-robot coupling itself adds anything the
    # readout can use, beyond what N non-interacting sensors already give
    # it - see the module docstring/simulation()'s robot_coupling param.
    N = x.shape[0]
    M = other_x.shape[0]
    fx = np.zeros(N)
    fy = np.zeros(N)
    for i in range(N):
        for j in range(M):
            if j < N:
                if j == i:
                    continue  # don't sense yourself among the robot entries
                if not robot_coupling:
                    continue  # baseline: no robot-robot sensing at all
            dx = x[i] - other_x[j]
            dy = y[i] - other_y[j]
            dvx = vx[i] - other_vx[j]
            dvy = vy[i] - other_vy[j]
            d = (dx * dx + dy * dy) ** 0.5
            d_safe = max(d, 1e-3)
            closing_rate = -(dx * dvx + dy * dvy) / d_safe
            R_eff = R + tau[i] * max(closing_rate, 0.0)
            if d < R_eff:
                if exponential:
                    mag = K[i] * (np.exp((R_eff - d) / lam[i]) - 1.0) / d_safe
                else:
                    mag = K[i] * (R_eff - d) / d_safe
                fx[i] += mag * dx
                fy[i] += mag * dy
    return fx, fy


@njit(cache=True)
def wall_repulsion(x, y, vx, vy, size, K, R, tau, lam, exponential):
    # a wall is treated exactly like any other sensed point now: same
    # force law as proximity_repulsion (zero outside R, R_eff extended by
    # tau[i]*closing_rate while approaching, same linear/exponential switch
    # - see its docstring), same K/R/tau/lam per robot - not a separate
    # wall-specific gain or curve, so a robot reacts to an approaching wall
    # no differently than an approaching robot or the walker (see module
    # docstring). The edges are axis-aligned planes, not points, so the
    # push direction is just +/-x or +/-y and the closing rate is just
    # -vx[i]/+vx[i] (or the y equivalent) rather than proximity_repulsion's
    # general dot-product form - no need to normalize a direction vector or
    # track a moving "other" point, since a wall never moves and the
    # nearest point on it is always directly ahead along one axis.
    N = x.shape[0]
    fx = np.zeros(N)
    fy = np.zeros(N)
    for i in range(N):
        d = x[i]
        R_eff = R + tau[i] * max(-vx[i], 0.0)
        if d < R_eff:
            if exponential:
                fx[i] += K[i] * (np.exp((R_eff - d) / lam[i]) - 1.0)
            else:
                fx[i] += K[i] * (R_eff - d)
        d = size - x[i]
        R_eff = R + tau[i] * max(vx[i], 0.0)
        if d < R_eff:
            if exponential:
                fx[i] -= K[i] * (np.exp((R_eff - d) / lam[i]) - 1.0)
            else:
                fx[i] -= K[i] * (R_eff - d)
        d = y[i]
        R_eff = R + tau[i] * max(-vy[i], 0.0)
        if d < R_eff:
            if exponential:
                fy[i] += K[i] * (np.exp((R_eff - d) / lam[i]) - 1.0)
            else:
                fy[i] += K[i] * (R_eff - d)
        d = size - y[i]
        R_eff = R + tau[i] * max(vy[i], 0.0)
        if d < R_eff:
            if exponential:
                fy[i] -= K[i] * (np.exp((R_eff - d) / lam[i]) - 1.0)
            else:
                fy[i] -= K[i] * (R_eff - d)
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
def resolve_hard_collisions(x, y, anchor, size, robot_radius, x_s, y_s, walker_min_dist, walker_active, n_iters=4):
    # hard limit on top of wall_repulsion/proximity_repulsion's soft push:
    # those forces make getting close expensive, but a
    # strong enough combination of other forces in one tick can still beat
    # them and end the tick overlapping. This is a direct position
    # correction run after integration - a robot's centre can never end a
    # tick closer than robot_radius to the arena edge, 2*robot_radius to
    # another robot's centre, or walker_min_dist to the walker, full stop.
    # Anchored robots are treated as immovable obstacles (matching
    # _step_core, which never moves them either) - only the non-anchored
    # side of a pair gets pushed; the walker is always treated this way too
    # (it has its own independent motion model, not something this
    # function should perturb).
    #
    # The walker guarantee used to not exist here at all (the assumption
    # was that proximity_repulsion's soft push would be enough on its own).
    # It isn't: with force summed over every sensed point, crowd pressure
    # regularly outweighs the walker's contribution (see module docstring),
    # and empirically a robot could end up ~0.002 from the walker's exact
    # position - ~24x closer than robots ever get to each other under the
    # equivalent robot-robot limit. This closes that gap the same way the
    # wall/robot cases are already handled.
    #
    # Resolving one overlap can reopen another (pushing i away from j may
    # put it back inside k, back past a wall, or into the walker), so this
    # sweeps a few times rather than once - a simple Gauss-Seidel-style
    # relaxation, not an exact simultaneous solve. Ends with one extra wall
    # clamp on its own: without it, the loop's last operation would always
    # be a pairwise/walker push, which can walk a robot back past the edge
    # with nothing left afterward to catch it.
    # walker_active=False skips this walker-distance guarantee entirely -
    # when robots can't sense the walker (see _step_core/module docstring),
    # there's nothing to keep a hard minimum distance from either; it's as
    # if it isn't in the room.
    N = x.shape[0]
    for _ in range(n_iters):
        _clamp_to_walls(x, y, anchor, size, robot_radius)

        if walker_active:
            for i in range(N):
                if anchor[i]:
                    continue
                dx = x[i] - x_s
                dy = y[i] - y_s
                d = (dx * dx + dy * dy) ** 0.5
                if d < walker_min_dist:
                    if d < 1e-9:
                        dx, dy, d = 1.0, 0.0, 1e-9
                    overlap = walker_min_dist - d
                    x[i] += overlap * dx / d
                    y[i] += overlap * dy / d

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
                        dx, dy, d = 1.0, 0.0, 1e-9  # coincident centres: nudge apart arbitrarily
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
    # each robot's own pull back toward its starting position - not a
    # reaction to a neighbour or the source, just the real DotNode's
    # dead-reckoning drift correction, so this stays a Hookean spring
    # rather than a local-sensor interaction
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
def _step_core(x, y, theta, vx, vy, x0, y0, K_avoid, R_avoid, tau_avoid, lam_avoid, exponential_repulsion,
               K_self, BETA, M, anchor,
               x_s, y_s, vx_s, vy_s, size,
               robot_radius, walker_min_dist, dt, max_speed, hard_collisions_enabled, walker_active,
               robot_coupling):
    N = x.shape[0]
    # every sensed point in one array - the N robots themselves, then the
    # walker (only when walker_active - see module docstring) - so
    # proximity_repulsion has no way to single the walker out. With
    # walker_active=False the walker's slot is left out entirely: robots
    # can't sense it at all, same as if it weren't in the room (its own
    # random-walk position keeps advancing via update_source regardless -
    # see step() - so it re-enters smoothly, not via a teleport, whenever
    # walker_active flips back to True).
    M_other = N + 1 if walker_active else N
    other_x = np.empty(M_other)
    other_y = np.empty(M_other)
    other_vx = np.empty(M_other)
    other_vy = np.empty(M_other)
    other_x[:N] = x
    other_y[:N] = y
    other_vx[:N] = vx
    other_vy[:N] = vy
    if walker_active:
        other_x[N] = x_s
        other_y[N] = y_s
        other_vx[N] = vx_s
        other_vy[N] = vy_s

    Px, Py = proximity_repulsion(x, y, vx, vy, other_x, other_y, other_vx, other_vy,
                                  K_avoid, R_avoid, tau_avoid, lam_avoid, exponential_repulsion, robot_coupling)
    Hx, Hy = home_spring(x, y, x0, y0, K_self)
    Wx, Wy = wall_repulsion(x, y, vx, vy, size, K_avoid, R_avoid, tau_avoid, lam_avoid, exponential_repulsion)

    Fx = Px + Hx + Wx
    Fy = Py + Hy + Wy

    x_new = np.empty(N)
    y_new = np.empty(N)
    vx_new = np.empty(N)
    vy_new = np.empty(N)
    theta_new = np.empty(N)
    for i in range(N):
        # plain damped-driven point mass in world x/y - no heading
        # projection, so a push from any direction moves the robot in
        # exactly that direction (see module docstring).
        dvx = (Fx[i] - BETA[i] * vx[i]) / M
        dvx = min(max(dvx, -4.0), 4.0)
        vxn = vx[i] + dvx * dt

        dvy = (Fy[i] - BETA[i] * vy[i]) / M
        dvy = min(max(dvy, -4.0), 4.0)
        vyn = vy[i] + dvy * dt

        # cap speed magnitude (isotropic), not per-axis, so the cap doesn't
        # bias direction
        speed = (vxn * vxn + vyn * vyn) ** 0.5
        if speed > max_speed:
            scale = max_speed / speed
            vxn *= scale
            vyn *= scale

        # anchors don't move (real node skips its whole update() when self.anchor)
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
            # theta is cosmetic now (see module docstring) - just point it
            # along the direction of travel; hold the last heading rather
            # than snapping to an arbitrary angle while nearly stationary
            if speed > 1e-6:
                theta_new[i] = np.arctan2(vyn, vxn)
            else:
                theta_new[i] = theta[i]
    if hard_collisions_enabled:
        x_new, y_new = resolve_hard_collisions(x_new, y_new, anchor, size, robot_radius, x_s, y_s, walker_min_dist, walker_active)
    return x_new, y_new, theta_new, vx_new, vy_new


def step(state, t, T):
    # thin wrapper: unpacks state, runs the numba-compiled core, writes the
    # result back - update_source() stays plain Python since it's cheap and
    # touches numpy's global RNG (numba has its own separate RNG state)
    x_s, y_s = update_source(state, t, T)
    x_new, y_new, theta_new, vx_new, vy_new = _step_core(
        state['x'], state['y'], state['theta'], state['vx'], state['vy'],
        state['x0'], state['y0'], state['K_avoid'], state['R_avoid'], state['tau_avoid'],
        state['lam_avoid'], state['exponential_repulsion'],
        state['K_self'], state['BETA'], state['M'], state['anchor'],
        x_s, y_s, state['source_vx'], state['source_vy'],
        state['size'],
        state['robot_radius'], state['walker_min_dist'], DT, MAX_SPEED,
        state['hard_collisions_enabled'], state['walker_active'], state['robot_coupling'])
    state['x'] = x_new
    state['y'] = y_new
    state['theta'] = theta_new
    state['vx'] = vx_new
    state['vy'] = vy_new

#%% SETUP

N = 10       # number of robots
size = 1
T = 70     # period of the global input signal

INIT_LAYOUT = 'random'   # 'random', 'grid', or 'circle'
INIT_MIN_DIST = 0.1 * size   # 'random' layout only: minimum spacing between
                              # robots' start positions (None = unconstrained)
x0, y0 = init_positions(INIT_LAYOUT, N, size, min_dist=INIT_MIN_DIST)
theta = np.random.uniform(0, 2*np.pi, N)  # cosmetic initial facing only -
                                           # tracks direction of travel once
                                           # moving (see module docstring)

# NOTE: BETA_LOW/BETA_HIGH below, plus AVOID_K_SCALE/AVOID_R_FRAC/
# SOURCE_STEP further down, are the best trial (#338, test_nmse=0.1569)
# from param_optimisation_local_sensors.py's Bayesian search against this
# exact model (holonomic motion, universal linear/Hookean repulsion across
# robots/walker/walls - EXPONENTIAL_REPULSION=False here to match). That
# search runs with K_self/home_spring disabled entirely (not just scaled to
# 0), so K_SELF_SCALE stays 0 here too for a fair comparison - see that
# file's module docstring.
K_SELF_SCALE = 0 #4.842140376051767

K_self = (7.8788 + np.random.uniform(size=N) * 5.0) * 0.3 * K_SELF_SCALE

BETA_LOW = 8.661317527979818
BETA_HIGH = 14.938739283334057
BETA = np.random.uniform(BETA_LOW, BETA_HIGH, size=N)
M = 1.0  # DotNode overrides M to 1 regardless of launch params

# repulsion-only avoidance - no rest length, no attraction - shared by
# proximity_repulsion (every sensed point: other robots and the walker
# alike, indistinguishably - see module docstring) and wall_repulsion (the
# arena edge, using the exact same K/R/tau and the same linear force law -
# see wall_repulsion's docstring). One K/R/tau dial rather than several
# that can drift out of sync - it's all the same kind of avoidance, just
# against different kinds of "object". K_self (home_spring, above) stays
# independent: it's not collision avoidance, it's the real DotNode's own
# dead-reckoning drift correction back to its start position.
AVOID_K_SCALE = 0.105 #0.8624863785153422
AVOID_R_FRAC = 0.19363465925641732
AVOID_K = 6.0 * AVOID_K_SCALE
AVOID_R = AVOID_R_FRAC * size

# how far ahead (in simulated time) proximity_repulsion "leads" a closing
# point: effective sensing radius = R + LOOM_TAU * closing_rate while
# something's approaching (see proximity_repulsion's docstring) - this is
# what lets the walker make an outsized impression without being labeled
# as special, since it's normally the only thing around moving with real,
# sustained velocity. Untuned first guess (not yet part of
# param_optimisation_local_sensors.py's search) - if this doesn't produce
# a visible effect, that's the first knob to check.
LOOM_TAU = 0.0

# force-law switch, applied identically to proximity_repulsion (robots and
# the walker) and wall_repulsion (see both docstrings): False = linear/
# Hookean (magnitude K*(R_eff-d)); True = exponential-in-penetration
# (K*(exp((R_eff-d)/lam)-1)) - zero at the same sensing radius R either
# way, but softer while still far from contact and much harder right at
# it, with the decay length lam set by REPULSION_STEEPNESS below. Flip
# back to False to compare directly against the linear behaviour.
EXPONENTIAL_REPULSION = False

# whether robots sense/repel each other at all (they always still sense the
# walker regardless - see proximity_repulsion's docstring). False turns the
# swarm into N independently-reacting sensors with no coupling between them
# - a baseline for checking whether inter-robot interaction itself adds
# anything decodable beyond what N non-interacting sensors already give the
# readout. Overridable per-call via simulation(robot_coupling=...) without
# touching this default - see its docstring.
ROBOT_COUPLING_ENABLED = True

# only used when EXPONENTIAL_REPULSION is True: each robot's exponential
# decay length lam = R / steepness, so the exponential force is exactly 0
# at the sensing radius R and rockets up over the last 1/steepness of that
# radius - higher steepness = softer until very close, then a harder push
# right at contact.
REPULSION_STEEPNESS = 1.3721232915314543

# each robot gets its own K/steepness/tau rather than sharing one value -
# a uniform random multiplier around AVOID_K/REPULSION_STEEPNESS/LOOM_TAU,
# drawn once at setup (same idea as K_self/BETA varying per robot above:
# this is each robot's own sensor/controller gain, not a property of what
# it's reacting to). ROBOT_PARAM_SPREAD=0.5 means each robot's K/steepness/
# tau land uniformly in [0.5x, 1.5x] the tuned central value. R (the
# sensing radius) stays a shared scalar - not varied per robot here.
# Applies identically to robots, the walker, and walls (see
# wall_repulsion's docstring) - one set of per-robot gains, not one per
# obstacle type.
ROBOT_PARAM_SPREAD = 0.5
K_PER_ROBOT = AVOID_K * np.random.uniform(1 - ROBOT_PARAM_SPREAD, 1 + ROBOT_PARAM_SPREAD, N)
STEEPNESS_PER_ROBOT = REPULSION_STEEPNESS * np.random.uniform(
    1 - ROBOT_PARAM_SPREAD, 1 + ROBOT_PARAM_SPREAD, N)
LAMBDA_PER_ROBOT = AVOID_R / STEEPNESS_PER_ROBOT
TAU_PER_ROBOT = LOOM_TAU * np.random.uniform(1 - ROBOT_PARAM_SPREAD, 1 + ROBOT_PARAM_SPREAD, N)

K_AVOID = K_PER_ROBOT
R_AVOID = AVOID_R
TAU_AVOID = TAU_PER_ROBOT
LAMBDA_AVOID = LAMBDA_PER_ROBOT

# hard collision limit (see resolve_hard_collisions): after integration,
# robots physically cannot end a tick closer than ROBOT_RADIUS to the arena
# edge, 2*ROBOT_RADIUS to another robot's centre, or WALKER_MIN_DIST to the
# walker - guaranteed regardless of how the soft forces above played out
# that tick (proximity_repulsion's force is summed over every sensed
# point, so crowd pressure can and does overwhelm the walker's own
# contribution - see module docstring).
ROBOT_RADIUS = 0.02 * size

# set False to pause resolve_hard_collisions entirely - robots then rely on
# the soft forces (proximity_repulsion/wall_repulsion) alone and can overlap
# each other, the walls, or the walker
HARD_COLLISIONS_ENABLED = True

# rough guess at a person's footprint - bigger than a robot's own
# ROBOT_RADIUS since a walker is physically much larger than one of these
# robots, but not calibrated to anything real; easy to adjust
WALKER_RADIUS = 0.05 * size
WALKER_MIN_DIST = ROBOT_RADIUS + WALKER_RADIUS

SOURCE_MODE = 'random_walk_wall_avoidance'   # 'circle' or 'random_walk' or 'random_walk_wall_avoidance
SOURCE_STEP = 0.2 * size    # random_walk only: velocity-innovation scale per tick
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

def walker_removal_mask(n_steps, removal_start, removal_steps):
    # builds the walker_active_mask for the "train with the walker in,
    # remove it and let the swarm settle back toward home_spring
    # equilibrium, then bring it back" experiment: True for
    # [0, removal_start), False for [removal_start, removal_start +
    # removal_steps), True again afterward. Feed the True/False phases to
    # data_processing.calc_nmse_transfer as the train/transfer-test split -
    # fit a readout on the first True phase only, then check whether it can
    # still decode the walker's position in the second True phase with no
    # further fitting.
    mask = np.ones(n_steps, dtype=bool)
    mask[removal_start:removal_start + removal_steps] = False
    return mask


def simulation(show=False, coord_frame='global', walker_active_mask=None, robot_coupling=None):
    # coord_frame='local' stores each robot's displacement from its own
    # starting position (x - x0, y - y0) in data_states instead of absolute
    # arena coordinates - removes the arbitrary "where did this robot
    # happen to start" offset, so the readout sees relative motion instead.
    # Only affects what's stored/returned; the animation (if show=True)
    # always uses true absolute positions, since a "local" plot wouldn't be
    # spatially meaningful.
    #
    # walker_active_mask: optional bool array, one entry per tick (length
    # num_iterations), controlling whether robots can sense the walker that
    # tick (see _step_core/resolve_hard_collisions's walker_active param
    # and the module docstring). None (default) means "always active" -
    # identical to every run before this option existed. Use
    # walker_removal_mask() below to build the common "train with the
    # walker in, remove it and let the swarm settle, then bring it back"
    # schedule for testing whether a readout trained only on the first
    # phase can still track the walker's position after it returns with no
    # further training.
    #
    # robot_coupling: optional override for ROBOT_COUPLING_ENABLED (see its
    # docstring) - None (default) uses that module-level setting; pass
    # True/False explicitly to run one condition without touching the
    # other's default, e.g. for an interacting-vs-independent-sensors A/B
    # comparison in the same session.
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
        'x': x0.copy(), 'y': y0.copy(), 'theta': theta.copy(),
        'vx': np.zeros(N), 'vy': np.zeros(N),
        'x0': x0, 'y0': y0, 'K_avoid': K_AVOID, 'R_avoid': R_AVOID, 'tau_avoid': TAU_AVOID,
        'lam_avoid': LAMBDA_AVOID, 'exponential_repulsion': EXPONENTIAL_REPULSION,
        'K_self': K_self,
        'BETA': BETA, 'M': M, 'anchor': anchor,
        'size': size,
        'robot_radius': ROBOT_RADIUS,
        'walker_min_dist': WALKER_MIN_DIST, 'hard_collisions_enabled': HARD_COLLISIONS_ENABLED,
        'walker_active': True,
        'robot_coupling': ROBOT_COUPLING_ENABLED if robot_coupling is None else robot_coupling,
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

    if walker_active_mask is not None and len(walker_active_mask) != n_steps:
        raise ValueError(
            f"walker_active_mask has {len(walker_active_mask)} entries, expected {n_steps} (one per tick)")

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
    walker_active_log = np.empty(n_steps, dtype=bool)

    for i, t in enumerate(iterations):
        if walker_active_mask is not None:
            state['walker_active'] = bool(walker_active_mask[i])
        step(state, t, T)
        x_coords[:, i] = state['x']
        y_coords[:, i] = state['y']
        theta_coords[:, i] = state['theta']
        s_array[:, i] = np.hypot(state['vx'], state['vy'])  # speed magnitude, holonomic motion has no signed forward speed
        source_x[i] = state['source_x']
        source_y[i] = state['source_y']
        walker_active_log[i] = state['walker_active']

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
                               source_radius=state['R_avoid'], robot_radius=state['robot_radius'])
    else:
        ani = None

    source_data = np.array([source_x, source_y])
    return data_states, ani, source_data, walker_active_log

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


def replay(data_states, source_data, N, R_avoid, robot_radius=None, max_frames=500, from_start=True):
    # re-render the animation for an already-computed run (still in memory,
    # or reloaded from node-simulation.npz) without resimulating
    x_coords, y_coords, theta_coords, _ = unpack_data_states(data_states, N)
    source_x, source_y = source_data[0], source_data[1]
    return plots.animation(x_coords, y_coords, theta_coords,
                            source_x=source_x, source_y=source_y,
                            source_radius=R_avoid, robot_radius=robot_radius, max_frames=max_frames,
                            from_start=from_start)

#%% RUN + ANIMATE

T = 50

num_iterations = 5000

data_states, ani, source_data, walker_active_log = simulation(show=False, coord_frame='global', robot_coupling=True)
#display(HTML(ani.to_jshtml()))

#%%
ani = replay(data_states, source_data, N, R_AVOID)
display(HTML(ani.to_jshtml()))

#ani.save("animation.gif", writer="pillow", fps=10)

#%%
plots.source_path(source_data[0], source_data[1], size=size)

#%%
filename = 'node-simulation.npz'

# reuse the run from RUN + ANIMATE above rather than re-simulating
np.savez(filename, data_states=data_states,
         source_x=source_data[0], source_y=source_data[1], T=T,
         walker_active=walker_active_log)


#%%

lr, nmses, ys, Xs, predictions = data_processing.calc_nmse(source_data, lag=0, plot=True, alpha = 300)
#data_processing.plot_coefficients(lr)

# %%

y_train, y_test = ys
prediction_train, prediction_test = predictions

data_processing.plot_predictions(y_test[:2000], prediction_test[:2000])
data_processing.plot_trajectory_2d(y_test[100:500], prediction_test[100:500], size=size)

#%%

data_processing.plot_predictions(y_train[:2000], prediction_train[:2000])
data_processing.plot_trajectory_2d(y_train[1000:1200], prediction_train[1000:1200], size=size)


# %% output weightings for linear readout

plots.weight_heatmap(lr, x0, y0, size=size)

data_processing.plot_weight_matrix(lr)

# %%

ani = data_processing.animate_trajectory_2d(y_test, prediction_test, size=size)
display(HTML(ani.to_jshtml()))

# %% AFTER A PAUSE
#
# Trains a readout only while the walker is present, then removes it (robots
# can no longer sense it - see walker_active in _step_core) and lets the
# swarm relax under home_spring/mutual repulsion alone, then brings the
# walker back and checks whether the *same, un-refit* readout can still
# track its position - i.e. no further training after it returns.
# calc_nmse_transfer (data_processing.py) does the fit-once/evaluate-later
# split; see its docstring for exactly which samples go where. This is a
# separate run from everything above (its own simulation() call, its own
# npz) - doesn't touch or depend on data_states/lr/etc. from the cells above.

REMOVAL_START = int(0.6 * num_iterations)   # walker present for the first 60% of the run
REMOVAL_STEPS = int(0.1 * num_iterations)   # then absent for 10%, to let the swarm settle
mask = walker_removal_mask(num_iterations, REMOVAL_START, REMOVAL_STEPS)

(removal_data_states, removal_ani, removal_source_data,
 removal_walker_active) = simulation(show=False, coord_frame='global', walker_active_mask=mask)

np.savez('node-simulation-removal.npz', data_states=removal_data_states,
         source_x=removal_source_data[0], source_y=removal_source_data[1], T=T,
         walker_active=removal_walker_active)

transfer = data_processing.calc_nmse_transfer(
    removal_source_data, removal_walker_active, filename='node-simulation-removal.npz', plot=True)

#%%

data_processing.plot_trajectory_2d(transfer['y_transfer'][:500], transfer['prediction_transfer'][100:2000], size=size)

#%%

print(f"transient NMSE (first {len(transfer['y_transient'])} samples after reintroduction): {transfer['transient_nmse']:.6f}")
print(f"steady NMSE (remaining {len(transfer['y_steady'])} samples):                          {transfer['steady_nmse']:.6f}")

data_processing.plot_trajectory_2d(transfer['y_transient'], transfer['prediction_transient'], size=size)
data_processing.plot_trajectory_2d(transfer['y_steady'][:500], transfer['prediction_steady'][:500], size=size)


# %% BASELINE: does robot-robot interaction actually help decoding, or would
# N independent sensors do just as well?
#
# Runs the exact same model/readout pipeline twice, changing only
# robot_coupling (see proximity_repulsion/simulation()'s docstrings):
#   - interacting: the normal swarm - robots sense the walker AND each
#     other, so a robot can carry information about a neighbour's reaction
#     to the walker, not just its own.
#   - independent: robots still move (still react to the walker, home_spring,
#     walls - same forces otherwise), but robot_coupling=False strips out
#     the robot-robot term in proximity_repulsion, so each one only ever
#     reacts to the walker directly. N separate single-sensor readouts,
#     mechanically incapable of taking advantage of any coupling.
#
# Same alpha for both (300, matching the calc_nmse call above) so the
# comparison isn't confounded by a different regularization search.
COUPLING_ALPHA = 300

interacting_data_states, _, interacting_source_data, _ = simulation(
    show=False, coord_frame='global', robot_coupling=True)
np.savez('node-simulation-interacting.npz', data_states=interacting_data_states,
         source_x=interacting_source_data[0], source_y=interacting_source_data[1], T=T)
_, interacting_nmses, _, _, _ = data_processing.calc_nmse(
    interacting_source_data, alpha=COUPLING_ALPHA, filename='node-simulation-interacting.npz')

independent_data_states, _, independent_source_data, _ = simulation(
    show=False, coord_frame='global', robot_coupling=False)
np.savez('node-simulation-independent.npz', data_states=independent_data_states,
         source_x=independent_source_data[0], source_y=independent_source_data[1], T=T)
_, independent_nmses, _, _, _ = data_processing.calc_nmse(
    independent_source_data, alpha=COUPLING_ALPHA, filename='node-simulation-independent.npz')

print(f"\ninteracting swarm  - train NMSE: {interacting_nmses[0]:.6f}  test NMSE: {interacting_nmses[1]:.6f}")
print(f"independent sensors - train NMSE: {independent_nmses[0]:.6f}  test NMSE: {independent_nmses[1]:.6f}")

# %%

#%%
import numpy as np
import plots
from IPython.display import HTML
import importlib
importlib.reload(plots)
import os 
#%% Global constants - mirrors the real DotNode control law (LEDs stripped out),
# but centralised into one vectorised update instead of N independent ROS nodes.
DT = 0.1
MAX_SPEED = 0.35


def global_input(t, T):
    # stand-in for the /global_input topic broadcast to every robot each tick
    f1, f2, f3 = 2.11, 3.73, 4.33
    return 0.5 * np.sin(2*np.pi*f1*t/T) * np.sin(2*np.pi*f2*t/T) * np.sin(2*np.pi*f3*t/T)


def Du(x, y, x0, y0, K, A, K_self):
    # K[i, j] is the stiffness robot i perceives towards robot j. In the real
    # node this comes from deterministic_k(), which despite its name draws a
    # fresh random value on every call - so K is NOT symmetric here either,
    # each side of an edge can honestly disagree about how stiff it is.
    dx = x[:, None] - x[None, :]   # dx[i, j] = x_i - x_j
    dy = y[:, None] - y[None, :]
    d = np.hypot(dx, dy)
    np.fill_diagonal(d, 1.0)  # avoid /0 on the diagonal; dx/dy are 0 there anyway

    energy_x = K * (A - d) * dx / d
    energy_y = K * (A - d) * dy / d
    sum_x = energy_x.sum(axis=1)
    sum_y = energy_y.sum(axis=1)

    # spring back to each robot's own starting position
    dx0, dy0 = x - x0, y - y0
    d0 = np.hypot(dx0, dy0)
    mask = d0 > 1e-3
    sum_x += np.where(mask, -0.5 * K_self * dx0, 0.0)
    sum_y += np.where(mask, -0.5 * K_self * dy0, 0.0)

    return sum_x, sum_y


def step(state, t, T):
    x, y, theta, s = state['x'], state['y'], state['theta'], state['s']

    Dx, Dy = Du(x, y, state['x0'], state['y0'], state['K'], state['A'], state['K_self'])
    energy_vector = Dx * np.cos(theta) + Dy * np.sin(theta)
    f_s = state['INPUT'] * global_input(t, T)

    ds = (energy_vector + f_s - state['BETA'] * s) / state['M']
    s_new = s + np.clip(ds, -4, 4) * DT
    s_new = np.clip(s_new, -MAX_SPEED, MAX_SPEED)

    x_new = x + DT * np.cos(theta) * s_new
    y_new = y + DT * np.sin(theta) * s_new

    # anchors don't move (real node skips its whole update() when self.anchor)
    anchor = state['anchor']
    state['x'] = np.where(anchor, x, x_new)
    state['y'] = np.where(anchor, y, y_new)
    state['s'] = np.where(anchor, s, s_new)

#%% SETUP

N = 10       # number of robots
size = 1
T = 70     # period of the global input signal

x0 = np.random.uniform(0, size, N)
y0 = np.random.uniform(0, size, N)
theta = np.random.uniform(0, 2*np.pi, N)  # fixed for the whole run: the real
                                           # node never publishes angular.z

# resting spring length between every pair = their distance at t=0, matching
# neighbours_rest_spring being set on the first neighbour callback
dx0 = x0[:, None] - x0[None, :]
dy0 = y0[:, None] - y0[None, :]
A = np.hypot(dx0, dy0)

K = (7.8788 + np.random.uniform(size=(N, N)) * 5.0) * 0.3
np.fill_diagonal(K, 0.0)
K_self = (7.8788 + np.random.uniform(size=N) * 5.0) * 0.3

BETA = np.random.uniform(1.31989, 2.830454, size=N) * 1.3
M = 1.0  # DotNode overrides M (and J) to 1 regardless of launch params

# a handful of robots are "driven" (mirrors the r35/r21/r37/r10/r20 overrides
# in the real deployment), the rest have INPUT = 0
INPUT = np.zeros(N)
driven_idx = np.random.choice(N, size=5, replace=False)
INPUT[driven_idx] = np.random.uniform(-2.5 * 1.6, 4.5 * 1.6, size=5)

anchor = np.zeros(N, dtype=bool)
# anchor[np.random.randint(N)] = True  # uncomment to freeze one robot in place

num_iterations = 300000

#%% SIMULATION

def simulation(show=False):
    state = {
        'x': x0.copy(), 'y': y0.copy(), 'theta': theta, 's': np.zeros(N),
        'x0': x0, 'y0': y0, 'K': K, 'A': A, 'K_self': K_self,
        'BETA': BETA, 'INPUT': INPUT, 'M': M, 'anchor': anchor,
    }

    x_coords = [[] for _ in range(N)]
    y_coords = [[] for _ in range(N)]
    theta_coords = [[] for _ in range(N)]
    s_array = [[] for _ in range(N)]

    iterations = np.arange(0, num_iterations * DT, DT)

    for t in iterations:
        step(state, t, T)
        for n in range(N):
            x_coords[n].append(state['x'][n])
            y_coords[n].append(state['y'][n])
            theta_coords[n].append(state['theta'][n])
            s_array[n].append(state['s'][n])

    data = np.stack([x_coords, y_coords, theta_coords, s_array])
    data_states = data.reshape(-1, data.shape[2]).T

    if show:
        ani = plots.animation(x_coords, y_coords, theta_coords)
    else:
        ani = None

    return data_states, ani

#%% RUN + ANIMATE

T = 80

data_states, ani = simulation(show=True)
display(HTML(ani.to_jshtml()))

# %%
#%% DATA SETS
filename = 'node-simulation.npz'

if os.path.exists(filename):
    os.remove(filename)
file = {}

# increasing this ratio does not help for some reason
period_ratios = np.arange(60, 80, 10) 
for ratio in period_ratios:
    print(ratio)
    T = ratio
    data_states, ani = simulation()
    file[f'T={ratio}'] = data_states
    np.savez('node-simulation.npz', **file)
# %%


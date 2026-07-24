#%%
# Non-distributed simulation of the dynamics in global-simulation.py (the real-robot ROS2 node),
# ignoring LEDs / ROS entirely. Structured like simulation-swarmy.py.
#
# Two things in global-simulation.py come from outside that file and have no single answer here
# — flagged inline where used:
#   1. Neighbour topology: real robots get an explicit neighbour list from a launch config.
#      Here it defaults to fully-connected, same as simulation-swarmy.py.
#   2. The driving signal `u`: real robots receive it over the /global_input ROS topic.
#      Here it's generated locally with the same triple-sine u(t) used in data-processing.py /
#      simulation-swarmy.py, since nothing in this repo defines what publishes /global_input.
#
# Everything else (M, J, BETA, INPUT, K, the -0.5 anchor-spring factor, MAX_SPEED, DT, and the
# absence of angular/theta dynamics) mirrors global-simulation.py's DotNode as closely as a
# non-distributed sim allows. See global-simulation.py for the source of truth.

import numpy as np
from numpy import random
import os
import plots
from IPython.display import HTML
import importlib
importlib.reload(plots)

#%% Constants (matching global-simulation.py)

MAX_SPEED = 0.35
DT = 0.01

BETA_LOW, BETA_HIGH = 1.31989, 2.830454
BETA_SCALE = 1.3

# Observed real-robot INPUT values were hardcoded per named robot (~5 out of ~40 nonzero,
# range roughly -2.5..4.5). Generalised here as: most robots get INPUT=0, a fraction get a
# random nonzero value in that observed range.
INPUT_NONZERO_FRACTION = 0.15
INPUT_LOW, INPUT_HIGH = -2.5, 4.5

K_BASE, K_SPAN, K_SCALE = 7.8788, 5.0, 0.3  # matches deterministic_k() in global-simulation.py


def spring_k():
    # global-simulation.py's deterministic_k() despite its name just draws np.random.uniform()
    # every call. Kept the same distribution here, but drawn ONCE per edge and shared by both
    # endpoints (the real code draws independently per robot, giving asymmetric springs).
    return (K_BASE + np.random.uniform() * K_SPAN) * K_SCALE


def u(t, T):
    # Stand-in for the external /global_input signal (see module docstring).
    f1, f2, f3 = 2.11, 3.73, 4.33
    return 0.5 * np.sin(2 * np.pi * f1 * t / T) * np.sin(2 * np.pi * f2 * t / T) * np.sin(2 * np.pi * f3 * t / T)


#%% Node

class Node:
    dt = DT
    id_array = np.array([])
    A = []          # rest spring length matrix
    K = []          # spring stiffness matrix (symmetric)
    all_nodes = []
    T = 25

    def __init__(self, id, x, y, theta, s, beta, input_gain):
        self.id = id
        self.matrix_row = int(np.where(Node.id_array == id)[0][0])

        self.x = x
        self.y = y
        self.x_origin = x
        self.y_origin = y

        self.theta = theta  # fixed: global-simulation.py never drives theta (comes from real odometry)
        self.s = s

        self.M = 1
        self.J = 1  # unused (no angular dynamics), kept for parity with the real node
        self.beta = beta
        self.input_gain = input_gain

    def f_s(self, t):
        return self.input_gain * u(t, Node.T)

    def Du(self):
        sum_x, sum_y = 0.0, 0.0
        for node in Node.all_nodes:
            if node is self:
                continue
            Kij = Node.K[self.matrix_row, node.matrix_row]
            Aij = Node.A[self.matrix_row, node.matrix_row]
            dx, dy = self.x - node.x, self.y - node.y
            d = np.hypot(dx, dy)
            if d < 1e-3:
                continue
            sum_x += Kij * (Aij - d) * dx / d
            sum_y += Kij * (Aij - d) * dy / d

        # spring back to own starting position (own-name entry in SPRING_K on the real robot)
        Kii = Node.K[self.matrix_row, self.matrix_row]
        dx, dy = self.x - self.x_origin, self.y - self.y_origin
        d = np.hypot(dx, dy)
        if d > 1e-3:
            sum_x += -0.5 * Kii * dx
            sum_y += -0.5 * Kii * dy

        return sum_x, sum_y

    def update(self, t):
        Dx, Dy = self.Du()
        energy_vector = Dx * np.cos(self.theta) + Dy * np.sin(self.theta)

        ds = (energy_vector + self.f_s(t) - self.beta * self.s) / self.M
        self.s += np.clip(ds, -4, 4) * Node.dt
        self.s = np.clip(self.s, -MAX_SPEED, MAX_SPEED)

        self.x += Node.dt * np.cos(self.theta) * self.s
        self.y += Node.dt * np.sin(self.theta) * self.s


#%% SETUP

N = 9
size = 15

ids = np.arange(1, N + 1)

x_array = np.random.uniform(0, size, N)
y_array = np.random.uniform(0, size, N)
theta_array = np.random.uniform(0, 2 * np.pi, N)

X, Y = np.meshgrid(x_array, y_array)
A = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)
np.fill_diagonal(A, 0)

# symmetric spring matrix, one draw per unordered pair (including self-anchor entries on the diagonal)
K = np.zeros((N, N))
for i in range(N):
    for j in range(i, N):
        k = spring_k()
        K[i, j] = k
        K[j, i] = k

Node.id_array = ids
Node.A = A
Node.K = K

input_mask = np.random.rand(N) < INPUT_NONZERO_FRACTION
input_gains = np.where(input_mask, np.random.uniform(INPUT_LOW, INPUT_HIGH, size=N), 0.0)

beta_values = np.random.uniform(BETA_LOW, BETA_HIGH, size=N) * BETA_SCALE

num_iterations = 30000

#%% SIMULATION

def simulation(show=False):
    Node.all_nodes = []
    for i, id in enumerate(ids):
        node = Node(id=id, x=x_array[i], y=y_array[i], theta=theta_array[i],
                    s=0.0, beta=beta_values[i], input_gain=input_gains[i])
        Node.all_nodes.append(node)

    x_coords = [[] for _ in range(N)]
    y_coords = [[] for _ in range(N)]
    theta_coords = [[] for _ in range(N)]
    s_array = [[] for _ in range(N)]

    step = 0.01
    iterations = np.arange(0, num_iterations * step, step)

    for it in iterations:
        for n, node in enumerate(Node.all_nodes):
            node.update(it)
            x_coords[n].append(node.x)
            y_coords[n].append(node.y)
            theta_coords[n].append(node.theta)
            s_array[n].append(node.s)

    data = np.stack([x_coords, y_coords, theta_coords, s_array])
    data_states = data.reshape(-1, data.shape[2]).T

    ani = plots.animation(x_coords, y_coords, theta_coords) if show else None
    return data_states, ani


#%% DATA SETS

filename = 'global-node-simulation.npz'
if os.path.exists(filename):
    os.remove(filename)
file = {}

period_ratios = np.arange(10, 42.5, 2.5)
for ratio in period_ratios:
    print(ratio)
    Node.T = ratio
    data_states, _ = simulation()
    file[f'T={ratio}'] = data_states
    np.savez(filename, **file)

#%% END OF SIMULATION

data_states, ani = simulation(show=True)
display(HTML(ani.to_jshtml()))

# %%
ani.save("global-animation.gif", writer='pillow', fps=10)

# %%

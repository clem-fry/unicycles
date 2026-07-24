#%%
import numpy as np
from numpy import random
import pandas as pd
import os
import plots
from IPython.display import HTML
import importlib
importlib.reload(plots)
from matplotlib.animation import PillowWriter
import random 

#%% 

MAX_SPEED = 0.3
MAX_ANG = 1.0

class Node():
    N = []
    dt = 0.01

    # robot would need a matrix of ids to know whose who?
    id_array = np.array([])
    # alternatively each robot could have a different impression of the springs ?
    #adjacency_matrix = [] # existance of connections - corresponding to id list?
    A = [] # spring lengths between agents i and j matrix
    all_nodes = []
    T = 25
    K = []

    def __init__(self, id, x, z, theta, s, w, J, beta, zeta, T_theta, T_s, m):
        self.id = id
        self.matrix_row = int(np.where(Node.id_array == id)[0][0])

        self.x = x
        self.z = z

        self.x_origin = x
        self.z_origin = z

        self.theta = theta
        self.s = s
        self.w = w
        self.J = J # = 1/2 M R^2
        self.beta = beta
        self.zeta = zeta
        self.T_theta = T_theta
        self.T_s = T_s
        self.m = m
        self.K = np.random.rand(len(ids)) * 49.3

        self.connections = []

    def add_connection(self, unicycle_connection):
        if unicycle_connection not in self.connections:
            self.connections.append(unicycle_connection)

    def distance_to(self, other_node):
        if self.x == other_node.x and self.z == other_node.z:
            return 0
        else:
            return np.sqrt((self.x - other_node.x) ** 2 + (self.z - other_node.z) ** 2)

    def update_all_neighbours(self):
        for node in Node.all_nodes:
            if (self.distance_to(node) < Node.threshold) and (node.id != self.id):
                self.add_connection(node)

    def f_theta(self, t):
        f = np.dot(self.T_theta, Node.u(t))
        return f
    
    def f_s(self, t):
        f = np.dot(self.T_s, Node.u(t))
        return f

    def Du(self):
        sum_x = 0
        sum_z = 0
        for node in Node.all_nodes: # including ones-self
            Kij =  Node.K[node.matrix_row, self.matrix_row]
            Aij = Node.A[self.matrix_row, node.matrix_row] # resting spring length
            d = self.distance_to(node) # filter out onesself
            if d > 1e-3:
                energy_x = Kij * (Aij - d) * (self.x - node.x) / d
                energy_z = Kij * (Aij - d) * (self.z - node.z) / d
                sum_x += energy_x
                sum_z += energy_z

        Kij =  Node.K[node.matrix_row, self.matrix_row]
        dx, dz = self.x - self.x_origin, self.z - self.z_origin
        d = np.sqrt(dx**2 + dz**2)
        if d > 1e-3:
            energy_x = -Kij * dx
            energy_z = -Kij * dz

            sum_x += energy_x
            sum_z += energy_z
                
        return sum_x, sum_z
    
    # def repulsion_force(self):
    #     fx, fz = 0, 0
    #     d_min = 5
    #     k_rep = 1
    #     for node in Node.all_nodes:
    #         if node is self:
    #             continue
    #         dx = self.x - node.x
    #         dz = self.z - node.z
    #         d = np.sqrt(dx**2 + dz**2) # distance to node

    #         if d < d_min and d > 0: # when within threshold
    #             f = k_rep * (1/d**2 - 1/d_min**2)
    #             fx += f * dx / d
    #             fz += f * dz / d

    #     return fx, fz

    def update(self, t):

        dw = 1/self.J * (self.f_theta(t) - self.zeta * self.w)

        self.w = self.w + Node.dt * dw
        self.w = np.clip(self.w, -MAX_ANG, MAX_ANG)
        #print('w: ', self.w)

        dtheta = self.w
        self.theta = self.theta + 0.01 * dtheta
        #print('theta: ',self.theta)

        Dudx, Dudz = self.Du()

        #fx_rep, fz_rep = self.repulsion_force()
        ds = 1/self.m * (Dudx * np.cos(self.theta) + 
                         Dudz * np.sin(self.theta) + 
                         self.f_s(t) - self.beta * self.s)
    
        #print(ds * 0.0001)
        self.s = self.s + 0.01 * ds
        self.s = np.clip(self.s, -MAX_SPEED, MAX_SPEED)
        # print('Dudx: ', Dudx)
        # print('Dudz: ', Dudz)
        # print('f_s: ', self.f_s(t))

        # print('s: ',self.s)

        self.x = self.x + Node.dt * (np.cos(self.theta) * self.s)
        self.z = self.z + Node.dt * (np.sin(self.theta) * self.s)

    def u(t): # input signal 
        f1 = 2.11 # frequencies
        f2 = 3.73
        f3 = 4.33
        u = 0.5 * np.sin(2*np.pi * f1 * t / Node.T) * np.sin(2*np.pi * f2 * t / Node.T) * np.sin(2*np.pi * f3 * t / Node.T)
        return u

#%% SETUP

N = 9 # number of nodes: 15 to 20
size = 15

ids = np.arange(1, N + 1)

# locations (triangle)
x_array = np.random.uniform(0, size, N)
z_array = np.random.uniform(0, size, N)

theta_array = np.random.uniform(0, 2*np.pi, N)

X, Z = np.meshgrid(x_array, z_array)
A = np.sqrt((X - X.T)**2 + (Z - Z.T)**2) # starting spring lengths -> beginning

np.fill_diagonal(A, 0)

num_iterations = 30000 # 30000

Node.N = N
Node.A = A
Node.id_array = ids
Node.all_nodes = []
#Node.K = np.random.rand(len(ids)) * 30
n = len(ids)

K = np.zeros((n, n))
upper = np.triu_indices(n)
K[upper] = np.random.rand(len(upper[0])) * 7.0
K[(upper[1], upper[0])] = K[upper]  # mirror

Node.K = K

random_values = np.random.uniform(-11.440195 * 1.6, 4.289848 * 1.6, size = len(ids))
mask = np.random.rand(len(ids)) < 0.7
random_values[mask] = 0
random_inputs = random_values
#%% SIMULATION

def simulation(show=False):
    Node.all_nodes = []
    for i, id in enumerate(ids):
        random_input = random_inputs[i]
        # node = Node(id = id, x = x_array[i], z=z_array[i], 
        #             theta=theta_array[i], s=0, w=0, 
        #             J = 0.5, beta = 3.0, zeta = 0.05, 
        #             T_theta = 10.0 , T_s = 12.0, m = 0.1)

        node = Node(id = id, x = x_array[i], z=z_array[i], 
                    theta=theta_array[i], s=0, w=0, 
                    J = 0.5, beta =  0.3, zeta = 0.05, 
                    T_theta = 1.0 , T_s = random_input, m = 0.1)

        Node.all_nodes.append(node)

    x_coords = [[] for _ in range(N)]
    z_coords = [[] for _ in range(N)]
    theta_coords = [[] for _ in range(N)]
    w_array = [[] for _ in range(N)]
    s_array =  [[] for _ in range(N)]

    step = 0.01
    iterations = np.arange(0, num_iterations * step, step)

    for iter in iterations:
        for n, node in enumerate(Node.all_nodes): # updates all variables - not syncronised between nodes
            Node.update(node, iter)
            x_coords[n].append(node.x)
            z_coords[n].append(node.z)
            theta_coords[n].append(node.theta)
            w_array[n].append(node.w)
            s_array[n].append(node.s)

    data = np.stack([x_coords, z_coords, theta_coords, s_array, w_array])
    data_states = data.reshape(-1, data.shape[2]).T

    if show:
        ani = plots.animation(x_coords, z_coords, theta_coords)
    else:
        ani = None

    return data_states, ani    

#%% DATA SETS
filename = 'node-simulation.npz'

if os.path.exists(filename):
    os.remove(filename)
file = {}

# increasing this ratio does not help for some reason
period_ratios = np.arange(1, 4.25, 0.25) 
period_ratios = np.arange(10, 42.5, 2.5) 
for ratio in period_ratios:
    print(ratio)
    Node.T = ratio
    data_states, ani = simulation()
    file[f'T={ratio}'] = data_states
    np.savez('node-simulation.npz', **file)

#%% END OF SIMULATION
# add to jupyter notebook

data_states, ani = simulation(show = True)
display(HTML(ani.to_jshtml()))

# %%
ani.save("animation.gif", writer='pillow', fps=10)

# %%
# %%

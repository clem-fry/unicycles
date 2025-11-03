#%%
import numpy as np
from numpy import random
import pandas as pd
import os
import plots
from IPython.display import HTML
importlib.reload(plots)
from matplotlib.animation import PillowWriter

#%% 

class Node():

    N = []
    dt = 0.01

    # robot would need a matrix of ids to know whose who?
    id_array = np.array([])
    # alternatively each robot could have a different impression of the springs ?
    #adjacency_matrix = [] # existance of connections - corresponding to id list?
    A = [] # spring lengths between agents i and j matrix
    all_nodes = []
    T = 3

    def __init__(self, id, x, z, theta, s, w, J, beta, zeta, T_theta, T_s, m):
        self.id = id
        self.matrix_row = int(np.where(Node.id_array == id)[0][0])

        self.x = x
        self.z = z
        self.theta = theta
        self.s = s
        self.w = w
        self.J = J # = 1/2 M R^2
        self.beta = beta
        self.zeta = zeta
        self.T_theta = T_theta
        self.T_s = T_s
        self.m = m
        self.K = random.randint(10, 30, size = len(ids)) / 20

        self.connections = []
        self.anchor = False

    def add_connection(self, unicycle_connection):
        if unicycle_connection not in self.connections:
            self.connections.append(unicycle_connection)

    def distance_to(self, other_node):
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

    def Dudx(self):
        sum = 0
        for node in Node.all_nodes:
            Kij =  self.K[node.matrix_row]
            Aij = Node.A[self.matrix_row, node.matrix_row]
            d = self.distance_to(node)
            if d > 0:
                energy = Kij * (Aij - d) * (self.x - node.x) / d
                sum += energy
        return sum

    def Dudz(self):
        sum = 0
        for node in Node.all_nodes:
            Kij =  self.K[node.matrix_row]
            Aij = Node.A[self.matrix_row, node.matrix_row]
            d = self.distance_to(node)
            if d > 0:
                energy = Kij * (Aij - d) * (self.z - node.z) / d
                sum += energy
        return sum
    
    def repulsion_force(self):
        fx, fz = 0, 0
        d_min = 5
        k_rep = 1
        for node in Node.all_nodes:
            if node is self:
                continue
            dx = self.x - node.x
            dz = self.z - node.z
            d = np.sqrt(dx**2 + dz**2) # distance to node

            if d < d_min and d > 0: # when within threshold
                f = k_rep * (1/d**2 - 1/d_min**2)
                fx += f * dx / d
                fz += f * dz / d

        return fx, fz

    def update(self, t):
        if self.anchor:
            return()
        
        dw = 1/self.J * (self.f_theta(t) - self.zeta * self.w)
        self.w = self.w + Node.dt * dw
        #print('w: ', self.w)

        dtheta = self.w
        self.theta = self.theta + Node.dt * dtheta
        #print('theta: ',self.theta)

        #fx_rep, fz_rep = self.repulsion_force()
        ds = 1/self.m * (self.Dudx() * np.cos(self.theta) + 
                         self.Dudz() * np.sin(self.theta) + 
                         self.f_s(t) - self.beta * self.s)
    
        self.s = self.s + Node.dt * ds
        #print('Dudx: ', self.Dudx())
        #print('Dudz: ', self.Dudz())
        #print('f_s: ', self.f_s(t))

        #print('s: ',self.s)

        self.x = self.x + Node.dt * (np.cos(self.theta) * self.s)
        self.z = self.z + Node.dt * (np.sin(self.theta) * self.s)


    def u(t): # input signal 
        f1 = 2.11 # frequencies
        f2 = 3.73
        f3 = 4.33
        u = 0.2 * np.sin(2*np.pi * f1 * t / Node.T) * np.sin(2*np.pi * f2 * t / Node.T) * np.sin(2*np.pi * f3 * t / Node.T)
        return u

#%% SETUP

N = 20 # number of nodes: 15 to 20
size = 10

ids = np.arange(1, N + 1)

anchors = np.zeros(N, dtype=int)
anchor_idx = np.random.randint(N)  # pick one random anchor
anchors[anchor_idx] = 1

# locations (triangle)
x_array = np.random.uniform(0, size, N)
z_array = np.random.uniform(0, size, N)

theta_array = np.random.uniform(0, 2*np.pi, N)

X, Z = np.meshgrid(x_array, z_array)
A = np.sqrt((X - X.T)**2 + (Z - Z.T)**2) # starting spring lengths -> beginning

np.fill_diagonal(A, 0)

iterations = 20000 # 30000
delay = 1 # number of iterations - 2 * dt = 0.02 seconds

Node.N = N
Node.A = A
Node.id_array = ids
Node.all_nodes = []

random_inputs = random.randint(3000, 5000, size = len(ids))

#%% SIMULATION

def simulation(show=False):
    Node.all_nodes = []
    for i, id in enumerate(ids):
        random_input = random_inputs[i]
        node = Node(id = id, x = x_array[i], z=z_array[i], 
                    theta=theta_array[i], s=0, w=0, 
                    J = 1, beta = 0.9, zeta = 0.05, 
                    T_theta = 0 , T_s = random_input, m = 1)
        # J = 2.5
        if anchors[i]:
            node.anchor=True

        Node.all_nodes.append(node)

    x_coords = [[] for _ in range(N)]
    z_coords = [[] for _ in range(N)]
    theta_coords = [[] for _ in range(N)]
    w_array = [[] for _ in range(N)]
    s_array =  [[] for _ in range(N)]

    for iter in range(iterations):
        for n, node in enumerate(Node.all_nodes): # updates all variables - not syncronised between nodes
            Node.update(node, iter // delay)
            if iter % delay == 0:
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
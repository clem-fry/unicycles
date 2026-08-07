#%%
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import plots
from IPython.display import HTML
import importlib
importlib.reload(plots)
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real, Integer

#%% 

min_dist = 1

class Node():

    N = []
    dt = 0.01
    ids = []

    # robot would need a matrix of ids to know whose who?
    id_array = np.array([])
    # alternatively each robot could have a different impression of the springs ?
    #adjacency_matrix = [] # existance of connections - corresponding to id list?
    A = [] # spring lengths between agents i and j matrix
    all_nodes = []
    T = 1
    K = []

    def __init__(self, id, x, z, theta, s, w, J, beta, zeta, T_theta, T_s, m):
        self.id = id
        self.matrix_row = int(np.where(Node.id_array == id)[0][0])

        self.x = x
        self.x_initial = x
        self.dx = 0
        self.z = z
        self.z_initial = z
        self.dz = 0

        self.theta = theta
        self.s = s
        self.w = w
        self.J = J # = 1/2 M R^2
        self.beta = beta
        self.zeta = zeta
        self.T_theta = T_theta
        self.T_s = T_s
        self.m = m

        self.connections = []
        self.anchor = False

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
        for node in Node.all_nodes:
            Kij =  Node.K[node.matrix_row]
            Aij = Node.A[self.matrix_row, node.matrix_row]
            d = self.distance_to(node)
            if d <= 0:
                continue
            # print('Kij: ', Kij)
            # print('self.dx: ', self.dx)
            # print('self.dz: ', self.dz)
            energy_x = Kij * self.dx
            energy_z = Kij * self.dz
            # print('energy_x: ', energy_x)
            # print('energy_z: ', energy_z)
            sum_x += energy_x
            sum_z += energy_z
            # print('sum_x: ', sum_x)
            # print('sum_z: ', sum_z)

        return sum_x, sum_z


    def update(self, t):
        if self.anchor:
            return()
        
        dw = 1/self.J * (self.f_theta(t) - self.zeta * self.w)
        self.w = self.w + Node.dt * dw
        #print('w: ', self.w)

        dtheta = self.w
        self.theta = self.theta + Node.dt * dtheta
        #print('theta: ',self.theta)

        Dudx, Dudz = self.Du()

        #fx_rep, fz_rep = self.repulsion_force()
        ds = 1/self.m * (Dudx * np.cos(self.theta) + 
                         Dudz * np.sin(self.theta) + 
                         self.f_s(t) - self.beta * self.s)
                
        self.s = self.s + Node.dt * ds
        # print('Dudx: ', Dudx)
        # print('Dudz: ', Dudz)
        # print('f_s: ', self.f_s(t))

        # print('s: ',self.s)


        self.x = self.x + Node.dt * (np.cos(self.theta) * self.s)
        self.z = self.z + Node.dt * (np.sin(self.theta) * self.s)
        
        self.dx = self.x_initial - self.x
        self.dz = self.z_initial - self.z

        #print('x: ',self.x)
        #print('z: ',self.z)

    def u(t): # input signal 
        f1 = 2.11 # frequencies
        f2 = 3.73
        f3 = 4.33
        u = 0.2 * np.sin(2*np.pi * f1 * t / Node.T) * np.sin(2*np.pi * f2 * t / Node.T) * np.sin(2*np.pi * f3 * t / Node.T)
        return u

def simulation(N, spring_stiffness, delay, input_size, J, beta, m, T = 1):
    ids = np.arange(1, N + 1)
    Node.ids = ids

    anchors = np.zeros(N, dtype=int)
    anchor_idx = np.random.randint(N)  # pick one random anchor
    anchors[anchor_idx] = 1

    # locations (triangle)
    x_array = np.random.uniform(0, 10, N)
    z_array = np.random.uniform(0, 10, N)

    theta_array = np.random.uniform(0, 2*np.pi, N)

    K = np.ones(N) * spring_stiffness # spring stiffnesses

    X, Z = np.meshgrid(x_array, z_array)
    A = np.sqrt((X - X.T)**2 + (Z - Z.T)**2) # starting spring lengths -> beginning

    np.fill_diagonal(A, 0)

    iterations = 50000

    Node.N = N
    Node.K = K
    Node.A = A
    Node.id_array = ids
    Node.all_nodes = []

    random_inputs = random.randint(0, input_size, size = len(ids))

    Node.all_nodes = []
    for i, id in enumerate(ids):
        random_input = random_inputs[i]
        node = Node(id = id, x = x_array[i], z=z_array[i], 
                    theta=theta_array[i], s=0, w=0, 
                    J = J, beta = beta, zeta = 0.05, 
                    T_theta = 0, T_s = random_input, m = m)
        
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

    y_array = NARAM_2(np.shape(data_states)[0], T)

    cut = int(np.shape(data_states)[0] * 0.1) # cut first 10 percent
    X = data_states[cut:, :]
    #y_array = np.repeat(y_array_rep, 2)

    y = np.array(y_array[cut:])

    #X_train, X_test, y_train, y_test = train_test_split(data_states, y_array, test_size=0.2, random_state=17)
    split_idx = int(0.7 * X.shape[0])
    X_train, X_test = X[:split_idx, :], X[split_idx:, :]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler() # weight all state elements the same
    X_train_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    lr = LinearRegression()
    lr = Ridge(alpha=1e-6)

    lr.fit(X_train_transformed, y_train)

    prediction_test = lr.predict(X_test_transformed)
    def nmse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2) / np.mean((y_true - np.mean(y_true))**2)

    test_nmse  = nmse(y_test[:100], prediction_test[:100])

    return test_nmse

def NARAM_2(time, T): # NARMA_2
    y_array = [0, 0]
    for t in range(1, time):
        y = 0.4 * y_array[t] + 0.4 * y_array[t] * y_array[t-1] + 0.6 * Node.u(t) ** 3 + 0.1
        y_array.append(y)
    return y_array[1:]

def NARAM_5(time, T): # NARMA_2
    y_array = [0, 0, 0, 0]
    n = 5
    for t in range(1, time):
        sumation = np.sum([y_array[t-j] for j in range(0, n-1)])
        y = 0.3 * y_array[t] + 0.05 * sumation + 1.5 * u(t-n+1, T) * u(t, T) + 0.1
        y_array.append(y)
    return y_array[3:]


# %%

def objective(params):
    spring_stiffness, J, friction, mass, input_size = params

    # Round integer-like params
    N = 10
    delay = 1
    input_size = int(input_size)

    try:
        cost = simulation(N, spring_stiffness, delay,
                             input_size, J, beta=friction, m=mass)
        if np.isnan(cost) or cost == np.inf:
            return 1e6
        return cost
    except Exception:
        return 1e6

space = [
    Real(1.0, 5.0, name='spring_stiffness'),
    Real(0.5, 3.0, name='J'),
    #Integer(10, 20, name='N'),
    Real(1.0, 3.0, name='friction'),
    Real(0.1, 3, name='mass'),
    Integer(1200, 16000, name='input_size')
]


result = gp_minimize(
    func=objective,          # your objective function
    dimensions=space,        # parameter space
    n_calls=25,              # total function evaluations
    n_initial_points=7,      # random initial points before GP starts
    acq_func="EI",           # acquisition function: Expected Improvement
    random_state=42,
    verbose = True
)

# %%

print("Best score:", result.fun)
print("Best parameters:", result.x)

from skopt.plots import plot_convergence
plot_convergence(result)
# %%

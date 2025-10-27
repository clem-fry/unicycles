#%%

import numpy as np
import random
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd

#%% 

class Node():

    N = []
    dt = 0.01

    # robot would need a matrix of ids to know whose who?
    id_array = np.array([])
    # alternatively each robot could have a different impression of the springs ?
    #adjacency_matrix = [] # existance of connections - corresponding to id list?
    K = [] # spring stiffness between agents i and j matrix
    A = [] # spring lengths between agents i and j matrix
    all_nodes = []


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
        f = np.dot(self.T_theta, u(t))
        return f
    
    def f_s(self, t):
        f = np.dot(self.T_s, u(t))
        return f

    def Dudx(self):
        sum = 0
        for node in Node.all_nodes:
            Kij =  Node.K[self.matrix_row, node.matrix_row]
            Aij = Node.A[self.matrix_row, node.matrix_row]
            d = self.distance_to(node)
            if d > 0:
                energy = Kij * (Aij - d) * (self.x - node.x) / d
                sum += energy
        return sum

    def Dudz(self):
        sum = 0
        for node in Node.all_nodes:
            Kij =  Node.K[self.matrix_row, node.matrix_row]
            Aij = Node.A[self.matrix_row, node.matrix_row]
            d = self.distance_to(node)
            if d > 0:
                energy = Kij * (Aij - d) * (self.z - node.z) / d
                sum += energy
        return sum


    def update(self, t):
        if self.anchor:
            return()
        
        dw = 1/self.J * (self.f_theta(t) - self.zeta * self.w)
        self.w = self.w + Node.dt * dw
        #print('w: ', self.w)

        dtheta = self.w
        self.theta = self.theta + Node.dt * dtheta
        #print('theta: ',self.theta)

        ds = 1/self.m * (self.Dudx() * np.cos(self.theta) + self.Dudz() * np.sin(self.theta) + self.f_s(t) - self.beta * self.s)
        
        self.s = self.s + Node.dt * ds
        #print('Dudx: ', self.Dudx())
        #print('Dudz: ', self.Dudz())
        #print('f_s: ', self.f_s(t))

        #print('s: ',self.s)

        self.x = self.x + Node.dt * (np.cos(self.theta) * self.s)
        self.z = self.z + Node.dt * (np.sin(self.theta) * self.s)
        #print('x: ',self.x)
        #print('z: ',self.z)
#%%

def u(t): # input signal 
    f1 = 2.11 # frequencies
    f2 = 3.73
    f3 = 4.33
    T = 1
    u = 0.2 * np.sin(2*np.pi * f1 * t / T) * np.sin(2*np.pi * f2 * t / T) * np.sin(2*np.pi * f3 * t / T)
    return u

def NARAM_2(T): # NARMA_2
    y_array = [0, 0]
    for t in range(1, T):
        y = 0.4 * y_array[t] + 0.4 * y_array[t] * y_array[t-1] + 0.6 * u(t) ** 3 + 0.1
        y_array.append(y)
    return y_array

def NARMA_n(T, n): # NARMA_n
    alpha = 0.3
    beta = 0.05
    gamma = 1.5
    delta = 0.1
    y_array = [0]
    for t in range(1, T):
        sum_array = [y_array[t - j] for j in range(0, n-1)]
        y = alpha * y_array[-1] + beta * y_array[-1] * sum(sum_array) + gamma * u(t - n + 1) * u(t) + delta
        y_array.append(y)

    return y_array

#%%

# def u(t): # input signal 
#     u = np.sin(t) * 10000
#     return u

N = 5 # number of nodes
size = 10

delay = 2 # number of iterations - 2 * dt = 0.02 seconds 

ids = np.arange(1, N + 1)

anchors = np.zeros(N, dtype=int)
anchor_idx = np.random.randint(N)  # pick one random anchor
anchors[anchor_idx] = 1

# locations (triangle)
x_array = np.random.uniform(0, size, N)
z_array = np.random.uniform(0, size, N)

theta_array = np.random.uniform(0, 2*np.pi, N)

K = np.ones((N, N)) * 10 # spring stiffnesses

X, Z = np.meshgrid(x_array, z_array)
A = np.sqrt((X - X.T)**2 + (Z - Z.T)**2) # starting spring lengths -> beginning

np.fill_diagonal(A, 0)
np.fill_diagonal(K, 0)

Node.N = N
Node.K = K
Node.A = A
Node.id_array = ids
Node.all_nodes = []

for i, id in enumerate(ids):
    random_number = random.random()

    node = Node(id = id, x = x_array[i], z=z_array[i], 
                theta=theta_array[i], s=0, w=0, 
                J = 1, beta = random_number, zeta = random_number, 
                T_theta = 10000*random_number, T_s = 1000*random_number, m = 1)
    
    if anchors[i]:
        node.anchor=True

    Node.all_nodes.append(node)

# %% NUMERICAL INTEGRATION

iterations = 20000

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

#%% DATA SETS

data = np.stack([x_coords, z_coords, theta_coords, s_array, w_array])
data_states = data.reshape(-1, data.shape[2]).T

updates = int(np.shape(data_states)[0] // delay)

y_array = NARAM_2(np.shape(data_states)[0])
#y_array = np.repeat(y_array_rep, 2)

cut = int(np.shape(data_states)[0] * 0.1) # cut first 10 percent

X = data_states[cut:, :]
y = np.array(y_array[cut + 1:])

#X_train, X_test, y_train, y_test = train_test_split(data_states, y_array, test_size=0.2, random_state=17)
split_idx = int(0.7 * X.shape[0])
X_train, X_test = X[:split_idx, :], X[split_idx:, :]
y_train, y_test = y[:split_idx], y[split_idx:]

print("The dimension of X_train is {}".format(X_train.shape))
print("The dimension of X_test is {}".format(X_test.shape))

print("The dimension of y_train is {}".format(len(y_train)))
print("The dimension of y_test is {}".format(len(y_test)))

#%%
scaler = StandardScaler() # weight all state elements the same
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LinearRegression()
lr = Ridge(alpha=1e-6)

lr.fit(X_train, y_train)

prediction_test = lr.predict(X_test)
prediction_train = lr.predict(X_train)

actual = y_test

train_score_lr = lr.score(X_train, y_train)
test_score_lr = lr.score(X_test, y_test)

#%%
print("The train score for lr model is {}".format(train_score_lr))
print("The test score for lr model is {}".format(test_score_lr))

def nmse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2) / np.mean((y_true - np.mean(y_true))**2)

# Example usage
train_nmse = nmse(y_train, prediction_train)
test_nmse  = nmse(y_test[:100], prediction_test[:100])

print(f"Train NMSE: {train_nmse:.6f}")
print(f"Test NMSE:  {test_nmse:.6f}")

# %% sample result
%matplotlib inline
plt.plot(y[6000:6500], label="goal output")
plt.plot(lr.predict(scaler.transform(X))[6000:6500], label="lr output")
plt.xlabel("iteration")
plt.ylabel("output")
plt.legend()
plt.show()

#%% whole result
%matplotlib inline
plt.plot(y)
plt.plot(lr.predict(scaler.transform(X)))
plt.show()

#%% READOUT

%matplotlib inline
plt.plot(y_train[60:])
plt.plot(prediction_train[60:])
plt.show()

# %% PLOTTING
%matplotlib notebook
from matplotlib.animation import FuncAnimation

# --- Create figure and axis ---
fig, ax = plt.subplots()
ax.set_xlim(np.min(x_coords) * 0.9, np.max(x_coords) * 1.1)
ax.set_ylim(np.min(z_coords) * 0.9, np.max(z_coords) * 1.1)
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_title("Node Movement in X–Z Plane")

# Create scatter-like markers (each node as a dot)
points = [ax.plot([], [], "o", label=f"Node {i}")[0] for i in range(N)]

quiver = ax.quiver([0]*N, [0]*N, [0]*N, [0]*N,
                   angles='xy', scale_units='xy', scale=1, color='r')

ax.legend()
arrow_length = 0.1

# --- Init function ---
def init():
    for p in points:
        p.set_data([], [])

    quiver.set_UVC([0]*N, [0]*N)
    quiver.set_offsets(np.zeros((N, 2)))
    return points + [quiver]

# --- Update function ---
def update(frame):
    for i, p in enumerate(points):
        # Wrap in list brackets so each is a sequence (x=[..], y=[..])
        p.set_data([x_coords[i][frame]], [z_coords[i][frame]])

    U = arrow_length * np.cos([theta_coords[i][frame] for i in range(N)])
    V = arrow_length * np.sin([theta_coords[i][frame] for i in range(N)])
    
    # Update quiver positions and vectors
    quiver.set_offsets(np.c_[[x_coords[i][frame] for i in range(N)],
                              [z_coords[i][frame] for i in range(N)]])
    quiver.set_UVC(U, V)
    
    return points + [quiver]

# --- Keep the animation in a variable ---
ani = FuncAnimation(fig, update, frames=np.shape(x_coords)[1], init_func=init,
                    blit=False, interval=100, repeat=True)

plt.show()

#%%
from IPython.display import HTML
HTML(ani.to_jshtml())

# %%

from matplotlib.animation import PillowWriter
ani_200 = FuncAnimation(fig, update, frames=range(200), blit=True)
ani_200.save("animation.gif", writer='pillow', fps=10)
# %%

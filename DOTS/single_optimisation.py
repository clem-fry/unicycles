#%%
from skopt import gp_minimize
from skopt.space import Real, Integer
import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
import subprocess
import matplotlib.pyplot as plt
import gzip

def NARAM_2(u_array, t_array):
    # each iter is dt not one second
    iterations = len(t_array)
    y_array = [0, 0]
    for iter in range(1, iterations):
        # i want u(t) to be exactly the same as the instance it was initiated. i need the right t value.
        y = 0.4 * y_array[-1] + 0.4 * y_array[-1] * y_array[-2] + 0.6 * u_array[iter] ** 3 + 0.1
        y_array.append(y)
    return y_array[1:]

def NARAM(u_array, t_array, n):
        # each iter is dt not one second
    iterations = len(t_array)
    y_array = [0] * (n-1)
    # n = naram number
    for iter in range(1, iterations):
        # i want u(t) to be exactly the same as the instance it was initiated. i need the right t value.
        summation = np.sum([y_array[iter-j] for j in range(0, n-1)])
        y = 0.3 * y_array[iter] + 0.05 * y_array[iter] * summation + 1.5 * u_array[iter-n+1] * u_array[iter] + 0.1
        y_array.append(y)
    return y_array[n-2:]

#%%


#M, J, K, BETA, INPUT = 4.197018051401195, 0.1, 2.2013207416574865, 0.7595618237851425, 200

#second baysien rn
#Loaded parameters: M=2.2830724068420882, J=0.8961301327430423, K=0.5693183782920298, BETA=2.7444819837539214, INPUT=132.0

M, J, K, BETA, INPUT =2.2830724068420882, 0.8961301327430423, 0.5693183782920298, 2.7444819837539214, 132.0

# !! run simulation !!

print(f"ros2 launch dots_example_controllers sim_unicycles.launch.py \"M:={M}\" \"J:={J}\" \"K:={K}\" \"BETA:={BETA}\" \"INPUT:={INPUT}\" ")

#%%

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = '/home/simonj/projects/dots_sirius/workspaces/clem/dots_templates_arm/time-period-80-retake5.csv.gz'

#os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = np.loadtxt(csv_path, delimiter=",", skiprows=1) # .gz

data = np.loadtxt(
    csv_path,
    delimiter=",",
    skiprows=1
)

with gzip.open(csv_path, "rt") as f:
    header = f.readline().strip().split(",")
    
# Load numeric data
data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

#Find indices of columns to keep (exclude qz and qw)
keep_indices = [
    i for i, h in enumerate(header)
    if not (h.endswith("_omega")or h.endswith("_qw") or h.endswith("_qz") or h.endswith("command") or h.endswith('measured'))
]

# Filter data
data = data[:, keep_indices]
data = data[:-1500, :]
data = data[::, :]

headers_filtered = [header[i] for i in keep_indices]

print("data shape: ", data.shape)

t_array = data[:, 0]

idx = np.argmin(np.abs(t_array - 10.0)) # start at t=10 to avoid initial
idx = 500

# cut_start = 500
# cut_end = 6000
# X = data[cut_start:cut_end, :]

print("data shape: ", data.shape)

u_array = data[:, 1]
t_array_cut = data[:, 0]

u_array_normalised = u_array - min(u_array)
u_array_normalised = 0.5 * (u_array_normalised)/max(u_array_normalised)


X = data[:, 2:] # correct input data

y = NARAM_2(u_array_normalised, t_array_cut)
#y = NARAM(u_array_normalised, t_array_cut, 5)

cut_idx = int(len(y) * 0.15)
y = y[cut_idx:]
X = X[cut_idx:]

split_idx = int(0.2* X.shape[0])
X_test, X_train  = X[:split_idx, :], X[split_idx:, :]
y_test, y_train = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_trans = scaler.fit_transform(X_train)
X_test_trans = scaler.transform(X_test)

lr = Ridge(alpha=6e2)
#lr = LinearRegression()

lr.fit(X_train_trans, y_train)
y_pred = lr.predict(X_test_trans)

def nmse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / np.var(y_true) #np.mean((y_true - np.mean(y_true))**2)

test_nmse = nmse(y_test, y_pred)
print('test nmse: ', test_nmse)

#%%
from sklearn.linear_model import RidgeCV

lr = RidgeCV(alphas=np.logspace(-3, 4, 50))
lr.fit(X_train_trans, y_train)

print("Best alpha:", lr.alpha_)


# %%
plt.title('NARAM8')
plt.plot(y_train, label="actual")
plt.plot(lr.predict(X_train_trans), label="prediction")
plt.legend()

# %%
plt.title('NARAM5')
plt.plot(y_test, label="actual")
plt.plot(y_pred, label="prediction")
plt.legend()
# %%
plt.plot(y_pred[100:700], label="prediction")
plt.plot(y_test[100:700], label="actual")
plt.legend()
# %%
plt.plot(y_pred[20100:20500], label="prediction")
plt.plot(y_test[20100:20500], label="actual")
plt.legend()
#%%

plt.figure()

for i, h in enumerate(headers_filtered):
    #if h.endswith("_pos_y"):
    if h.endswith("_pos_x"):
    #if h.endswith("_linear_x_command"):
    #if h.endswith("_linear_x_measured"):
        plt.plot(data[:, i], label=h)

plt.legend()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

coefs = lr.coef_.ravel()
n_nodes = 10

# Remove first two non-state columns if needed
state_columns = headers_filtered[2:]   # use filtered list
n_states = len(state_columns) // n_nodes

coef_matrix = coefs.reshape(n_nodes, n_states) * 100

with gzip.open(csv_path, "rt") as f:
    header = f.readline().strip().split(",")

# Take the first robot's states
first_robot = state_columns[:n_states]

# Remove robot prefix properly
state_names = [
    col.split("_", 2)[-1]   # removes "r25_" etc
    for col in first_robot
]

node_names = [f"Node_{i+1}" for i in range(n_nodes)]
#state_names = [f"State_{i+1}" for i in range(n_states)]

fig, ax = plt.subplots(figsize=(10, 6))

v = np.max(np.abs(coef_matrix))
im = ax.imshow(coef_matrix, cmap="coolwarm", vmin=-v, vmax=v, aspect="auto")

ax.set_xticks(range(n_states))
ax.set_yticks(range(n_nodes))
ax.set_xticklabels(state_names, rotation=45, ha="right")
ax.set_yticklabels(node_names)

for i in range(n_nodes):
    for j in range(n_states):
        ax.text(j, i, f"{coef_matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=9)

plt.colorbar(im, ax=ax, label="Coefficient (%)")
ax.set_title("Coefficient Heatmap")
plt.tight_layout()
plt.show()
# %%

plt.plot(X[cut_idx:, 0])
plt.plot(X[cut_idx:, 4])
plt.plot(X[cut_idx:, 8])
plt.plot(X[cut_idx:, 12])
plt.plot(X[cut_idx:, 16])
plt.plot(X[cut_idx:, 20])
#plt.plot(X[cut_idx:, 36])
# %%


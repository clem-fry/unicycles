#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

#%%

def u(t, T): # input signal 
    f1 = 2.11 # frequencies
    f2 = 3.73
    f3 = 4.33
    u = 0.2 * np.sin(2*np.pi * f1 * t / T) * np.sin(2*np.pi * f2 * t / T) * np.sin(2*np.pi * f3 * t / T)
    return u

def NARAM_2(time, T): # NARMA_2
    y_array = [0, 0]
    for t in range(1, time):
        y = 0.4 * y_array[t] + 0.4 * y_array[t] * y_array[t-1] + 0.6 * u(t, T) ** 3 + 0.1
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

def calc_nmse(y_func):
    test_nmse_array = []
    period_ratios = np.arange(1, 4.25, 0.25)
    simulation_data = np.load('node-simulation.npz')
    for ratio in period_ratios:
        T = ratio
        data_states = simulation_data[f'T={ratio}']
        y_array = y_func(np.shape(data_states)[0], T)

        cut = int(np.shape(data_states)[0] * 0.1) # cut first 10 percent
        X = data_states[cut:, :]
        #y_array = np.repeat(y_array_rep, 2)

        y = np.array(y_array[cut:])

        #X_train, X_test, y_train, y_test = train_test_split(data_states, y_array, test_size=0.2, random_state=17)
        split_idx = int(0.8 * X.shape[0])
        X_train, X_test = X[:split_idx, :], X[split_idx:, :]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print("The dimension of X_train is {}".format(X_train.shape))
        print("The dimension of X_test is {}".format(X_test.shape))

        print("The dimension of y_train is {}".format(len(y_train)))
        print("The dimension of y_test is {}".format(len(y_test)))

        scaler = StandardScaler() # weight all state elements the same
        X_train_transformed = scaler.fit_transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        lr = LinearRegression()

        lr.fit(X_train_transformed, y_train)

        prediction_test = lr.predict(X_test_transformed)
        prediction_train = lr.predict(X_train_transformed)

        actual = y_test

        train_score_lr = lr.score(X_train_transformed, y_train)
        test_score_lr = lr.score(X_test_transformed, y_test)

        print("The train score for lr model is {}".format(train_score_lr))
        print("The test score for lr model is {}".format(test_score_lr))

        def nmse(y_true, y_pred):
            return np.mean((y_true - y_pred)**2) / np.mean((y_true - np.mean(y_true))**2)

        # Example usage
        train_nmse = nmse(y_train, prediction_train)
        test_nmse  = nmse(y_test, prediction_test)

        print(f"Train NMSE: {train_nmse:.6f}")
        print(f"Test NMSE:  {test_nmse:.6f}")
        test_nmse_array.append(test_nmse)
    return test_nmse_array, period_ratios

# %%

test_nmse_array_2, period_ratios = calc_nmse(NARAM_2)
test_nmse_array_5, period_ratios = calc_nmse(NARAM_5)

#%% plot
%matplotlib inline
plt.clf()
plt.cla()
plt.plot(period_ratios, test_nmse_array_2, 'o-', label="NARAM 2")
plt.plot(period_ratios, test_nmse_array_5, 'o-', label = "NARAM 5")

plt.xlabel("Input period ratio")
plt.ylabel("Test NMSE")
plt.ylim((0, 1))
plt.legend()
plt.show()

# %%

import seaborn as sns  # if not installed: pip install seaborn
import pandas as pd

coefs = lr.coef_.ravel() 
n_nodes = 10
n_states = 5

coef_matrix = coefs.reshape(n_nodes, n_states)
node_names = [f"Node_{i+1}" for i in range(n_nodes)]
state_names = [f"State_{j+1}" for j in range(n_states)]

coef_df = pd.DataFrame(coef_matrix, columns=state_names, index=node_names)

plt.figure(figsize=(10, 6))
sns.heatmap(
    coef_df,
    annot=True,        # show numbers inside cells
    fmt=".2f",         # number format
    cmap="coolwarm",   # color palette
    center=0           # white = 0, red = positive, blue = negative
)
%matplotlib inline
plt.title("Linear Regression Coefficients by Node and State", fontsize=14)
plt.xlabel("State")
plt.ylabel("Node")
plt.tight_layout()
plt.show()
# %%

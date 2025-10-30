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
    return y_array

period_ratios = np.arange(1, 4.25, 0.25)
simulation_data = np.load('node-simulation.npz')

test_nmse_array = []

for ratio in period_ratios:
    T = ratio
    data_states = simulation_data[f'T={ratio}']
    y_array = NARAM_2(np.shape(data_states)[0], T)

    cut = int(np.shape(data_states)[0] * 0.1) # cut first 10 percent
    X = data_states[cut:, :]
    #y_array = np.repeat(y_array_rep, 2)

    y = np.array(y_array[cut + 1:])

    #X_train, X_test, y_train, y_test = train_test_split(data_states, y_array, test_size=0.2, random_state=17)
    split_idx = int(0.7 * X.shape[0])
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
    lr = Ridge(alpha=1e-6)

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
    test_nmse  = nmse(y_test[:100], prediction_test[:100])

    print(f"Train NMSE: {train_nmse:.6f}")
    print(f"Test NMSE:  {test_nmse:.6f}")
    test_nmse_array.append(test_nmse)

# %%
%matplotlib inline
plt.clf()
plt.cla()
plt.plot(period_ratios, test_nmse_array, 'o-')
plt.xlabel("Input period ratio")
plt.ylabel("Test NMSE")
plt.ylim((0, 1))
plt.legend()
plt.show()

# %%

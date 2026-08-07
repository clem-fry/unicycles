#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def u(t, T): # input signal 
    f1 = 2.11 # frequencies
    f2 = 3.73
    f3 = 4.33
    u = 0.2 * np.sin(2*np.pi * f1 * t / T) * np.sin(2*np.pi * f2 * t / T) * np.sin(2*np.pi * f3 * t / T)
    return u

def NARAM_2(time, T): # NARMA_2
    y_array = [0, 0]
    step = 0.1 # CHECK THIS MATCHES
    iterations = np.arange(0, time * step, step)
    iterations = iterations[1:]
    for t in iterations:
        y = 0.4 * y_array[-1] + 0.4 * y_array[-1] * y_array[-2] + 0.6 * u(t, T) ** 3 + 0.1
        y_array.append(y)
    return y_array[1:]

def NARAM(time, T, n): # NARMA_2
    y_array = [0] * (n-1)
    step = 0.1
    iterations = np.arange(0, time * step, step)
    iterations = iterations[1:]
    for i, t in enumerate(iterations):
        sumation = np.sum([y_array[i-j] for j in range(0, n-1)])
        y = 0.3 * y_array[i] + 0.05 * y_array[i] * sumation + 1.5 * u(t-n+1, T) * u(t, T) + 0.1
        y_array.append(y)
    return y_array[n-2:]

def calc_nmse_narma(y_array):
    test_nmse_array = []
    #period_ratios = np.arange(1, 4.25, 0.25)
    #period_ratios = np.arange(10, 42.5, 2.5)
    period_ratios = np.arange(60, 80, 10)
    simulation_data = np.load('node-simulation.npz')
    for ratio in period_ratios:
        T = ratio
        data_states = simulation_data[f'T={ratio}']

        cut = int(np.shape(data_states)[0] * 0.1) # cut first 10 percent
        X = data_states[cut:, :]
        #y_array = np.repeat(y_array_rep, 2)

        y = np.array(y_array[cut:])

        #X_train, X_test, y_train, y_test = train_test_split(data_states, y_array, test_size=0.2, random_state=17)
        split_idx = int(0.95 * X.shape[0])
        X_train, X_test = X[:split_idx, :], X[split_idx:, :]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print("The dimension of X_train is {}".format(X_train.shape))
        print("The dimension of X_test is {}".format(X_test.shape))

        print("The dimension of y_train is {}".format(len(y_train)))
        print("The dimension of y_test is {}".format(len(y_test)))

        scaler = StandardScaler() # weight all state elements the same
        X_train_transformed = scaler.fit_transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        #lr = LinearRegression()
        lr = Ridge()

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


def add_lag(data_states, lag):
    # appends just the single state from `lag` timesteps before each row
    # (not every step in between) - one delayed copy, not a full tapped
    # delay line. Rows too early to have `lag` of real history behind them
    # are zero-padded; that only touches the first `lag` rows, which the 10%
    # warm-up cut in calc_nmse discards anyway as long as lag isn't a large
    # fraction of the run.
    if lag == 0:
        return data_states
    n, d = data_states.shape
    delayed = np.vstack([np.zeros((lag, d)), data_states[:-lag]])
    return np.concatenate([data_states, delayed], axis=1)



def calc_nmse(y_array, lag=0, plot=False):
    # fits the swarm's recorded states to the moving repulsion source's own
    # position - i.e. can the reservoir decode where the stimulus currently is
    simulation_data = np.load('node-simulation.npz')
    data_states = simulation_data['data_states']
    data_states = add_lag(data_states, lag)

    cut = int(np.shape(data_states)[0] * 0.1) # cut first 10 percent
    X = data_states[cut:, :]

    # y_array is [source_x, source_y], shape (2, n) - transpose so samples
    # are on axis 0 like X, giving (n, 2) with columns [x_s, y_s]
    y = np.array(y_array)[:, cut:].T

    #X_train, X_test, y_train, y_test = train_test_split(data_states, y_array, test_size=0.2, random_state=17)
    split_idx = int(0.95 * X.shape[0])
    X_train, X_test = X[:split_idx, :], X[split_idx:, :]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("The dimension of X_train is {}".format(X_train.shape))
    print("The dimension of X_test is {}".format(X_test.shape))

    print("The dimension of y_train is {}".format(len(y_train)))
    print("The dimension of y_test is {}".format(len(y_test)))

    scaler = StandardScaler() # weight all state elements the same
    X_train_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    #lr = LinearRegression()
    lr = Ridge()

    lr.fit(X_train_transformed, y_train)

    prediction_test = lr.predict(X_test_transformed)
    prediction_train = lr.predict(X_train_transformed)

    train_score_lr = lr.score(X_train_transformed, y_train)
    test_score_lr = lr.score(X_test_transformed, y_test)

    print("The train score for lr model is {}".format(train_score_lr))
    print("The test score for lr model is {}".format(test_score_lr))

    def nmse(y_true, y_pred):
        # y_true/y_pred are (n_samples, 2) for [x_s, y_s]; normalize per
        # target column before combining, so x and y are weighted equally
        return np.mean((y_true - y_pred)**2) / np.mean((y_true - np.mean(y_true, axis=0))**2)

    train_nmse = nmse(y_train, prediction_train)
    test_nmse  = nmse(y_test, prediction_test)

    print(f"Train NMSE: {train_nmse:.6f}")
    print(f"Test NMSE:  {test_nmse:.6f}")

    if plot:
        plot_predictions(y_test, prediction_test)

    nmses = [train_nmse, test_nmse]
    ys = [y_train, y_test]
    Xs = [X_train_transformed, X_test_transformed]
    predictions = [prediction_train, prediction_test]

    return lr, nmses, ys, Xs, predictions

def plot_predictions(y_test, prediction_test):
    target_names = ["source_x", "source_y"]
    n_targets = y_test.shape[1]
    _, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 4), squeeze=False)
    for j, ax in enumerate(axes[0]):
        name = target_names[j] if j < len(target_names) else f"target_{j}"
        ax.plot(y_test[:, j], label="actual")
        ax.plot(prediction_test[:, j], label="predicted")
        ax.set_xlabel("test sample index")
        ax.set_ylabel(name)
        ax.set_title(f"{name}: actual vs predicted")
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_trajectory_2d(y, prediction, size=None):
    # actual vs predicted source path in the X-Y plane (rather than value vs
    # sample-index like plot_predictions) - pass a subsection, e.g.
    # y_test[:2000], prediction_test[:2000], to zoom in on a slice
    _, ax = plt.subplots(figsize=(6, 6))
    ax.plot(y[:, 0], y[:, 1], color="tab:blue", linewidth=1.5, label="actual")
    ax.plot(prediction[:, 0], prediction[:, 1], color="tab:orange", linewidth=1.5,
            linestyle="--", label="predicted")
    ax.scatter(*y[0], color="tab:blue", marker="o", s=60, zorder=5, label="actual start")
    ax.scatter(*prediction[0], color="tab:orange", marker="o", s=60, zorder=5, label="predicted start")

    if size is not None:
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Actual vs predicted source trajectory")
    ax.set_aspect("equal")
    ax.legend()
    plt.show()

# %%

#lr, train_nmse, test_nmse = calc_nmse(source_data)
#plot_coefficients(lr)

# %%

STATE_NAMES = ["x", "y", "theta", "s"]  # order matches data_states in global-sim-person.py

def plot_coefficients(lr):
    import seaborn as sns  # if not installed: pip install seaborn
    import pandas as pd

    n_states = len(STATE_NAMES)
    n_nodes = lr.coef_.shape[-1] // n_states
    node_names = [f"Node_{i+1}" for i in range(n_nodes)]

    # lr.coef_ is (n_targets, n_features): one row per predicted output
    # (source_x, source_y when y is 2D), so draw one heatmap per target
    coef_rows = np.atleast_2d(lr.coef_)
    target_names = ["source_x", "source_y"][:coef_rows.shape[0]]

    _, axes = plt.subplots(1, len(target_names), figsize=(6 * len(target_names), 6), squeeze=False)
    for ax, target_name, coefs in zip(axes[0], target_names, coef_rows):
        coef_matrix = coefs.reshape(n_states, n_nodes).T
        coef_df = pd.DataFrame(coef_matrix, columns=STATE_NAMES, index=node_names)

        sns.heatmap(
            coef_df,
            annot=True,        # show numbers inside cells
            fmt=".2f",         # number format
            cmap="coolwarm",   # color palette
            center=0,          # white = 0, red = positive, blue = negative
            ax=ax,
        )
        ax.set_title(f"Coefficients predicting {target_name}")
        ax.set_xlabel("State")
        ax.set_ylabel("Node")

    plt.tight_layout()
    plt.show()
# %%

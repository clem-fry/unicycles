#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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


def add_lag_history(data_states, lag):
    # like add_lag, but appends every lag from 1 up to `lag` (a full tapped
    # delay line) instead of just the single `lag`-step-back snapshot - the
    # readout gets the whole recent history as training features, not one
    # point from the past. More features (lag x as many), so more capacity
    # but also more overfitting risk than add_lag for the same lag value.
    if lag == 0:
        return data_states
    n, d = data_states.shape
    delayed_blocks = [np.vstack([np.zeros((l, d)), data_states[:-l]]) for l in range(1, lag + 1)]
    return np.concatenate([data_states] + delayed_blocks, axis=1)



def calc_nmse(y_array, lag=0, cumulative=False, plot=False, alpha=None, filename='node-simulation.npz',
              alpha_candidates=(0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0), alpha_holdout_frac=0.2):
    # fits the swarm's recorded states to the moving repulsion source's own
    # position - i.e. can the reservoir decode where the stimulus currently is
    #
    # alpha=None (default) auto-picks Ridge's alpha instead of using a fixed
    # value: it's swept over alpha_candidates, scored on a holdout carved out
    # of the *training* split only (the last alpha_holdout_frac of it) -
    # never X_test - so the reported test_nmse below isn't contaminated by
    # having picked alpha to do well on that same test data. Pass a number
    # to skip all that and use it directly, same as before.
    simulation_data = np.load(filename)
    data_states = simulation_data['data_states']

    data_states = add_lag_history(data_states, lag) if cumulative else add_lag(data_states, lag)

    cut = int(np.shape(data_states)[0] * 0.1) # cut first 10 percent
    X = data_states[cut:, :]

    # y_array is [source_x, source_y], shape (2, n) - transpose so samples
    # are on axis 0 like X, giving (n, 2) with columns [x_s, y_s]
    y = np.array(y_array)[:, cut:].T

    #X_train, X_test, y_train, y_test = train_test_split(data_states, y_array, test_size=0.2, random_state=17)
    split_idx = int(0.8 * X.shape[0])
    X_train, X_test = X[:split_idx, :], X[split_idx:, :]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("The dimension of X_train is {}".format(X_train.shape))
    print("The dimension of X_test is {}".format(X_test.shape))

    print("The dimension of y_train is {}".format(len(y_train)))
    print("The dimension of y_test is {}".format(len(y_test)))

    def nmse(y_true, y_pred):
        # y_true/y_pred are (n_samples, 2) for [x_s, y_s]; normalize per
        # target column before combining, so x and y are weighted equally
        return np.mean((y_true - y_pred)**2) / np.mean((y_true - np.mean(y_true, axis=0))**2)

    scaler = StandardScaler() # weight all state elements the same
    X_train_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    if alpha is None:
        holdout_start = int(X_train_transformed.shape[0] * (1 - alpha_holdout_frac))
        X_fit, X_holdout = X_train_transformed[:holdout_start], X_train_transformed[holdout_start:]
        y_fit, y_holdout = y_train[:holdout_start], y_train[holdout_start:]
        best_alpha, best_holdout_nmse = None, None
        for a in alpha_candidates:
            lr_try = Ridge(alpha=a)
            lr_try.fit(X_fit, y_fit)
            h_nmse = nmse(y_holdout, lr_try.predict(X_holdout))
            if best_holdout_nmse is None or h_nmse < best_holdout_nmse:
                best_alpha, best_holdout_nmse = a, h_nmse
        print(f"auto-selected alpha={best_alpha} (holdout NMSE {best_holdout_nmse:.6f})")
        alpha = best_alpha

    #lr = LinearRegression()
    lr = Ridge(alpha=alpha)

    lr.fit(X_train_transformed, y_train)

    prediction_test = lr.predict(X_test_transformed)
    prediction_train = lr.predict(X_train_transformed)

    train_score_lr = lr.score(X_train_transformed, y_train)
    test_score_lr = lr.score(X_test_transformed, y_test)

    print("The train score for lr model is {}".format(train_score_lr))
    print("The test score for lr model is {}".format(test_score_lr))

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


def calc_nmse_transfer(y_array, walker_active, filename='node-simulation.npz', lag=0, cumulative=False,
                        plot=False, alphas=(0.1, 1.0, 10.0, 100.0, 300.0, 1000.0), alpha_holdout_frac=0.2,
                        transient_window=100):
    # Answers "if I train a readout while the walker's present, remove it,
    # let the swarm settle, then bring it back, can that same readout still
    # track the walker with no further fitting?" walker_active is the
    # per-tick bool array global-sim-local-sensors.py's simulation(...,
    # walker_active_mask=...) returns/saves (True = robots can sense the
    # walker that tick) - used here only to find the phase boundaries, not
    # as a feature.
    #
    # Three phases come out of walker_active's first False->True->False
    # transition-pair:
    #   - trainable: [warmup_cut, first_removed) - walker present, used to
    #     fit the readout (with an internal holdout for picking alpha - see
    #     below)
    #   - removed: [first_removed, reintroduced) - walker absent, skipped
    #     entirely. The robots have no information about the walker while
    #     it's gone, so scoring predictions there wouldn't measure anything
    #     meaningful.
    #   - transfer: [reintroduced, end) - walker back, scored using the
    #     readout fit on the trainable phase ONLY - no refitting here, since
    #     that's the entire point of the test.
    #
    # alpha is picked by holding out the last alpha_holdout_frac of the
    # *trainable* phase, never any transfer-phase data - using transfer
    # samples to choose alpha would leak information about the segment
    # we're claiming not to have trained on, making the reported
    # transfer_nmse optimistic.
    #
    # transfer_nmse is also split into a transient_window-sample "transient"
    # right at reintroduction and a "steady" remainder, scored separately.
    # The trainable phase itself starts the same way (swarm at rest at
    # x0/y0 before the walker's done anything), but that configuration is a
    # small fraction of a long trainable phase once the walker gets going -
    # so this checks how much of the overall transfer gap is just the
    # readout being unfamiliar with that brief at-rest moment, versus a
    # genuine steady-state tracking cost after the swarm's back in motion.
    simulation_data = np.load(filename)
    data_states = simulation_data['data_states']
    walker_active = np.asarray(walker_active, dtype=bool)

    data_states = add_lag_history(data_states, lag) if cumulative else add_lag(data_states, lag)
    y = np.array(y_array).T  # (n, 2): columns [x_s, y_s]

    inactive_idx = np.flatnonzero(~walker_active)
    if inactive_idx.size == 0:
        raise ValueError("walker_active never goes False - nothing was removed")
    first_removed = inactive_idx[0]
    active_after = np.flatnonzero(walker_active[first_removed:])
    if active_after.size == 0:
        raise ValueError("walker never comes back after removal - nothing to transfer-test on")
    reintroduced = first_removed + active_after[0]

    cut = int(data_states.shape[0] * 0.1)  # same warm-up cut as calc_nmse
    train_start = min(cut, first_removed)

    X_train_full = data_states[train_start:first_removed]
    y_train_full = y[train_start:first_removed]
    X_transfer = data_states[reintroduced:]
    y_transfer = y[reintroduced:]

    print(f"trainable phase: samples [{train_start}, {first_removed})  -> {X_train_full.shape[0]} samples")
    print(f"removed phase:   samples [{first_removed}, {reintroduced}) -> {reintroduced - first_removed} samples (unused)")
    print(f"transfer phase:  samples [{reintroduced}, {data_states.shape[0]}) -> {X_transfer.shape[0]} samples")

    def nmse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2) / np.mean((y_true - np.mean(y_true, axis=0)) ** 2)

    holdout_start = int(X_train_full.shape[0] * (1 - alpha_holdout_frac))
    X_fit, X_holdout = X_train_full[:holdout_start], X_train_full[holdout_start:]
    y_fit, y_holdout = y_train_full[:holdout_start], y_train_full[holdout_start:]

    scaler = StandardScaler()
    X_fit_t = scaler.fit_transform(X_fit)
    X_holdout_t = scaler.transform(X_holdout)

    best_alpha, best_holdout_nmse = None, None
    for a in alphas:
        lr = Ridge(alpha=a)
        lr.fit(X_fit_t, y_fit)
        h_nmse = nmse(y_holdout, lr.predict(X_holdout_t))
        if best_holdout_nmse is None or h_nmse < best_holdout_nmse:
            best_alpha, best_holdout_nmse = a, h_nmse
    print(f"alpha selected via trainable-phase holdout: {best_alpha} (holdout NMSE {best_holdout_nmse:.6f})")

    # refit at the chosen alpha on the full trainable phase (fit + holdout
    # slice together) - this is the readout that goes untouched into the
    # transfer phase, no further fitting past this point
    scaler = StandardScaler()
    X_train_t = scaler.fit_transform(X_train_full)
    X_transfer_t = scaler.transform(X_transfer)

    lr = Ridge(alpha=best_alpha)
    lr.fit(X_train_t, y_train_full)

    prediction_train = lr.predict(X_train_t)
    prediction_transfer = lr.predict(X_transfer_t)

    train_nmse = nmse(y_train_full, prediction_train)
    transfer_nmse = nmse(y_transfer, prediction_transfer)

    # transient (first transient_window samples right after reintroduction)
    # vs steady (the rest) - see docstring. transient_nmse/steady_nmse are
    # None if the transfer phase is too short to split meaningfully.
    window = min(transient_window, X_transfer.shape[0])
    y_transient, pred_transient = y_transfer[:window], prediction_transfer[:window]
    y_steady, pred_steady = y_transfer[window:], prediction_transfer[window:]
    transient_nmse = nmse(y_transient, pred_transient) if window > 0 else None
    steady_nmse = nmse(y_steady, pred_steady) if y_steady.shape[0] > 0 else None

    print(f"Train NMSE (trainable phase, in-sample):                  {train_nmse:.6f}")
    print(f"Transfer NMSE (post-reintroduction, no further training): {transfer_nmse:.6f}")
    if transient_nmse is not None:
        print(f"  - transient (first {window} samples after reintroduction): {transient_nmse:.6f}")
    else:
        print("  - transient: n/a (transfer phase too short)")
    if steady_nmse is not None:
        print(f"  - steady (remaining {y_steady.shape[0]} samples):         {steady_nmse:.6f}")
    else:
        print("  - steady: n/a (transfer phase too short)")

    if plot:
        plot_predictions(y_transfer, prediction_transfer)

    return {
        'transient_nmse': transient_nmse, 'steady_nmse': steady_nmse,
        'y_transient': y_transient, 'prediction_transient': pred_transient,
        'y_steady': y_steady, 'prediction_steady': pred_steady,
        'alpha': best_alpha, 'train_nmse': train_nmse, 'transfer_nmse': transfer_nmse,
        'first_removed': first_removed, 'reintroduced': reintroduced,
        'lr': lr, 'y_train': y_train_full, 'y_transfer': y_transfer,
        'prediction_train': prediction_train, 'prediction_transfer': prediction_transfer,
    }


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


def animate_trajectory_2d(y, prediction, size=None, max_frames=500, trail_length=50):
    # animated version of plot_trajectory_2d: actual and predicted source
    # markers moving together over time, each with a short trailing path so
    # you can see them converge/diverge as prediction quality varies.
    # Returns the animation object - display with
    # display(HTML(ani.to_jshtml())) same as plots.animation.
    n_steps = y.shape[0]
    frame_idx = np.linspace(0, n_steps - 1, min(max_frames, n_steps)).astype(int)

    fig, ax = plt.subplots(figsize=(6, 6))
    if size is not None:
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Actual vs predicted source trajectory")
    ax.set_aspect("equal")

    actual_trail, = ax.plot([], [], color="tab:blue", alpha=0.4, linewidth=1.5)
    pred_trail, = ax.plot([], [], color="tab:orange", alpha=0.4, linewidth=1.5, linestyle="--")
    actual_point, = ax.plot([], [], "o", color="tab:blue", markersize=10, label="actual")
    pred_point, = ax.plot([], [], "o", color="tab:orange", markersize=10, label="predicted")
    ax.legend()

    def init():
        actual_trail.set_data([], [])
        pred_trail.set_data([], [])
        actual_point.set_data([], [])
        pred_point.set_data([], [])
        return actual_trail, pred_trail, actual_point, pred_point

    def update(i):
        frame = frame_idx[i]
        start = max(0, frame - trail_length)
        actual_trail.set_data(y[start:frame + 1, 0], y[start:frame + 1, 1])
        pred_trail.set_data(prediction[start:frame + 1, 0], prediction[start:frame + 1, 1])
        actual_point.set_data([y[frame, 0]], [y[frame, 1]])
        pred_point.set_data([prediction[frame, 0]], [prediction[frame, 1]])
        return actual_trail, pred_trail, actual_point, pred_point

    ani = FuncAnimation(fig, update, frames=len(frame_idx), init_func=init,
                         blit=False, interval=100, repeat=True)
    return ani

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


def plot_weight_matrix(lr):
    # the raw readout weight matrix in one shot: rows are the predicted
    # output positions (source_x, source_y), columns are every input
    # feature (each robot's x/y/theta/s) - unlike plot_coefficients, which
    # reshapes each target into a separate Node x State grid, this keeps
    # every feature as its own column for an at-a-glance view of the whole
    # readout
    import seaborn as sns  # if not installed: pip install seaborn
    import pandas as pd

    n_states = len(STATE_NAMES)
    coef_rows = np.atleast_2d(lr.coef_)  # (n_targets, n_features)
    n_features = coef_rows.shape[-1]
    n_nodes = n_features // n_states
    target_names = ["source_x", "source_y"][:coef_rows.shape[0]]

    # feature order matches data_states' state-major packing in
    # global-sim-person.py: state_idx * n_nodes + node_idx
    feature_labels = [f"{state}_{node + 1}" for state in STATE_NAMES for node in range(n_nodes)]

    weight_df = pd.DataFrame(coef_rows, index=target_names, columns=feature_labels)

    plt.figure(figsize=(max(10, n_features * 0.3), 1.5 * len(target_names) + 1))
    sns.heatmap(weight_df, cmap="coolwarm", center=0, cbar_kws={"label": "weight"})
    plt.title("Readout weight matrix (output position x input feature)")
    plt.xlabel("Input feature (state_node)")
    plt.ylabel("Output position")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
# %%

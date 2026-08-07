#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
# #%% whole result
# plt.plot(y_train, label="goal output")
# plt.plot(prediction_train, label="lr output")
# plt.xlabel("iteration")
# plt.ylabel("output")
# plt.legend()
# plt.show()

# #%% test result
# plt.plot(y_test, label="goal output")
# plt.plot(prediction_test, label="lr output")
# plt.xlabel("iteration")
# plt.ylabel("output")
# plt.legend()
# plt.show()

# #%%
# plt.plot(y_test[-100:], label="goal output")
# plt.plot(prediction_test[-100:], label="lr output")
# plt.xlabel("iteration")
# plt.ylabel("output")
# plt.legend()
# plt.show()

# #%%

# plt.plot(y_test[:100], label="goal output")
# plt.plot(prediction_test[:100], label="lr output")
# plt.xlabel("iteration")
# plt.ylabel("output")
# plt.legend()
# plt.show()

# #%% READOUT

# plt.plot(y_train[60:])
# plt.plot(prediction_train[60:])
# plt.show()

# %% PLOTTING

def animation(x_coords, z_coords, theta_coords, source_x=None, source_y=None, source_radius=None, max_frames=500):
    N = np.shape(x_coords)[0]
    total_steps = np.shape(x_coords)[1]
    # sample evenly across the *whole* run rather than just the first
    # max_frames steps, so long runs (large num_iterations) are still shown
    # in full instead of only their opening moments - rendering cost stays
    # capped at max_frames either way
    frame_idx = np.linspace(0, total_steps - 1, min(max_frames, total_steps)).astype(int)

    # --- Create figure and axis ---
    fig, ax = plt.subplots()
    xlim = [np.min(x_coords), np.max(x_coords)]
    ylim = [np.min(z_coords), np.max(z_coords)]
    if source_x is not None:
        xlim = [min(xlim[0], np.min(source_x)), max(xlim[1], np.max(source_x))]
        ylim = [min(ylim[0], np.min(source_y)), max(ylim[1], np.max(source_y))]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    # Create scatter-like markers (each node as a dot)
    points = [ax.plot([], [], "o", label=f"Node {i}")[0] for i in range(N)]

    quiver = ax.quiver([0]*N, [0]*N, [0]*N, [0]*N,
                    angles='xy', scale_units='xy', scale=1, color='r')

    arrow_length = 0.2

    # optional marker for the moving repulsion source
    source_point = None
    source_circle = None
    if source_x is not None:
        source_point, = ax.plot([], [], "*", color="purple", markersize=15,
                                 label="repulsion source")
        if source_radius is not None:
            source_circle = plt.Circle((0, 0), source_radius, color="purple",
                                        fill=False, linestyle="--", alpha=0.4)
            ax.add_patch(source_circle)

    # --- Init function ---
    def init():
        for p in points:
            p.set_data([], [])

        quiver.set_UVC([0]*N, [0]*N)
        quiver.set_offsets(np.zeros((N, 2)))

        artists = points + [quiver]
        if source_point is not None:
            source_point.set_data([], [])
            artists.append(source_point)
        if source_circle is not None:
            source_circle.set_center((0, 0))
            artists.append(source_circle)
        return artists

    # --- Update function ---
    def update(i):
        frame = frame_idx[i]
        for j, p in enumerate(points):
            # Wrap in list brackets so each is a sequence (x=[..], y=[..])
            p.set_data([x_coords[j][frame]], [z_coords[j][frame]])

        U = arrow_length * np.cos([theta_coords[j][frame] for j in range(N)])
        V = arrow_length * np.sin([theta_coords[j][frame] for j in range(N)])

        # Update quiver positions and vectors
        quiver.set_offsets(np.c_[[x_coords[j][frame] for j in range(N)],
                                [z_coords[j][frame] for j in range(N)]])
        quiver.set_UVC(U, V)

        artists = points + [quiver]
        if source_point is not None:
            source_point.set_data([source_x[frame]], [source_y[frame]])
            artists.append(source_point)
        if source_circle is not None:
            source_circle.set_center((source_x[frame], source_y[frame]))
            artists.append(source_circle)
        return artists

    # --- Keep the animation in a variable ---
    ani = FuncAnimation(fig, update, frames=len(frame_idx), init_func=init,
                        blit=False, interval=100, repeat=True)

    return ani


def source_path(source_x, source_y, size=None):
    # static plot of the repulsion source's full trajectory - lets you check
    # how much ground it actually covered (the thing the reservoir is being
    # trained to decode) without rendering a full animation
    source_x = np.asarray(source_x)
    source_y = np.asarray(source_y)

    fig, ax = plt.subplots()
    ax.plot(source_x, source_y, color="gray", alpha=0.4, linewidth=0.5)
    sc = ax.scatter(source_x, source_y, c=np.arange(len(source_x)), cmap="viridis", s=3)
    fig.colorbar(sc, ax=ax, label="timestep")

    if size is not None:
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Repulsion source trajectory")
    ax.set_aspect("equal")
    plt.show()

    path_length = np.sum(np.hypot(np.diff(source_x), np.diff(source_y)))
    print(f"Path length: {path_length:.3f}")
    print(f"X range: [{source_x.min():.3f}, {source_x.max():.3f}]")
    print(f"Y range: [{source_y.min():.3f}, {source_y.max():.3f}]")

#%%
# u_array = []
# for i in range(200):
#     u_array.append(u(i))

# plt.plot(u_array)
# plt.xlabel("iteration")
# plt.ylabel("input")
# plt.show()

# # %%

# from matplotlib.animation import PillowWriter
# ani.save("animation.gif", writer='pillow', fps=10)
# # %%

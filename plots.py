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

def animation(x_coords, z_coords, theta_coords):
    N = np.shape(x_coords)[0]

    # --- Create figure and axis ---
    fig, ax = plt.subplots()
    ax.set_xlim(np.min(x_coords) * 0.9, np.max(x_coords) * 1.1)
    ax.set_ylim(np.min(z_coords) * 0.9, np.max(z_coords) * 1.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    # Create scatter-like markers (each node as a dot)
    points = [ax.plot([], [], "o", label=f"Node {i}")[0] for i in range(N)]

    quiver = ax.quiver([0]*N, [0]*N, [0]*N, [0]*N,
                    angles='xy', scale_units='xy', scale=1, color='r')

    arrow_length = 1

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
    ani = FuncAnimation(fig, update, frames=200, init_func=init,
                        blit=False, interval=100, repeat=True)

    return ani

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

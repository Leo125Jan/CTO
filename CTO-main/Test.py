import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the simulation area
area_size = (100, 100)

# Initialize the target's starting position
positions = [np.array([np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1])]) for _ in range(5)]

# Define the number of steps
num_steps = 1000

# Initialize the trajectory
trajectories = [[] for _ in range(5)]

# Function to perform Lévy flights
def levy_flight(num_steps, change_inv=30):

    direction_change_counters = [0] * 5  # One counter for each target
    angles = [np.random.uniform(0, 2*np.pi) for _ in range(5)]  # Initial random angles for each target

    for _ in range(num_steps):

        for i in range(5):

            if direction_change_counters[i] == change_inv:

                # Generate a random angle
                angles[i] = np.random.uniform(0, 2*np.pi)
                direction_change_counters[i] = 0

            # Generate a step length from a Lévy distribution
            step_length = np.random.pareto(a=3) + 1  # 'a' is the shape parameter

            # Calculate the new position
            new_position = positions[i] + step_length * np.array([np.cos(angles[i]), np.sin(angles[i])])

            # Apply boundary conditions
            new_position = np.maximum(new_position, [0, 0])
            new_position = np.minimum(new_position, area_size)

            # Update the position
            positions[i][:] = new_position

            # Store the new position for each target
            trajectories[i].append(new_position.tolist())

            # Store the new position
            trajectories.append(new_position.tolist())

            direction_change_counters[i] += 1

# Function to perform Brownian motion for multiple targets
def brownian_motion(num_steps, change_interval=3):

    direction_change_counters = [0] * 5  # One counter for each target

    # Generate a step from a normal distribution
    step_x = np.random.normal(loc=0, scale=1)  # 'loc' is the mean, 'scale' is the standard deviation
    step_y = np.random.normal(loc=0, scale=1)

    for _ in range(num_steps):

        for i in range(5):

            if direction_change_counters[i] == change_interval:

                # Generate a step from a normal distribution
                step_x = np.random.normal(loc=0, scale=1)  # 'loc' is the mean, 'scale' is the standard deviation
                step_y = np.random.normal(loc=0, scale=1)

                direction_change_counters[i] = 0  # Reset the counter

            # Calculate the new position for each target
            new_position = positions[i] + np.array([step_x, step_y])

            # Apply boundary conditions
            new_position = np.maximum(new_position, [0, 0])
            new_position = np.minimum(new_position, area_size)

            # Update the position for each target
            positions[i][:] = new_position

            # Store the new position for each target
            trajectories[i].append(new_position.tolist())

            # Increment the counter for each target
            direction_change_counters[i] += 1

# Perform the Lévy flight
levy_flight(num_steps)

# Perform the Brownian motion for each target
# brownian_motion(num_steps)

# Convert trajectory to numpy array for plotting
trajectories = [np.array(trajectory) for trajectory in trajectories]

# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, area_size[0])
ax.set_ylim(0, area_size[1])
lines = [ax.plot([], [], '-o', markersize=2)[0] for _ in range(5)]

# Initialization function for the animation
def init():

    for line in lines:

        line.set_data([], [])
    return lines

# Animation function which updates the plot
def animate(i):
    for j, line in enumerate(lines):
        line.set_data(trajectories[j][:i, 0], trajectories[j][:i, 1])
    return lines

# Create the animation
ani = FuncAnimation(fig, animate, init_func=init, frames=num_steps, interval=20, blit=True)

# To save the animation, uncomment the following line
# ani.save('levy_walk.mp4')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

ON = 1
OFF = 0

def image_to_grid(image_path, grid_size):
    """Converts an image to a binary grid based on its brightness."""
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((grid_size, grid_size))   # Resize to fit the grid
    img_np = np.array(img)
    
    # Convert grayscale values to binary based on brightness threshold
    grid = np.where(img_np < 128, ON, OFF)
    return grid

def random_grid(size):
    """Generates a random initial grid"""
    return np.random.choice([ON, OFF], size * size, p=[0.5, 0.5]).reshape(size, size)

def update(frameNum, img, grid, target_grid, size):
    # Check if current grid matches the target grid
    if np.array_equal(grid, target_grid):
        print("Goal reached! Stopping animation.")
        ani.event_source.stop()  # Stop the animation when the goal is reached
        return img,
    
    new_grid = grid.copy()
    for i in range(size):
        for j in range(size):
            total = int((grid[i, (j - 1) % size] + grid[i, (j + 1) % size] +
                         grid[(i - 1) % size, j] + grid[(i + 1) % size, j] +
                         grid[(i - 1) % size, (j - 1) % size] + grid[(i - 1) % size, (j + 1) % size] +
                         grid[(i + 1) % size, (j - 1) % size] + grid[(i + 1) % size, (j + 1) % size]))

            # Apply Conway's rules
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    new_grid[i, j] = OFF
            else:
                if total == 3:
                    new_grid[i, j] = ON

    # Aggressively move toward the target pattern by increasing probability
    match_probability = min(1, 0.1 + frameNum * 0.02)  # Gradually increase the match probability
    for i in range(size):
        for j in range(size):
            if np.random.rand() < match_probability:  # Higher chance of matching target as time progresses
                new_grid[i, j] = target_grid[i, j]

    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img,

# User input: specify image path and grid size
image_path = "390089_v9_bb.jpg"  # Replace with your image path
GRID_SIZE = 50

# Initialize a random grid and load the target image grid
grid = random_grid(GRID_SIZE)
target_grid = image_to_grid(image_path, GRID_SIZE)

# Animation duration settings for 15 seconds
frames = 100               # Number of frames (adjusted to fit 15 seconds)
interval = 200             # Interval between frames in milliseconds

# Set up the figure and animation
fig, ax = plt.subplots()
img = ax.imshow(grid, interpolation='nearest', cmap='binary')

# Define the animation with a stop condition
ani = animation.FuncAnimation(fig, update, fargs=(img, grid, target_grid, GRID_SIZE),
                              frames=frames, interval=interval, save_count=frames)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

ON = 1
OFF = 0

def image_to_grid(image_path, grid_size):
    """Converts an image to a binary grid based on its brightness."""
    img = Image.open(image_path).convert("L")  #L means Luminance(it changed the color of the image to represent brightness),
                                                #we used it to convert the image to grayscale
    img = img.resize((grid_size, grid_size))   # Resize to match the grid size
    img_np = np.array(img)
    
    # Convert grayscale values(bright) to binary values(ON means black and OFF means white)
    grid = np.where(img_np < 128, ON, OFF)
    return grid

def random_grid(size):
    """Generates a random initial grid"""
    return np.random.choice([ON, OFF], size * size, p=[0.5, 0.5]).reshape(size, size)

def update(frameNum, img, grid, target_grid, size):
    # Check if current grid matches the target grid. If it matches, then stops
    if np.array_equal(grid, target_grid):
        print("Goal reached! Stop animation.")
        ani.event_source.stop()  
        return img,
    
    new_grid = grid.copy()
    #Check if neighbors are alive or dead(conway) then calculate the number of alive neighbor
    for i in range(size):
        for j in range(size):
            total = int((grid[i, (j - 1) % size] + grid[i, (j + 1) % size] +
                         grid[(i - 1) % size, j] + grid[(i + 1) % size, j] +
                         grid[(i - 1) % size, (j - 1) % size] + grid[(i - 1) % size, (j + 1) % size] +
                         grid[(i + 1) % size, (j - 1) % size] + grid[(i + 1) % size, (j + 1) % size]))

            # Conway's Game of life rule 1) If the currenct cell is alive(ON), 
            # a. smaller than 2 neighbor: Die. b. 2 or 2 neighbors: alive. c.larger than 3 neighbor: Die
            # If the current cell is die -> if 3 neighbors: alive
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    new_grid[i, j] = OFF
            else:
                if total == 3:
                    new_grid[i, j] = ON

    # Calculate match_probability, it is used to determine the probability that a cell in the new grid matches the target grid
    #It starts at 10% and incleases by 2% per frameNum
    match_probability = min(1, 0.1 + frameNum * 0.02)  
    for i in range(size):
        for j in range(size):
            if np.random.rand() < match_probability:  
                new_grid[i, j] = target_grid[i, j]

    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img,

# User input
image_path = "390089_v9_bb.jpg"  # Replace with your image path

#grid size
GRID_SIZE = 50

# Initialize a random grid 
grid = random_grid(GRID_SIZE)

#load the target image grid
target_grid = image_to_grid(image_path, GRID_SIZE)


frames = 100              
interval = 200             

# Set up the figure and axes for displaying the grid
# Then display the initial grid as a binary(black and white) image
fig, ax = plt.subplots()
img = ax.imshow(grid, interpolation='nearest', cmap='binary')

# animation for our project
ani = animation.FuncAnimation(fig, update, fargs=(img, grid, target_grid, GRID_SIZE),
                              frames=frames, interval=interval, save_count=frames)

plt.show()
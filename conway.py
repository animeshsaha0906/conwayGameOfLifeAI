import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os


ON = 1
OFF = 0

"""
image_to_grid function converts an inserted image to a binary grid (black and white/ ON and OFF)
based on its brightness. We opened and resized the image by using PIL image. 
Then we converted the image into a Numpy array (for using 2D arrays efficiently). 
If the grid grayscale is less than 128, the value of the grid should be ON(1 = White), 
otherwise, its value is OFF (0 = Black)
"""
def image_to_grid(image_path, grid_size):
    img = Image.open(image_path).convert("L")  
    img = img.resize((grid_size, grid_size))  
    img_np = np.array(img)
    grid = (img_np < 128) * ON + (img_np >= 128) * OFF
    return grid




"""
random_grid function generate a random initial grid to start the game
It generates a random size x size array values between 0 and 1.For example, 
if the value is more than 0.5 then, it will be ON (white) and if it is less than 0.5, 
it will be OFF (black)

"""
def random_grid(size):
    random_array = np.random.rand(size, size)
    grid = (random_array > 0.5) * ON + (random_array <= 0.5) * OFF
    return grid


"""
Update function is our main functionality of the project. Here we have - 
1) Checked if current grid matches the target grid, if so it will stop the animation by using Matplotilib
2) Applied Conway's game of Life rules to update the state of each cell in grid
    For each cell, we calculate the sum of its neighbors and use this to determine
    if the cell will survive, die.
    # Apply Conway's rules
    # If the current cell is alive(ON) : a. smaller than 2 neighbor: Die, b. 2 or 3 neighbors: alive, c. more than 3 neighbor: die
    # If the current cell is dead(OFF) : 3 neighbor: born
3) Gradually increasing the match_probability of matching cells.Randomly updating cells to align with the target grid
    The match_probability starts at 0.1 and increase linearly with each animation frame
"""
def update(frameNum, img_plot, grid, target_grid, size):
    if np.array_equal(grid, target_grid):
        print("Goal reached! Stopping animation.")
        ani.event_source.stop()  
        return img_plot,
   
    new_grid = grid.copy()
    for i in range(size):
        for j in range(size):
            total = int((
                grid[i, (j - 1) % size] + grid[i, (j + 1) % size] +
                grid[(i - 1) % size, j] + grid[(i + 1) % size, j] +
                grid[(i - 1) % size, (j - 1) % size] + grid[(i - 1) % size, (j + 1) % size] +
                grid[(i + 1) % size, (j - 1) % size] + grid[(i + 1) % size, (j + 1) % size]
            ))
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    new_grid[i, j] = OFF
            else:
                if total == 3:
                    new_grid[i, j] = ON
    match_probability = min(1, 0.1 + frameNum * 0.02)  
    for i in range(size):
        for j in range(size):
            if np.random.rand() < match_probability:
                new_grid[i, j] = target_grid[i, j]
    img_plot.set_data(new_grid)
    grid[:] = new_grid[:]
    return img_plot,

"""
process_and_animate function initializes and runs the animation for the game.
Our goal was to -
1) Converts the input image into a binary grid by using image_to_grid()
2) Initialize the starting grid by using random_grid()
3) Set up some figures (a. Animation duration and timing, b. matplotilib )
4) Display the animation
"""
def process_and_animate(image_path):
    GRID_SIZE = 50
    target_grid = image_to_grid(image_path, GRID_SIZE)
    grid = random_grid(GRID_SIZE)
    frames = 100            
    interval = 200    
    fig, ax = plt.subplots()
    img_plot = ax.imshow(grid, interpolation='nearest', cmap='binary')
    ax.axis('off') 

    global ani
    ani = animation.FuncAnimation(
        fig, update, fargs=(img_plot, grid, target_grid, GRID_SIZE),
        frames=frames, interval=interval, save_count=frames
    )
    plt.show()


# We used Tkinter(python library for creating GUIs)
root = tk.Tk()
root.title("Conway's Game of Life Image Selector")
root.geometry("600x600")
root.selected_image_path = None
label = tk.Label(
    root, text="Select an image to start the animation", font=("Helvetica", 14)
)
label.pack(pady=10)


preselected_images = {
    "Sample Image 1": r"C:\Users\kimsh\OIP.jpg",
    "Sample Image 2": r"C:\Users\kimsh\v793-ning-24.jpg",
    "Sample Image 3": r"C:\Users\kimsh\390089_v9_bb (1).jpg",
}


thumbnail_size = (150, 150)
thumbnails = {}
#Our preselected images will be loaded by the following function. 
# This images will already be given in along with the file where we included the file path 
# in previous line preselected_images.
for name, path in preselected_images.items():
    if os.path.exists(path):
        img = Image.open(path)
        img.thumbnail(thumbnail_size)
        thumbnails[name] = ImageTk.PhotoImage(img)
    else:
        print(f"Image {path} not found.")

"""
display_delected_image function is for loading, resizing and displaying a preview of a selected image
When the user upload the image, it is displayed in the preview section
"""
def display_selected_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    preview_label.config(image=img_tk)
    preview_label.image = img_tk  

"""
choose_preselected_image function handle the selection and validation of a preselected image
It retrieves the image path based on the selected image name
"""
def choose_preselected_image(image_name):
    image_path = preselected_images.get(image_name)
    if not image_path or not os.path.exists(image_path):
        print("Selected image not found.")
        return
    display_selected_image(image_path)
    start_button.config(state="normal")
    root.selected_image_path = image_path

"""
upload_image function allows the user to upload image file using a file dialog
And stores the image path for further use during the animation process
"""
def upload_image():
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    if not image_path:
        print("No image selected.")
        return
    display_selected_image(image_path)
    start_button.config(state="normal")
    root.selected_image_path = image_path

"""
start_animation function starts the animation
1) It checks if the selected image exists
2) Then close the tkinter window and then starts the animation by calling the process_and_animate()
"""
def start_animation():
    """Starts the animation after processing the image."""
    image_path = getattr(root, 'selected_image_path', None)
    if not image_path or not os.path.exists(image_path):
        print("No image selected to animate.")
        return
    root.destroy()
    process_and_animate(image_path)


"""
This is what we did for creating the GUIs
1) create a frame for preselected image
2) Add thumbnails of preselected image (it allows the user to click on a thumbnail button to select the image)
3) Add separator
4) Add upload image button
5) Add image privew section
6) Add start animation button
"""
preselected_frame = tk.Frame(root)
preselected_frame.pack(pady=10)

for idx, (name, img_tk) in enumerate(thumbnails.items()):
    btn = tk.Button(
        preselected_frame, image=img_tk, command=lambda n=name: choose_preselected_image(n)
    )
    btn.grid(row=0, column=idx, padx=10)
    lbl = tk.Label(preselected_frame, text=name, font=("Helvetica", 10))
    lbl.grid(row=1, column=idx, padx=10)

separator = ttk.Separator(root, orient='horizontal')
separator.pack(fill='x', pady=10)


upload_button = tk.Button(
    root, text="Upload Your Own Image", command=upload_image, font=("Helvetica", 12)
)
upload_button.pack(pady=5)


preview_label = tk.Label(root)
preview_label.pack(pady=10)


start_button = tk.Button(
    root, text="Start Animation", command=start_animation, font=("Helvetica", 12), state="disabled"
)
start_button.pack(pady=5)

root.mainloop()


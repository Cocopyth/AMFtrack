import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Load an image
img = plt.imread(
    r"C:\Users\coren\AMOLF-SHIMIZU Dropbox\Corentin Bisot\temp\Img_r09_c07.png"
)

fig, ax = plt.subplots()
im = ax.imshow(
    img, extent=(0, 10, 0, 10)
)  # You might need to adjust the extent depending on your image and points

# Create some points
points = np.random.rand(5, 2) * 10  # 5 random points. Change as needed.

# Plot points as individual scatter plots and store in a list
scatter_plots = [ax.scatter(x, y) for x, y in points]

# Variable to store the selected point information
selected_point_info = {}


def onclick(event):
    global selected_point, selected_rectangle
    distances = [
        (point[0] - event.xdata) ** 2 + (point[1] - event.ydata) ** 2
        for point in points
    ]
    closest_point_index = np.argmin(distances)
    selected_point = points[closest_point_index]
    print(
        f"You clicked closest to point at coordinates ({selected_point[0]}, {selected_point[1]})"
    )

    # Draw a rectangle around the selected point, and remove the previous one
    if selected_rectangle is not None:
        selected_rectangle.remove()
    selected_rectangle = Rectangle(
        (selected_point[0] - 0.5, selected_point[1] - 0.5), 1, 1, fill=True, color="red"
    )
    ax.add_patch(selected_rectangle)
    fig.canvas.draw()


# Connect the click event with the callback function
cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()

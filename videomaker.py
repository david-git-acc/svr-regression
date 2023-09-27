import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# This is the file that generates the gifs from the images. You can't just use anim.save() since pillow seems
# to have exponential time complexity. So we split it into subdivisions, e.g 20 subdivisions and then combine the gifs
# together using online software

# In order to do this:
# -we need to use svr_surfimgmaker.py to generate the frames themselves, stored in the frames folder
# -then read them into the program using imread() and store in a list for O(1) access time
# -divide the work into the subdivisions using np.linspace(), e.g if we had [124,247,370,...] it would mean that one
# -clip would go from frames 125 to 247, then the next would go from 248 to 370, and so on
# -then once we have the clips, convert them from gif to mp4 and combine them using online software to get the final video
# (last additional step: for linkedin, will have to convert to gif so that I can also include still images in the same post)

# This is a function I copied off stackoverflow
# When you show the image, it's not the same size as the original image was, this function modifies the figsize so the image is
# in its actual size so we don't get a smaller diagram.
def display_image_in_actual_size(im_path):

    dpi = 80
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')


    
dpi=80
height, width, _ = plt.imread(f"frames/{1}.png").shape

# What size does the figure need to be in inches to fit the image?
figsize = width / float(dpi), height / float(dpi)

# Create a figure of the right size with one axes that takes up the full figure
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0, 0, 1, 1])

# Hide spines, ticks, etc.
ax.axis('off')

# This list will store the pictures
piclist = []

# The actual animation itself
def animate(i):
    
    # Clear prev axes otherwise all the pictures will accumulate on the same axes, causing lag
    plt.cla()
    
    # Hide spines, ticks, etc.
    ax.axis('off')

    img = piclist[i]
    
    # Display the image.
    ax.imshow(img)

# Divide the work into sections so that pillow doesn't take forever to save the videos
video_divisions = 20
segments = np.linspace(1,2341, video_divisions).astype(int)



# Generating the minigifs
for i in range(1,9+1):
    # Clear previous piclist so we can load the new images for the next clip
    piclist = []
    
    # Redefine the start and end points 
    a,b = (segments[i]+1,segments[i+1])

    # Add the next set of pictures to the piclist
    for video_index in range(a,b+1):
        image = plt.imread(f"frames/{video_index}.png")
        piclist.append(image)
    
    anim = FuncAnimation(fig, animate, frames=b-a+1)
    
    # Save the animation
    anim.save(f"clips/v{i}.gif", fps = min( int( 20 * np.sqrt(i)), 50)) # int( 20 * np.sqrt(i))
    
    # Progress checker
    print("done:" , i)
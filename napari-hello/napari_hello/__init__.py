# from napari.utils.notifications import show_info

# def show_hello_message():
#     show_info('Hello, world!')

# import numpy as np
# import napari

# # Test 1 -------------------------------------------------------------
# # # Create a red image
# # red_image = np.zeros((100, 100, 3), dtype=np.uint8)
# # red_image[:, :, 0] = 255  # Set red channel to maximum value

# # # Create the Napari viewer
# # viewer = napari.Viewer()

# # # Add the red image to the viewer
# # viewer.add_image(red_image, name='Red Image', colormap='red')

# # # Run the Napari event loop
# # napari.run()
# # Test 1 -------------------------------------------------------------


# # Test 2 -------------------------------------------------------------
# import napari
# import numpy as np

# # add the image
# image_size = (256, 256)
# background = np.zeros(image_size, dtype=np.uint8)

# # create some ellipses
# ellipse = np.array([[59, 222], [100, 100]])

# # ellipse2 = np.array([[165, 329], [165, 401], [400, 400], [400, 400]])

# # put both shapes in a list
# ellipses = [ellipse] #, ellipse2]

# # add an empty shapes layer
# shapes_layer = viewer.add_shapes()

# # add ellipses using their convenience method
# shapes_layer.add_ellipses(
#   ellipses,
#   edge_width=0,
#   edge_color='white',
#   face_color='white'
# )
# # Test 2 -------------------------------------------------------------



# Test 3 -------------------------------------------------------------
import numpy as np
import random
import math
from skimage.filters import gaussian
from skimage.draw import ellipse

def circles_no_overlap(circle1, circle2, overlap_percent):
    # Get the center coordinates and radii of the circles
    center1 = circle1[0]
    center2 = circle2[0]

    radius1 = max(circle1[1][0], circle1[1][1]) 
    radius2 = max(circle2[1][0], circle2[1][1])
    
    # Calculate the distance between the centers of the circles
    distance = math.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
    
    # Calculate the maximum allowable distance for the desired overlap percentage
    max_distance = (radius1 + radius2) * (1 - overlap_percent)
    
    # Check if the circles overlap
    if distance > max_distance:
        return True
    else:
        return False
    
def generate_non_overlapping_circles(image_size, p_circle_count, p_diameter_min, p_diameter_max, p_radius_funk_max, p_overlap_max):
    circles = []
    
    for _ in range(p_circle_count):
        while True:
            diameter = random.randint(p_diameter_min, p_diameter_max)
            x = random.randint(0, image_size[0] - diameter)
            y = random.randint(0, image_size[1] - diameter)

            x_vect = diameter * random.uniform(p_radius_funk_max, 1)
            y_vect = diameter * random.uniform(p_radius_funk_max, 1)
            x = x * random.uniform(p_radius_funk_max, 1)
            y = y * random.uniform(p_radius_funk_max, 1)

            circle = np.array([[x, y], [x_vect, y_vect]])

            # Check if the new circle overlaps with any previous circle
            if all(circles_no_overlap(circle, prev_circle,p_overlap_max) for prev_circle in circles):
                circles.append(circle)
                break

    return circles

def convert_shapes_to_image(shapes_layer):
    # Get the current shapes from the layer
    shapes = shapes_layer.data

    # Create an empty image with the determined size
    image = np.zeros(image_size, dtype=np.uint8)

    # Rasterize each shape onto the image
    for shape in shapes:
        x_vect = (shape[1][0]-shape[0][0])/2
        y_vect = (shape[2][1]-shape[1][1])/2
        x = shape[0][0] + x_vect
        y = shape[0][1] + y_vect

        rr, cc = ellipse(x, y, x_vect, y_vect)
        image[rr, cc] = 255

    # Create a napari image layer
    #viewer.add_image(image, colormap='gray')

    return image

p_img_side_len = 256
image_size = (p_img_side_len, p_img_side_len)
background = np.zeros(image_size, dtype=np.uint8)

# create a list to store the circles
circles = []

p_diameter_min = 10
p_diameter_max = 50
p_radius_funk_max = 0.7
p_circle_count = 50
p_overlap_max = 0.15



#TODO: Configure the distribution of the radii variation
circles = generate_non_overlapping_circles(image_size, p_circle_count, p_diameter_min, p_diameter_max, p_radius_funk_max, p_overlap_max)

#add an empty shapes layer
shapes_layer = viewer.add_shapes(visible=False)

#add circles to the shapes layer
shapes_layer.add_ellipses(
circles,
edge_width=0,
edge_color='white',
face_color='white'
)

raster_img = convert_shapes_to_image(shapes_layer)

# # Access the raw image data as a regular 2D array
print(raster_img)

# print(image_layer.data)

raster_img_blurred = gaussian(np.array(raster_img), sigma=3.0) #, preserve_range=True)
# image_layer.refresh()

print(raster_img_blurred)

viewer.add_image(raster_img_blurred, colormap='gray')

# viewer.layers.remove(shapes_layer)
# Test 3 -------------------------------------------------------------

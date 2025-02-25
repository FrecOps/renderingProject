import pyvista as pv

# Create a Plotter instance to visualize our 3D text
plotter = pv.Plotter()

# Create the 3D text mesh with specified extrusion depth and scaling
text_mesh = pv.Text3D("Hello, 3D!", depth=0.9)
text_mesh.scale([1.9, 1.9, 1.9])
plotter.add_mesh(text_mesh, color='cyan')

# Open a movie file to record the animation
plotter.open_movie("rotation_animation.mp4", quality=5)  # quality can be adjusted

# Rotate the camera in a loop and capture each frame
for _ in range(360):  # This will create a full 360° rotation
    plotter.camera.Azimuth(1)  # Rotate the camera by 1 degree
    plotter.render()           # Render the updated scene
    plotter.write_frame()      # Write the current frame to the movie

# Close the movie file
plotter.close_movie()

# Optionally, display the scene interactively after the animation is recorded
plotter.show()



from ipyleaflet import Map, WidgetControl
from pythreejs import (
    Mesh, BoxGeometry, MeshStandardMaterial, AmbientLight,
    PerspectiveCamera, Scene, Renderer, OrbitControls
)
import ipywidgets as widgets


latitude=32.971425,
longitude=-96.822403,

# Create an ipyleaflet Map centered on a sample location (latitude, longitude)
m = Map(center=(latitude, longitude), zoom=16)

# Create a simple 3D cube using pythreejs
cube = Mesh(
    geometry=BoxGeometry(width=1, height=1, depth=1),
    material=MeshStandardMaterial(color='red')
)

# Create a 3D scene containing the cube and an ambient light
scene = Scene(children=[
    cube,
    AmbientLight(color='#ffffff')
])

# Set up a perspective camera for the 3D scene
camera = PerspectiveCamera(position=[3, 3, 3], fov=75)

# Enable orbit controls for interactive rotation
controls = [OrbitControls(controlling=camera)]

# Create a Renderer for the 3D scene; adjust width and height as needed.
renderer = Renderer(camera=camera, scene=scene, controls=controls, width=400, height=400)

# Embed the 3D renderer as a widget control on the map (placed at top-right)
widget_control = WidgetControl(widget=renderer, position='topright')
m.add_control(widget_control)

# Display the map with the embedded 3D scene
m

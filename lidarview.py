import requests
import laspy
import open3d as o3d
import numpy as np

# --- Define Target Location using Latitude and Longitude ---
lat, lon = 37.5, -122.5  # Replace with your desired lat, lon coordinates
delta = 0.01             # Offset to create a bounding box around the point

south = lat - delta
north = lat + delta
west = lon - delta
east = lon + delta

# --- OpenTopography API Setup ---
API_KEY = "79e540e8f1580d7cea1bbefa659b88b0"
dataset_id = "OTNED.012021.4269.3"  # Replace with a valid dataset identifier from the data catalog

# Use the updated endpoint under the portal subdomain
api_url = "https://portal.opentopography.org/API/lidar"

# Construct the API parameters.
params = {
    "API_Key": API_KEY,
    "dataset": dataset_id,  # Required dataset identifier
    "south": south,
    "north": north,
    "west": west,
    "east": east,
    "outputFormat": "LAS"  # Requesting the LAS file format
}

print("Requesting LiDAR data from OpenTopography...")
response = requests.get(api_url, params=params)

if response.status_code == 200:
    las_filename = "downloaded.las"
    with open(las_filename, "wb") as f:
        f.write(response.content)
    print("LiDAR data downloaded successfully and saved as", las_filename)
else:
    print("Error downloading LiDAR data:", response.status_code)
    print("Response:", response.text)
    exit(1)

# --- Process and Visualize the LiDAR Data ---
# Load the downloaded LAS file using laspy
las = laspy.read(las_filename)

# Extract point coordinates
points = np.vstack((las.x, las.y, las.z)).transpose()

# Create an Open3D PointCloud object and set its points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Optionally, set grayscale colors based on elevation (z-value)
min_z = points[:, 2].min()
max_z = points[:, 2].max()
colors = (points[:, 2] - min_z) / (max_z - min_z)
pcd.colors = o3d.utility.Vector3dVector(np.repeat(colors[:, None], 3, axis=1))

# Visualize the point cloud using Open3D
o3d.visualization.draw_geometries([pcd])
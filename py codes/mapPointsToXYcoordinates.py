import json
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

# Earth's radius in meters
R = 6371000

# Load solar panel data from file.
# Each line is expected to be a JSON string representing one solar panel (a list of 4 points).
solar_panels = []
with open("mkHQ.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            # Each item should be a solar panel (list of 4 point dictionaries)
            for panel in data:
                if isinstance(panel, list) and len(panel) == 4:
                    solar_panels.append(panel)
                else:
                    print("Skipping group that does not have exactly 4 points")
        except Exception as e:
            print("Error parsing line:", e)

if not solar_panels:
    raise ValueError("No valid solar panel data found.")

# Compute a reference point (mean lat/lon) from all points.
all_points = [pt for panel in solar_panels for pt in panel]
mean_lat = sum(pt["lat"] for pt in all_points) / len(all_points)
mean_lon = sum(pt["lng"] for pt in all_points) / len(all_points)
lat0_rad = math.radians(mean_lat)

def project_point(lat, lon):
    # Equirectangular projection: convert lat/lon differences to meters.
    x = R * math.radians(lon - mean_lon) * math.cos(lat0_rad)
    y = R * math.radians(lat - mean_lat)
    return x, y

# Project each solar panel's points and compute its center.
projected_panels = []
panel_centers = []
for panel in solar_panels:
    proj = [project_point(pt["lat"], pt["lng"]) for pt in panel]
    projected_panels.append(proj)
    xs = [p[0] for p in proj]
    ys = [p[1] for p in proj]
    center = (sum(xs) / len(xs), sum(ys) / len(ys))
    panel_centers.append(center)

# Convert panel centers to a numpy array for clustering.
centers_arr = np.array(panel_centers)

# Use DBSCAN to group modules that are adjacent.
# eps defines the maximum distance (in meters) for modules to be considered connected.
eps = 10.0  # adjust as needed based on your data's spacing
db = DBSCAN(eps=eps, min_samples=1)
labels = db.fit_predict(centers_arr)

# Group modules by cluster label.
clusters = {}
for i, label in enumerate(labels):
    clusters.setdefault(label, []).append(panel_centers[i])

# For each cluster, determine the number of rows and columns.
print("Array details (Rows x Columns):")
for label in sorted(clusters):
    group = clusters[label]
    xs = [pt[0] for pt in group]
    ys = [pt[1] for pt in group]
    unique_x = sorted(set(round(x, 1) for x in xs))
    unique_y = sorted(set(round(y, 1) for y in ys))
    num_columns = len(unique_x)
    num_rows = len(unique_y)
    total_modules = len(group)
    print(f"Array {label+1}: {num_rows} rows x {num_columns} columns (total modules: {total_modules})")

# Determine, for each cluster, the "first module" as the module with the smallest (x,y) coordinate.
# Here we sort by x, then y.
first_module_per_cluster = {}
for label in clusters:
    indices = [i for i, lab in enumerate(labels) if lab == label]
    min_index = min(indices, key=lambda i: (panel_centers[i][0], panel_centers[i][1]))
    first_module_per_cluster[label] = panel_centers[min_index]

# Plot the modules with different colors for each array.
plt.figure(figsize=(10, 8))
cmap = plt.cm.get_cmap("tab10", len(clusters))
for i, proj in enumerate(projected_panels):
    xs = [p[0] for p in proj] + [proj[0][0]]  # closing the polygon
    ys = [p[1] for p in proj] + [proj[0][1]]
    cluster_label = labels[i]
    plt.plot(xs, ys, marker="o", color=cmap(cluster_label), linewidth=2)
    # Annotate each module with its panel number.
    cx, cy = panel_centers[i]
    plt.text(cx, cy, str(i+1), fontsize=8, ha="center", va="center", color="black")

# Annotate each cluster with its array name and module count.
for label, group in clusters.items():
    group_arr = np.array(group)
    cluster_center = group_arr.mean(axis=0)
    plt.text(cluster_center[0], cluster_center[1], f"Array {label+1}\n({len(group)})",
             fontsize=12, fontweight="bold", color="red", ha="center", va="center")

# Mark the first module (smallest (x,y) in each array) with a star marker.
for label, point in first_module_per_cluster.items():
    plt.scatter(point[0], point[1], s=200, color='k', marker='*', zorder=10)
    plt.text(point[0], point[1], f"First {label+1}", fontsize=10, color='blue', ha='right', va='bottom')

plt.title("Modules Grouped into Arrays with First Module Marked (Smallest (x,y))")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.axis("equal")
plt.grid(True)
plt.show()
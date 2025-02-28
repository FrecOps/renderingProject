import math
import numpy as np
import pyvista as pv  # pip install pyvista

flamingo_pink = '#FC8EAC'


##############################################################################
# 1) Utility: building PyVista geometry for "beam"-style shapes
##############################################################################

def create_beam_pv(start, end, width, thickness):
    """
    Return a PyVista mesh (PolyData) for a rectangular beam from 'start' to 'end'
    with cross-section = (width x thickness).
    Oriented perpendicular to the (start->end) axis.
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-9:
        raise ValueError("create_beam_pv: zero-length beam")

    d = direction / length
    ref = np.array([0, 0, 1], dtype=float)
    if abs(np.dot(d, ref)) > 0.99:
        ref = np.array([0, 1, 0], dtype=float)

    v = np.cross(d, ref)
    v /= np.linalg.norm(v)
    w = np.cross(d, v)
    w /= np.linalg.norm(w)

    a = width / 2.0
    b = thickness / 2.0

    # 8 corners: 4 at start and 4 at end
    corners_start = [
        start + (+a * v + b * w),
        start + (-a * v + b * w),
        start + (-a * v - b * w),
        start + (+a * v - b * w),
    ]
    corners_end = [
        end + (+a * v + b * w),
        end + (-a * v + b * w),
        end + (-a * v - b * w),
        end + (+a * v - b * w),
    ]
    vertices = np.array(corners_start + corners_end, dtype=float)
    faces_array = np.hstack([
        [4, 0, 1, 2, 3],  # near face
        [4, 4, 5, 6, 7],  # far face
        [4, 0, 1, 5, 4],  # side
        [4, 1, 2, 6, 5],
        [4, 2, 3, 7, 6],
        [4, 3, 0, 4, 7]
    ])
    return pv.PolyData(vertices, faces_array)


def create_footing_pv(center, width=0.5, length=0.5, depth=0.3):
    """
    Footing block from z=0 down to z=-depth, ~ width x length in XY-plane.
    """
    top = np.array([center[0], center[1], 0], dtype=float)
    bottom = top + np.array([0, 0, -depth], dtype=float)
    return create_beam_pv(top, bottom, length, width)


def create_column_pv(base_point, height, col_width=0.2, col_thickness=0.2):
    """
    Column from base_point to base_point+(0,0,height) with a cross-section defined by col_width x col_thickness.
    """
    top_pt = base_point + np.array([0, 0, height], dtype=float)
    return create_beam_pv(base_point, top_pt, col_width, col_thickness)



def create_purlin_pv(start, end, purlin_width=0.1, purlin_thickness=0.1):
    """
    c-purlin as a rectangular beam from 'start' to 'end' with given width and thickness.
    """
    return create_beam_pv(start, end, purlin_width, purlin_thickness)


##############################################################################
# 2) Modules as thin rectangular panels (representing solar panels)
##############################################################################

def create_module_pv(cornerA, cornerB, cornerC, cornerD):
    """
    Build a single quad PyVista mesh using the four corners in 3D.
    A->B->C->D is a rectangle or any quad.
    """
    vertices = np.array([cornerA, cornerB, cornerC, cornerD], dtype=float)
    faces_array = np.hstack([[4, 0, 1, 2, 3]])
    return pv.PolyData(vertices, faces_array)


##############################################################################
# 3) Putting it all together: 2D grid of columns, purlins, and modules
##############################################################################
def design_carport_2Dgrid_pv(
        # module geometry
        module_width=1.134,
        module_height=2.462,
        n_mods_x=5,
        n_mods_y=3,
        mod_gap=0.05,
        # number of sub-rows in Y (n_strings_y => lines_y = n_strings_y+1)
        n_strings_y=2,
        # tilt
        tilt_degs=10.0,
        col_base_z=0.0,
        front_col_top_z=3.0,
        final_col_top_z=None,
        # footings/columns
        footing_size=0.5,
        footing_depth=0.3,
        max_column_spacing=5.8,
        min_n_columns=2,
        # purlin splice
        purlin_splice_interval=5.0,
        z_above_purlins=0.1,
        # Module clearance (above columns)
        module_clearance=0.15,
        # New parameters for column and purlin dimensions
        column_width=0.2,
        column_thickness=0.2,
        purlin_width=0.1,
        purlin_thickness=0.1,
        # Colors
        color_footing='saddlebrown',
        color_column='silver',
        color_purlin='gray',
        color_module='lightskyblue'
):
    """
    Build a 2D grid of columns, footings, purlins, and modules.
    """
    plotter = pv.Plotter()
    tilt_radians = math.radians(tilt_degs)

    solar_panel_actors = []  # modules
    racking_actors = []  # columns, footings, purlins

    total_x = n_mods_x * module_width + (n_mods_x - 1) * mod_gap
    n_cols_x = max(min_n_columns, int(math.ceil(total_x / max_column_spacing)) + 1)// 2# COLUMNS IN THE X-AXIS (//)-means interger division
    col_spacing_x = total_x / (n_cols_x - 1) if n_cols_x > 1 else total_x
    total_y = n_mods_y * module_height + (n_mods_y - 1) * mod_gap
    lines_y = n_strings_y + 1

    if final_col_top_z is None:
        final_col_top_z = front_col_top_z + total_y * math.sin(tilt_radians)

    def line_j_yz(j):
        frac = 0 if n_strings_y < 1 else j / float(n_strings_y)
        y_ = frac * total_y
        z_ = (1 - frac) * front_col_top_z + frac * final_col_top_z
        return y_, z_

    # Build columns & footings
    col_positions = []
    for j in range(lines_y):
        y_j, top_zj = line_j_yz(j)
        row_positions = []
        for i in range(n_cols_x):
            x_i = i * col_spacing_x
            # Footing
            foot_mesh = create_footing_pv([x_i, y_j], footing_size, footing_size, footing_depth)
            actor = plotter.add_mesh(foot_mesh, color=color_footing, show_edges=True)
            racking_actors.append(actor)
            # Column using updated dimensions
            height = top_zj - col_base_z
            col_mesh = create_column_pv(np.array([x_i, y_j, col_base_z]), height,
                                        col_width=column_width, col_thickness=column_thickness)
            actor = plotter.add_mesh(col_mesh, color=color_column, show_edges=True)
            racking_actors.append(actor)
            row_positions.append([x_i, y_j, top_zj])
        col_positions.append(row_positions)

    # Add purlins using the updated dimensions
    def add_purlin_segment(ptA, ptB):
        dist = np.linalg.norm(np.array(ptB) - np.array(ptA))
        if dist <= purlin_splice_interval:
            pm = create_purlin_pv(ptA, ptB, purlin_width=purlin_width, purlin_thickness=purlin_thickness)
            actor = plotter.add_mesh(pm, color=color_purlin, show_edges=True)
            racking_actors.append(actor)
        else:
            segs = int(math.ceil(dist / purlin_splice_interval))
            for s in range(segs):
                alpha = s / segs
                beta = min(1.0, (s + 1) / segs)
                segA = ptA + alpha * (ptB - ptA)
                segB = ptA + beta * (ptB - ptA)
                pm = create_purlin_pv(segA, segB, purlin_width=purlin_width, purlin_thickness=purlin_thickness)
                actor = plotter.add_mesh(pm, color=color_purlin, show_edges=True)
                racking_actors.append(actor)

    # Horizontal purlins
    for j in range(lines_y):
        rowp = col_positions[j]
        for i in range(n_cols_x - 1):
            ptA = np.array(rowp[i])
            ptB = np.array(rowp[i + 1])
            add_purlin_segment(ptA, ptB)

    # Vertical purlins
    for j in range(lines_y - 1):
        rowA = col_positions[j]
        rowB = col_positions[j + 1]
        for i in range(n_cols_x):
            ptA = np.array(rowA[i])
            ptB = np.array(rowB[i])
            add_purlin_segment(ptA, ptB)

    # Place modules (solar panels)
    mod_x_sp = (total_x - n_mods_x * module_width) / (n_mods_x - 1) if n_mods_x > 1 else 0
    mod_y_sp = (total_y - n_mods_y * module_height) / (n_mods_y - 1) if n_mods_y > 1 else 0

    def module_z(y):
        frac = y / total_y if total_y > 1e-9 else 0
        base_z = (1 - frac) * front_col_top_z + frac * final_col_top_z
        return base_z + module_clearance

    for ix in range(n_mods_x):
        x_left = ix * (module_width + mod_x_sp)
        x_right = x_left + module_width
        for iy in range(n_mods_y):
            y_front = iy * (module_height + mod_y_sp)
            y_back = y_front + module_height
            A = [x_left, y_front, module_z(y_front)]
            B = [x_right, y_front, module_z(y_front)]
            C = [x_right, y_back, module_z(y_back)]
            D = [x_left, y_back, module_z(y_back)]
            module_points = np.array([A, B, C, D], dtype=float)
            faces = np.hstack([[4, 0, 1, 2, 3]])
            actor = plotter.add_mesh(pv.PolyData(module_points, faces),
                                     color=color_module, show_edges=True, opacity=0.9)
            solar_panel_actors.append(actor)

    # Additional code for grid, text, callbacks, etc.
    grid_mesh = pv.Plane(center=(total_x / 2, total_y / 2, 0),
                         direction=(0, 0, 1),
                         i_size=total_x, j_size=total_y,
                         i_resolution=10, j_resolution=10)
    grid_actor = plotter.add_mesh(grid_mesh, style='wireframe', color='gray', opacity=0.5)
    plotter.grid_actor = grid_actor

    # Attach actor groups to the plotter for later access
    plotter.solar_panel_actors = solar_panel_actors
    plotter.racking_actors = racking_actors

    # (Callback functions and other scene enhancements remain unchanged)
    return plotter



##############################################################################
# Main
##############################################################################

southArray  = [61,8] #Addison Project
westArray = [68,30] #Addison Project

selected  = southArray

selectedx = selected[0]
selectedy = selected[1]

if __name__ == "__main__":
    plotter = design_carport_2Dgrid_pv(
        n_mods_x=selectedx,
        n_mods_y=selectedy,
        tilt_degs=10.0,
        front_col_top_z=4.5,
        purlin_splice_interval=2.0,
        z_above_purlins=0.15,
        module_clearance=0.15,
        column_width=0.25,        # custom column width
        column_thickness=0.25,     # custom column thickness
        purlin_width=0.12,         # custom purlin width
        purlin_thickness=0.12,     # custom purlin thickness
        color_footing=flamingo_pink,
        color_column=flamingo_pink,
        color_purlin=flamingo_pink,
        color_module=flamingo_pink
    )

    # Create and store the bounding box actor (labels)
    bounds_actor = plotter.show_bounds(grid='front', location='outer', font_size=8, color='black')
    plotter.bounds_actor = bounds_actor

    # Define interactive callbacks
    def toggle_solar_panels():
        for actor in plotter.solar_panel_actors:
            actor.SetVisibility(not actor.GetVisibility())
        plotter.render()

    def toggle_racking():
        for actor in plotter.racking_actors:
            actor.SetVisibility(not actor.GetVisibility())
        plotter.render()

    def toggle_labels():
        if hasattr(plotter, 'bounds_actor'):
            current = plotter.bounds_actor.GetVisibility()
            plotter.bounds_actor.SetVisibility(not current)
            plotter.render()

    def toggle_grid():
        if hasattr(plotter, 'grid_actor'):
            current = plotter.grid_actor.GetVisibility()
            plotter.grid_actor.SetVisibility(not current)
            plotter.render()

    def reset_view():
        plotter.reset_camera()

    def rotate_animation():
        plotter.open_movie("WestArray.mp4", quality=5)
        rotation_step = 0.1  # degrees per frame
        num_steps = int(360 / rotation_step)
        for _ in range(num_steps):
            plotter.camera.Azimuth(rotation_step)
            plotter.render()
            plotter.write_frame()
        for _ in range(num_steps):
            plotter.camera.Elevation(rotation_step)
            plotter.render()
            plotter.write_frame()
        for _ in range(num_steps):
            plotter.camera.Roll(rotation_step)
            plotter.render()
            plotter.write_frame()
        plotter.close_movie()

    # Add key events for interactivity BEFORE calling plotter.show()
    plotter.add_key_event('s', toggle_solar_panels)
    plotter.add_key_event('r', toggle_racking)
    plotter.add_key_event('l', toggle_labels)
    plotter.add_key_event('g', toggle_grid)
    plotter.add_key_event('v', reset_view)
    plotter.add_key_event('p', rotate_animation)

    # Reset camera and start the interactive session
    plotter.reset_camera()
    plotter.show()
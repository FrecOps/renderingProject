import math
import numpy as np
import pyvista as pv  # pip install pyvista

flamingo_pink = '#FC8EAC'

##############################################################################
# 1) Utility: building PyVista geometry for "beam"-style shapes
##############################################################################
def create_beam_pv(start, end, width, thickness):
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
        [4, 0, 1, 2, 3],
        [4, 4, 5, 6, 7],
        [4, 0, 1, 5, 4],
        [4, 1, 2, 6, 5],
        [4, 2, 3, 7, 6],
        [4, 3, 0, 4, 7]
    ])
    return pv.PolyData(vertices, faces_array)

def create_footing_pv(center, width=0.5, length=0.5, depth=0.3):
    top = np.array([center[0], center[1], 0], dtype=float)
    bottom = top + np.array([0, 0, -depth], dtype=float)
    return create_beam_pv(top, bottom, length, width)

def create_column_pv(base_point, height, col_width=0.2, col_thickness=0.2):
    top_pt = base_point + np.array([0, 0, height], dtype=float)
    return create_beam_pv(base_point, top_pt, col_width, col_thickness)

def create_purlin_pv(start, end, purlin_width=0.1, purlin_thickness=0.1):
    return create_beam_pv(start, end, purlin_width, purlin_thickness)

##############################################################################
# 2) Modules as thin rectangular panels (representing solar panels)
##############################################################################
def create_module_pv(cornerA, cornerB, cornerC, cornerD):
    vertices = np.array([cornerA, cornerB, cornerC, cornerD], dtype=float)
    faces_array = np.hstack([[4, 0, 1, 2, 3]])
    return pv.PolyData(vertices, faces_array)

##############################################################################
# 3) Putting it all together: 2D grid of columns, purlins, modules, and optional sub-purlins
##############################################################################
def design_carport_2Dgrid_pv(
        module_width=1.134,
        module_height=2.462,
        n_mods_x=5,
        n_mods_y=3,
        mod_gap=0.05,
        n_strings_y=2,
        tilt_degs=10.0,
        col_base_z=0.0,
        front_col_top_z=3.0,
        final_col_top_z=None,
        footing_size=0.5,
        footing_depth=0.3,
        max_column_spacing=5.8,
        min_n_columns=2,
        purlin_splice_interval=5.0,
        z_above_purlins=0.1,
        module_clearance=0.16,
        column_width=0.2,
        column_thickness=0.2,
        purlin_width=0.1,
        purlin_thickness=0.1,
        add_sub_purlins=False,
        sub_purlin_offset=0.5,  # Unused in this approach
        color_footing='saddlebrown',
        color_column='silver',
        color_purlin='gray',
        color_module='lightskyblue'
):
    plotter = pv.Plotter()
    tilt_radians = math.radians(tilt_degs)
    solar_panel_actors = []
    racking_actors = []
    module_subpurlin_actors = []  # For module sub-purlins

    total_x = n_mods_x * module_width + (n_mods_x - 1) * mod_gap
    n_cols_x = max(min_n_columns, int(math.ceil(total_x / max_column_spacing)) + 1) // 2
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

    # Helper: compute module mounting level (this returns the solar panel top z)
    def module_z(y):
        frac = y / total_y if total_y > 1e-9 else 0
        base_z = (1 - frac) * front_col_top_z + frac * final_col_top_z
        return base_z + module_clearance

    # Build columns & footings
    col_positions = []
    for j in range(lines_y):
        y_j, top_zj = line_j_yz(j)
        row_positions = []
        for i in range(n_cols_x):
            x_i = i * col_spacing_x
            foot_mesh = create_footing_pv([x_i, y_j], footing_size, footing_size, footing_depth)
            actor = plotter.add_mesh(foot_mesh, color=color_footing, show_edges=True)
            racking_actors.append(actor)
            height = top_zj - col_base_z
            col_mesh = create_column_pv(np.array([x_i, y_j, col_base_z]), height,
                                        col_width=column_width, col_thickness=column_thickness)
            actor = plotter.add_mesh(col_mesh, color=color_column, show_edges=True)
            racking_actors.append(actor)
            row_positions.append([x_i, y_j, top_zj])
        col_positions.append(row_positions)

    # Add main purlins (connecting column tops)
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

    for j in range(lines_y):
        rowp = col_positions[j]
        for i in range(n_cols_x - 1):
            ptA = np.array(rowp[i])
            ptB = np.array(rowp[i + 1])
            add_purlin_segment(ptA, ptB)
    for j in range(lines_y - 1):
        rowA = col_positions[j]
        rowB = col_positions[j + 1]
        for i in range(n_cols_x):
            ptA = np.array(rowA[i])
            ptB = np.array(rowB[i])
            add_purlin_segment(ptA, ptB)

    # --- New Section: Add module sub-purlins along the y-axis ---
    # For each solar module cell, add 2 sub-purlins that run horizontally (from x_left to x_right).
    # Their top face is set to be exactly at the solar panel top.
    # We achieve this by computing panel_top = module_z(mid_y) and then setting:
    #    sub_purlin_start = [x_left, sub_y, panel_top - (purlin_thickness/2)]
    #    sub_purlin_end   = [x_right, sub_y, panel_top - (purlin_thickness/2)]
    # so that the beam (which is centered on the line) will have its top at panel_top.
    mod_x_sp = (total_x - n_mods_x * module_width) / (n_mods_x - 1) if n_mods_x > 1 else 0
    # --- New Section: Add continuous module sub-purlins along the x-axis ---
    # For each module row (iy), we create two continuous beams spanning from x=0 to x=total_x.
    # Their y-positions are set relative to the module row center (offset ±0.7), and their top is flush with the solar panel top.
    # The z-value is set to panel_top - (purlin_thickness/2), so the beam's top face is exactly at the panel top.
    mod_y_sp = (total_y - n_mods_y * module_height) / (n_mods_y - 1) if n_mods_y > 1 else 0

    for iy in range(n_mods_y):
        # Compute the y-range for this row:
        y_front = iy * (module_height + mod_y_sp)
        y_back = y_front + module_height
        mid_y = (y_front + y_back) / 2.0
        # Choose the two vertical offsets relative to the row center (±0.7 yields 1.4 spacing)
        offset = 0.7
        sub_y1 = mid_y - offset
        sub_y2 = mid_y + offset
        # Compute the solar panel top (the mounting level) at the row's center:
        panel_top = module_z(mid_y)
        # Set the sub-purlin's z so that its top face is flush with the panel top.
        z_val = panel_top - (purlin_thickness / 2.0)
        # For a continuous beam, span the full x extent:
        x_left = 0.0
        x_right = total_x
        # Create the two continuous beams:
        sub_beam1 = create_purlin_pv([x_left, sub_y1, z_val], [x_right, sub_y1, z_val],
                                     purlin_width, purlin_thickness)
        sub_beam2 = create_purlin_pv([x_left, sub_y2, z_val], [x_right, sub_y2, z_val],
                                     purlin_width, purlin_thickness)
        actor1 = plotter.add_mesh(sub_beam1, color=flamingo_pink, show_edges=True)
        actor2 = plotter.add_mesh(sub_beam2, color=flamingo_pink, show_edges=True)
        module_subpurlin_actors.extend([actor1, actor2])
    plotter.module_subpurlin_actors = module_subpurlin_actors
    # --- End new section ---

    # Place modules (solar panels)
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

    grid_mesh = pv.Plane(center=(total_x / 2, total_y / 2, 0),
                         direction=(0, 0, 1),
                         i_size=total_x, j_size=total_y,
                         i_resolution=10, j_resolution=10)
    grid_actor = plotter.add_mesh(grid_mesh, style='wireframe', color='gray', opacity=0.5)
    plotter.grid_actor = grid_actor

    # Watermark text "MARY KAY"
    y_value = total_y / 2
    frac = y_value / total_y if total_y > 1e-9 else 0
    base_z = (1 - frac) * front_col_top_z + frac * final_col_top_z
    mid_point = (total_x / 2, y_value, base_z + module_clearance)
    text_mesh = pv.Text3D("MARY KAY", depth=0.0)
    text_mesh.scale([10, 10, 10])
    text_center = text_mesh.center
    translation = np.array(mid_point) - np.array(text_center)
    text_mesh.translate(translation, inplace=True)
    text_mesh.rotate_x(tilt_degs, point=mid_point, inplace=True)
    plotter.add_mesh(text_mesh, color='white')

    plotter.solar_panel_actors = solar_panel_actors
    plotter.racking_actors = racking_actors

    return plotter

##############################################################################
# Main
##############################################################################
if __name__ == "__main__":
    plotter = design_carport_2Dgrid_pv(
        n_mods_x=68,
        n_mods_y=30,
        tilt_degs=2.5,
        front_col_top_z=4.5,
        purlin_splice_interval=3.0,
        z_above_purlins=0.15,
        module_clearance=0.16,
        column_width=0.75,
        column_thickness=0.75,
        purlin_width=0.35,
        purlin_thickness=0.35,
        color_footing=flamingo_pink,
        color_column=flamingo_pink,
        color_purlin=flamingo_pink,
        color_module=flamingo_pink,
        add_sub_purlins=True,
        sub_purlin_offset=0.5
    )

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

    def toggle_module_subpurlins():
        for actor in plotter.module_subpurlin_actors:
            actor.SetVisibility(not actor.GetVisibility())
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

    # Register key events BEFORE calling show()
    plotter.add_key_event('s', toggle_solar_panels)
    plotter.add_key_event('r', toggle_racking)
    plotter.add_key_event('l', toggle_labels)
    plotter.add_key_event('g', toggle_grid)
    plotter.add_key_event('v', reset_view)
    plotter.add_key_event('p', rotate_animation)
    plotter.add_key_event('d', toggle_module_subpurlins)

    plotter.reset_camera()
    plotter.show()
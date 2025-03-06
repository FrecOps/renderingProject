import math
import numpy as np
import trimesh
from shapely.geometry import Polygon



##############################################################################
# 1) Beam utility with Trimesh
##############################################################################

def create_beam_tm(start, end, width, thickness):
    """
    Returns a Trimesh box that spans from `start` to `end`,
    with cross-section = (width x thickness) oriented perpendicular
    to (end - start).
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-9:
        raise ValueError("Zero-length beam")

    # Create a box aligned with +X of size (length, width, thickness)
    box = trimesh.creation.box(extents=(length, width, thickness))

    # Shift so that one end is at (0,0,0) instead of centered on origin
    box.apply_translation((length / 2.0, 0, 0))

    # Rotate the box so that its X-axis aligns with the beam direction
    from trimesh.transformations import rotation_matrix
    x_axis = np.array([1, 0, 0], dtype=float)
    beam_dir = direction / length

    # angle between x_axis and beam_dir
    dot_val = np.clip(np.dot(x_axis, beam_dir), -1.0, 1.0)
    angle = math.acos(dot_val)

    rot_axis = np.cross(x_axis, beam_dir)
    norm_ax = np.linalg.norm(rot_axis)
    if norm_ax < 1e-9:
        # Parallel or anti-parallel
        rot_axis = np.array([0, 0, 1], dtype=float)  # arbitrary
    else:
        rot_axis /= norm_ax

    rot_mat_4x4 = rotation_matrix(angle, rot_axis)
    box.apply_transform(rot_mat_4x4)

    # Finally translate the box so that "start" is at the correct global location
    box.apply_translation(start)

    return box


##############################################################################
# 2) Footings, columns, purlins, modules
##############################################################################

def create_footing_tm(center, width=0.5, length=0.5, depth=0.3):
    """
    Create a footing from z=0 down to z=-depth, with plan view ~ (length x width).
    """
    top = np.array([center[0], center[1], 0], dtype=float)
    bottom = top + np.array([0, 0, -depth], dtype=float)
    return create_beam_tm(top, bottom, length, width)


def create_column_tm(base_point, height, col_width=0.2, col_thickness=0.2):
    """
    Create a vertical column from base_point to base_point+(0,0,height).
    """
    top_pt = np.array(base_point, dtype=float) + np.array([0, 0, height], dtype=float)
    return create_beam_tm(base_point, top_pt, col_width, col_thickness)


def create_purlin_tm(start, end, purlin_width=0.1, purlin_thickness=0.1):
    """
    Create a rectangular beam from start->end with cross-section purlin_width x purlin_thickness.
    """
    return create_beam_tm(start, end, purlin_width, purlin_thickness)


def create_module_tm(cornerA, cornerB, cornerC, cornerD, thickness=0.01):
    """
    Create a thin rectangular panel using corner A->B->C->D in 3D.
    We'll do a simple approach: measure the bounding box in X and Y,
    extrude a 2D rectangle by 'thickness' in the normal direction.
    """
    # We'll compute dimension from corners, but let's assume they're
    # basically a rectangle in the plane.
    cornerA = np.array(cornerA, dtype=float)
    cornerB = np.array(cornerB, dtype=float)
    cornerC = np.array(cornerC, dtype=float)
    # The 4th cornerD is not used for dimension, but let's keep it consistent.

    AB = cornerB - cornerA
    AD = cornerD - cornerA
    # We'll interpret AB as local x direction, AD as local y direction
    width = np.linalg.norm(AB)
    height = np.linalg.norm(AD)

    # Make a 2D polygon in local coords
    polygon_2d = np.array([
        [0.0, 0.0],
        [width, 0.0],
        [width, height],
        [0.0, height]
    ])

    # polygon_2d is your Nx2 NumPy array of 2D vertices
    poly_2d = Polygon(polygon_2d)

    # Extrude that 2D polygon by thickness
    panel = trimesh.creation.extrude_polygon(poly_2d, thickness, engine="earcut")

    # Now we must align panel's local X->AB, local Y->AD, and place it at cornerA
    # 1) The extrude_polygon puts the "plane" in XY, normal in +Z
    #    We'll figure out the actual normal from AB x AD.
    from trimesh.transformations import rotation_matrix, translation_matrix

    # normal vector from cross(AB, AD):
    normal = np.cross(AB, AD)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-9:
        raise ValueError("Module has zero area (AB x AD ~ 0)")

    normal /= norm_len

    # By default, extrude_polygon is normal = +Z => local Z. We'll rotate that +Z to align with 'normal'
    # angle:
    dot_val = np.clip(np.dot([0, 0, 1], normal), -1.0, 1.0)
    angle = math.acos(dot_val)
    rot_axis = np.cross([0, 0, 1], normal)
    if np.linalg.norm(rot_axis) < 1e-9:
        rot_axis = np.array([1, 0, 0], dtype=float)
    else:
        rot_axis /= np.linalg.norm(rot_axis)
    R = rotation_matrix(angle, rot_axis)
    panel.apply_transform(R)

    # Now let's align the local X->AB direction. After normal alignment, local X might not match AB.
    # We can do a further rotation in the plane of the panel if desired.
    # For a rectangle, that gets more complex. Let's skip second rotation for simplicity.
    #
    # Finally, translate to cornerA in 3D
    # The extruded shape currently has a corner at (0,0,0).
    # But after rotation, that corner might have moved.
    # We can measure the bounding box min corner after rotation,
    # or we can transform in two steps (plane alignment, then plane translation).

    # Easiest: do a simpler approach:
    #   - Place the center of the panel at (width/2, height/2, 0) in local coords
    #   - After normal rotation, move it so (0,0,0) in local -> cornerA in global.
    # We'll do a direct translation now:
    T = translation_matrix(cornerA)
    panel.apply_transform(T)

    return panel


##############################################################################
# 3) Full Carport: columns, footings, purlins, sub-purlins, modules
##############################################################################
def design_carport_2Dgrid_tm(
        # module geometry
        module_width=1.134,
        module_height=2.462,
        n_mods_x=5,
        n_mods_y=3,
        mod_gap=0.05,
        # sub-rows in Y
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
        # module clearance
        module_clearance=0.15,
        # dims
        column_width=0.2,
        column_thickness=0.2,
        purlin_width=0.1,
        purlin_thickness=0.1,
        # sub-purlins
        add_sub_purlins=True,
        sub_purlin_offset=0.7,
):
    """
    Return a single Trimesh representing the entire carport 2D grid:
      - footings, columns
      - purlins
      - optional sub-purlins
      - solar modules
    """
    tilt_radians = math.radians(tilt_degs)

    # Compute total X, Y extents
    total_x = n_mods_x * module_width + (n_mods_x - 1) * mod_gap
    total_y = n_mods_y * module_height + (n_mods_y - 1) * mod_gap

    # number of columns in x
    n_cols_x = max(min_n_columns, int(math.ceil(total_x / max_column_spacing)) + 1) // 2
    if n_cols_x < 1:
        n_cols_x = 1
    col_spacing_x = total_x / (n_cols_x - 1) if n_cols_x > 1 else total_x

    lines_y = n_strings_y + 1

    if final_col_top_z is None:
        final_col_top_z = front_col_top_z + total_y * math.sin(tilt_radians)

    def line_j_yz(j):
        frac = 0 if n_strings_y < 1 else j / float(n_strings_y)
        y_ = frac * total_y
        z_ = (1 - frac) * front_col_top_z + frac * final_col_top_z
        return y_, z_

    # We store all shapes in a list, then merge at the end
    all_meshes = []

    # Helper to splice purlins if needed
    def add_purlin_segments(ptA, ptB):
        ptA = np.array(ptA, dtype=float)
        ptB = np.array(ptB, dtype=float)
        dist = np.linalg.norm(ptB - ptA)
        if dist <= purlin_splice_interval:
            beam = create_purlin_tm(ptA, ptB, purlin_width, purlin_thickness)
            all_meshes.append(beam)
        else:
            segs = int(math.ceil(dist / purlin_splice_interval))
            for s in range(segs):
                alpha = s / segs
                beta = min(1.0, (s + 1) / segs)
                segA = ptA + alpha * (ptB - ptA)
                segB = ptA + beta * (ptB - ptA)
                beam = create_purlin_tm(segA, segB, purlin_width, purlin_thickness)
                all_meshes.append(beam)

    # Build columns & footings
    col_positions = []
    for j in range(lines_y):
        y_j, top_zj = line_j_yz(j)
        row_positions = []
        for i in range(n_cols_x):
            x_i = i * col_spacing_x
            # Footing
            footing_mesh = create_footing_tm([x_i, y_j], footing_size, footing_size, footing_depth)
            all_meshes.append(footing_mesh)

            # Column
            height = top_zj - col_base_z
            col_mesh = create_column_tm([x_i, y_j, col_base_z],
                                        height,
                                        column_width,
                                        column_thickness)
            all_meshes.append(col_mesh)

            row_positions.append([x_i, y_j, top_zj])
        col_positions.append(row_positions)

    # Add main purlins
    # Horizontal (in x) for each y-line
    for j in range(lines_y):
        rowp = col_positions[j]
        for i in range(n_cols_x - 1):
            ptA = np.array(rowp[i])
            ptB = np.array(rowp[i + 1])
            add_purlin_segments(ptA, ptB)

    # Vertical (in y) between consecutive rows
    for j in range(lines_y - 1):
        rowA = col_positions[j]
        rowB = col_positions[j + 1]
        for i in range(n_cols_x):
            ptA = np.array(rowA[i])
            ptB = np.array(rowB[i])
            add_purlin_segments(ptA, ptB)

    # Optionally add sub-purlins
    if add_sub_purlins and n_mods_y > 0:
        mod_y_sp = 0.0
        if n_mods_y > 1:
            mod_y_sp = (total_y - n_mods_y * module_height) / (n_mods_y - 1)

        for iy in range(n_mods_y):
            y_front = iy * (module_height + mod_y_sp)
            y_back = y_front + module_height
            mid_y = 0.5 * (y_front + y_back)

            # sub_purlin_offset up/down from mid_y
            sub_y1 = mid_y - sub_purlin_offset
            sub_y2 = mid_y + sub_purlin_offset

            # top z at that y
            frac = mid_y / total_y if total_y > 1e-9 else 0
            top_z = (1 - frac) * front_col_top_z + frac * final_col_top_z
            # place sub-purlin so top is at top_z => center is top_z - purlin_thickness/2
            z_val = top_z + module_clearance - (purlin_thickness / 2.0)

            x_left = 0.0
            x_right = total_x
            sp1 = create_purlin_tm([x_left, sub_y1, z_val],
                                   [x_right, sub_y1, z_val],
                                   purlin_width, purlin_thickness)
            sp2 = create_purlin_tm([x_left, sub_y2, z_val],
                                   [x_right, sub_y2, z_val],
                                   purlin_width, purlin_thickness)
            all_meshes.append(sp1)
            all_meshes.append(sp2)

    # Place modules
    mod_x_sp = (total_x - n_mods_x * module_width) / (n_mods_x - 1) if n_mods_x > 1 else 0
    mod_y_sp = (total_y - n_mods_y * module_height) / (n_mods_y - 1) if n_mods_y > 1 else 0

    def module_z(y):
        frac = y / total_y if total_y > 1e-9 else 0
        col_top = (1 - frac) * front_col_top_z + frac * final_col_top_z
        return col_top + module_clearance

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
            mod_mesh = create_module_tm(A, B, C, D, thickness=0.02)
            all_meshes.append(mod_mesh)

    # Merge all into one combined Trimesh
    if not all_meshes:
        return None

    combined = trimesh.util.concatenate(all_meshes)
    return combined


##############################################################################
# 4) Main: choose array size, build the carport, export as glTF
##############################################################################
if __name__ == "__main__":
    # Example "southArray" or "westArray"
    southArray = [61, 8]
    westArray = [68, 30]
    selected = southArray  # choose one

    selectedx, selectedy = selected[0], selected[1]

    # Build the carport as one mesh
    carport_mesh = design_carport_2Dgrid_tm(
        n_mods_x=selectedx,
        n_mods_y=selectedy,
        tilt_degs=10.0,
        front_col_top_z=4.5,
        purlin_splice_interval=2.0,
        module_clearance=0.15,
        column_width=0.25,
        column_thickness=0.25,
        purlin_width=0.12,
        purlin_thickness=0.12,
        add_sub_purlins=True,
        sub_purlin_offset=0.7
    )

    if carport_mesh is not None:
        carport_mesh.export("output/my_carport_trimesh.gltf")
        print("Exported the carport to my_carport_trimesh.gltf")
    else:
        print("No geometry was generated!")
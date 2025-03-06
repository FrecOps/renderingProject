import math
import numpy as np
import pyvista as pv
import argparse

# Define colors for the components
color_frame = 'darkgray'
color_base = 'lightgray'
color_marker = 'red'

def create_beam_pv(start, end, width, thickness):
    offset_factor = 1.0  # Adjust if further tweaking is needed.
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    d = direction / length

    # Choose a reference vector that is not parallel to d.
    ref = np.array([0, 0, 1])
    if abs(np.dot(d, ref)) > 0.99:
        ref = np.array([0, 1, 0])
    v = np.cross(d, ref)
    v /= np.linalg.norm(v)
    w = np.cross(d, v)
    w /= np.linalg.norm(w)

    a = width / 2.0
    b = (thickness / 2.0) * offset_factor

    # Extend the beam endpoints by half the thickness so edges overlap ("flash").
    start_offset = start - d * (thickness / 2.0)
    end_offset = end + d * (thickness / 2.0)

    corners_start = [
        start_offset + (a * v + b * w),
        start_offset + (-a * v + b * w),
        start_offset + (-a * v - b * w),
        start_offset + (a * v - b * w),
    ]
    corners_end = [
        end_offset + (a * v + b * w),
        end_offset + (-a * v + b * w),
        end_offset + (-a * v - b * w),
        end_offset + (a * v - b * w),
    ]
    vertices = np.array(corners_start + corners_end)
    faces_array = np.hstack([
        [4, 0, 1, 2, 3],
        [4, 4, 5, 6, 7],
        [4, 0, 1, 5, 4],
        [4, 1, 2, 6, 5],
        [4, 2, 3, 7, 6],
        [4, 3, 0, 4, 7]
    ])
    return pv.PolyData(vertices, faces_array)

def add_beam(plotter, start, end, beam_width, beam_thickness, color, label=None):
    beam = create_beam_pv(start, end, beam_width, beam_thickness)
    plotter.add_mesh(beam, color=color, show_edges=True)
    if label is not None:
        plotter.add_point_labels([end], [label], text_color=color_marker, point_size=10)

def create_tilted_frame_with_base(panel_width, panel_depth, height_front, tilt_angle,
                                  beam_width, beam_thickness,
                                  extension_length, extender_x, block_height,
                                  panel_extrude_thickness,
                                  module_count, single_module_width, module_spacing,
                                  plotter=None, offset=(0, 0, 0)):
    """
    Draws one unit (tilted frame with solar panel) into the provided plotter.
    The geometry is translated by the given offset.
    """
    # If no plotter is provided, create one.
    if plotter is None:
        plotter = pv.Plotter()

    # Convert offset to a numpy array for easy arithmetic.
    offset = np.array(offset, dtype=float)

    # Compute back height using the tilt angle (in degrees)
    height_back = height_front + panel_depth * math.tan(math.radians(tilt_angle))

    # Overall base plane (dimensions defined by the overall frame)
    base_center = (offset[0] + panel_width/2, offset[1] + panel_depth/2, offset[2])
    base = pv.Plane(center=base_center, direction=(0, 0, 1),
                    i_size=panel_width, j_size=panel_depth, i_resolution=10, j_resolution=10)
    plotter.add_mesh(base, color=color_base, opacity=0.5, show_edges=True)

    # Overall base frame corners and labels (for the entire panel)
    # Note: the base corners (of the panel face) start at offset.
    corners = [
        [offset[0], offset[1], offset[2]],
        [offset[0] + panel_width, offset[1], offset[2]],
        [offset[0] + panel_width, offset[1] + panel_depth, offset[2]],
        [offset[0], offset[1] + panel_depth, offset[2]]
    ]
    labels = ['A', 'B', 'C', 'D']
    for i in range(len(corners)):
        start_pt, end_pt = corners[i], corners[(i + 1) % len(corners)]
        add_beam(plotter, start_pt, end_pt, beam_width, beam_thickness, color_frame, labels[i])

    # Add extensions and blocks at each overall corner (as before)
    for corner, label in zip(corners, labels):
        # Determine if the corner is on the bottom or top edge based on y-coordinate.
        if abs(corner[1] - offset[1]) < 1e-6:
            ext_end = [corner[0], corner[1] - extension_length, 0]
        else:
            ext_end = [corner[0], corner[1] + extension_length, 0]
        add_beam(plotter, corner, ext_end, beam_width, beam_thickness, color_frame, f'{label}-Ext')
        # Determine if the corner is on the left or right edge based on x-coordinate.
        if abs(corner[0] - offset[0]) < 1e-6:
            ext_x_end = [ext_end[0] + extender_x, ext_end[1], 0]
        else:
            ext_x_end = [ext_end[0] - extender_x, ext_end[1], 0]
        add_beam(plotter, ext_end, ext_x_end, beam_width, beam_thickness, color_frame, f'{label}-ExtX')
        add_beam(plotter, ext_x_end, [ext_x_end[0], corner[1], 0], beam_width, beam_thickness, color_frame)

        # Block covering the extension plate
        block_length = abs(ext_x_end[0] - corner[0])
        block_width = abs(ext_end[1] - corner[1])
        block_center = [(corner[0] + ext_x_end[0]) / 2,
                        (corner[1] + ext_end[1]) / 2,
                        block_height / 2]
        block = pv.Cube(center=block_center, x_length=block_length, y_length=block_width, z_length=block_height)
        plotter.add_mesh(block, color='lightblue', show_edges=True)
        block_label = f'BX Ballast Block\n{block_length:.2f}m x {block_width:.2f}m x {block_height:.2f}m\n3000 PSI'
        plotter.add_point_labels([block_center], [block_label], text_color='black', point_size=10)

    # Vertical beams and overall top frame
    heights = [height_front, height_front, height_back, height_back]
    top_corners = [[corner[0], corner[1], h] for corner, h in zip(corners, heights)]
    for corner, h in zip(corners, heights):
        add_beam(plotter, corner, [corner[0], corner[1], h], beam_width, beam_thickness, color_frame)
    for i in range(len(top_corners)):
        start_pt, end_pt = top_corners[i], top_corners[(i + 1) % len(top_corners)]
        add_beam(plotter, start_pt, end_pt, beam_width, beam_thickness, color_frame)

    # --- Add individual module frames on the top face ---
    # Top face corners (from overall top frame)
    A_top = np.array(top_corners[0])
    B_top = np.array(top_corners[1])
    C_top = np.array(top_corners[2])
    D_top = np.array(top_corners[3])
    T = panel_width  # overall front edge length
    for i in range(module_count):
        frac_start = (i * single_module_width + i * module_spacing) / T
        frac_end = ((i * single_module_width + i * module_spacing) + single_module_width) / T
        front_left  = A_top + frac_start * (B_top - A_top)
        front_right = A_top + frac_end   * (B_top - A_top)
        back_left   = D_top + frac_start * (C_top - D_top)
        back_right  = D_top + frac_end   * (C_top - D_top)
        module_corners = [front_left.tolist(), front_right.tolist(), back_right.tolist(), back_left.tolist()]
        for j in range(len(module_corners)):
            add_beam(plotter, module_corners[j], module_corners[(j + 1) % 4],
                     beam_width, beam_thickness, 'silver')
       

    # Solar panel on the slanted top face
    panel_points = np.array(top_corners)
    panel_faces = np.hstack([[4, 0, 1, 2, 3]])
    panel_poly = pv.PolyData(panel_points, panel_faces) #
    A, B, C = panel_points[0], panel_points[1], panel_points[2]
    normal = np.cross(B - A, C - A)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    solar_panel = panel_poly.extrude(normal * panel_extrude_thickness)
    solar_panel.translate(normal * 0.01, inplace=True)
    plotter.add_mesh(solar_panel, color='grey', opacity=0.8, show_edges=True)
    top_cover_points = panel_points + normal * panel_extrude_thickness + normal * 0.01
    top_cover = pv.PolyData(top_cover_points, panel_faces)
    plotter.add_mesh(top_cover, color='#1D1D77', opacity=0.8, show_edges=True)

    # Do not call plotter.show() here so that multiple units can be added to the same plotter.
    return plotter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a tilted frame with a solar panel. Dimensions are in meters by default, but you can select inches with the --units option."
    )
    parser.add_argument('--units', type=str, default='m', choices=['m', 'in'],
                        help='Units for dimensions. Default is "m" (meters). Use "in" for inches.')
    parser.add_argument('--panel_width', type=float, default=2.46,
                        help='Width of a single module (default: 2.46 m ≈ 96.9 in)')
    parser.add_argument('--panel_depth', type=float, default=1.13,
                        help='Depth of a single module (default: 1.13 m ≈ 44.6 in)')
    parser.add_argument('--height_front', type=float, default=0.4,
                        help='Front frame height (default: 0.4 m)')
    parser.add_argument('--tilt_angle', type=float, default=10.0,
                        help='Tilt angle in degrees (default: 10)')
    parser.add_argument('--beam_width', type=float, default=0.1,
                        help='Beam width (default: 0.1 m)')
    parser.add_argument('--beam_thickness', type=float, default=0.1,
                        help='Beam thickness (default: 0.1 m)')
    parser.add_argument('--extension_length', type=float, default=0.25,
                        help='Extension length (default: 0.25 m)')
    parser.add_argument('--extender_x', type=float, default=0.35,
                        help='Extender x offset (default: 0.35 m)')
    parser.add_argument('--block_height', type=float, default=0.1,
                        help='Block height (default: 0.1 m)')
    parser.add_argument('--panel_extrude_thickness', type=float, default=0.035,
                        help='Solar panel extrusion thickness (default: 0.035 m ≈ 1.38 in)')
    parser.add_argument('--modules', type=int, default=1,
                        help='Number of modules in the frame (default: 2)')
    parser.add_argument('--module_spacing', type=float, default=0.1,
                        help='Spacing between each module (default: 0.1 m)')
    parser.add_argument('--moduleOrientation', default='Portrait', choices=['Landscape', 'Portrait'],
                        help='Solar Module Orientation (default: Landscape)')
    # New arguments for grid duplication
    parser.add_argument('--strings', type=int, default=2,
                        help='Number of rows (strings) of units (default: 2)')
    parser.add_argument('--columns', type=int, default=3,
                        help='Number of columns of units (default: 3)')
    args = parser.parse_args()

    # Determine conversion factor based on units.
    conv = 1.0 if args.units == 'm' else 0.0254

    module_count = args.modules
    module_spacing = args.module_spacing * conv
    single_module_width = args.panel_width * conv

    module_orientation = args.moduleOrientation

    panel_depth = args.panel_depth * conv
    height_front = args.height_front * conv
    # Overall width: total width of modules plus spacing between them.
    total_panel_width = module_count * single_module_width + (module_count - 1) * module_spacing

    if module_orientation == "Portrait":
        panel_depth = args.panel_width * conv
        single_module_width = args.panel_depth * conv
        total_panel_width = module_count * single_module_width + (module_count - 1) * module_spacing

    beam_width = args.beam_width * conv
    beam_thickness = args.beam_thickness * conv
    extension_length = args.extension_length * conv
    extender_x = args.extender_x * conv
    block_height = args.block_height * conv
    panel_extrude_thickness = args.panel_extrude_thickness * conv

    # Define grid dimensions.
    # The horizontal (east-west) unit size is total_panel_width.
    # The vertical (north-south) full footprint includes the base panel depth plus the extension on both bottom and top.
    unit_width = total_panel_width
    unit_depth = panel_depth + 2 * extension_length

    # Create a single plotter for all units.
    plotter = pv.Plotter()

    # Loop over rows ("strings") and columns ("units" in each row)
    for row in range(args.strings):
        for col in range(args.columns):
            # Calculate offset for each unit.
            # We add extension_length to the y offset so that the bottom-most extension does not go negative.
            offset = (col * unit_width, row * unit_depth + extension_length, 0)
            create_tilted_frame_with_base(total_panel_width, panel_depth, height_front, args.tilt_angle,
                                          beam_width, beam_thickness,
                                          extension_length, extender_x, block_height,
                                          panel_extrude_thickness,
                                          module_count, single_module_width, module_spacing,
                                          plotter=plotter, offset=offset)

    plotter.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

def create_module(center, width, height, tilt=0, rotation=0):
    """
    Create the 3D corner coordinates for a rectangular solar module.
    Corners order: bottom-left, bottom-right, top-right, top-left.
    The module is first defined in local coords, then:
      - rotate about Z
      - tilt about X
      - translate to center
    """
    corners = np.array([
        [-width / 2, -height / 2, 0],
        [ width / 2, -height / 2, 0],
        [ width / 2,  height / 2, 0],
        [-width / 2,  height / 2, 0]
    ])

    # 1) Rotation about Z-axis
    rot_z = np.array([
        [ np.cos(rotation), -np.sin(rotation), 0],
        [ np.sin(rotation),  np.cos(rotation), 0],
        [ 0,                0,                1]
    ])
    corners = corners @ rot_z.T

    # 2) Tilt about X-axis
    rot_x = np.array([
        [1,             0,              0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt),  np.cos(tilt)]
    ])
    corners = corners @ rot_x.T

    # 3) Translate to final center
    corners += np.array(center)
    return corners

def create_beam(start, end, beam_width, beam_thickness):
    """
    Create the vertices and faces for a rectangular beam (cuboid) spanning from start to end.
    Cross-section is (beam_width x beam_thickness), oriented perp. to the beam axis.
    """
    start = np.array(start)
    end = np.array(end)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Start and end points cannot be identical")
    d = direction / length

    # A reference vector not parallel to d
    ref = np.array([0, 0, 1])
    if abs(np.dot(d, ref)) > 0.99:
        ref = np.array([0, 1, 0])

    v = np.cross(d, ref)
    v /= np.linalg.norm(v)
    w = np.cross(d, v)
    w /= np.linalg.norm(w)

    a = beam_width / 2.0
    b = beam_thickness / 2.0
    offsets = [
        a*v + b*w,
        -a*v + b*w,
        -a*v - b*w,
        a*v - b*w
    ]

    vertices = []
    for offset in offsets:
        vertices.append(start + offset)
    for offset in offsets:
        vertices.append(end + offset)

    faces = [
        [0, 1, 2, 3],  # start face
        [4, 5, 6, 7],  # end face
        [0, 1, 5, 4],  # side face
        [1, 2, 6, 5],  # side face
        [2, 3, 7, 6],  # side face
        [3, 0, 4, 7]   # side face
    ]
    return vertices, faces

def degrees_to_radians(deg):
    return deg * math.pi / 180

def plot_solar_layout(ax, rows=1, cols=8, numStrings=2, spacing=0.05,
                      module_width=1.134, module_height=2.462,
                      tilt=0.1, rotation=0):
    """
    Plot modules in 'rows' horizontally, each row subdivided into 'numStrings' vertical strings.
    The top of string s lines up exactly with the bottom of string (s+1).
    """
    modules = []
    z_offset = (module_height / 2) * np.sin(tilt)  # ensures bottom edge is at z=0 for string 0
    string_gap = module_height * np.cos(tilt)      # shift in Y for each subsequent string

    for r in range(rows):
        y_row = r * (numStrings*module_height*np.cos(tilt) + spacing)
        for s in range(numStrings):
            y_s = y_row + s*string_gap
            z_s = z_offset + s*(module_height*np.sin(tilt))
            for c in range(cols):
                x = c*(module_width + spacing)
                mod_corners = create_module((x, y_s, z_s),
                                            module_width, module_height,
                                            tilt, rotation)
                modules.append(mod_corners)
                poly = Poly3DCollection([mod_corners], alpha=0.8, edgecolor='k')
                poly.set_facecolor('lightskyblue')
                ax.add_collection3d(poly)

    return modules

def plot_structures_with_shared_interface(ax,
                                          rows, cols, numStrings, spacing,
                                          module_width, module_height,
                                          tilt, rotation,
                                          rail_width=0.1, rail_thickness=0.05,
                                          post_width=0.08, post_thickness=0.08,
                                          rail_offset=0.03,
                                          footing_level=-0.3,
                                          rail_color='gray', post_color='dimgray',
                                          max_post_spacing=4.0):
    """
    For multiple strings in one row:
      - The front rail of string s is placed at the *shared interface* with string (s-1)'s rear rail
        => So the top edge of string s-1 is the same as the bottom edge of string s.
      - This ensures no "floating" top corners.
      - All rails sit below the modules by 'rail_offset' so they never protrude above.

    Steps for row r:
      1) We place a 'front rail' for string 0 (lowest string).
      2) For each string s from 1..(numStrings-1), we skip the “front rail” because it is the same rail
         as the previous string's "rear rail." We only place that previous rail once.
      3) For each string s, place the “rear rail” if s == (numStrings-1) or
         if you want to see separate rails for each string’s top edge.
      4) Each rail is fully supported down to the same footing_level.

    The function does:
      - For each row r, for each string s, compute front_edge_z, rear_edge_z, etc.
      - If s == 0, create a front rail. Then for every s, create a rear rail.
        The second string's 'front rail' is automatically the first string's 'rear rail,'
        so we do not double-draw or leave anything floating.
    """
    z_offset = (module_height / 2)*np.sin(tilt)
    string_gap = module_height * np.cos(tilt)

    for r in range(rows):
        base_y = r*(numStrings*module_height*np.cos(tilt) + spacing)

        for s in range(numStrings):
            # The Y center & Z center for string s
            y_s = base_y + s*string_gap
            z_center_s = z_offset + s*(module_height*np.sin(tilt))

            # Module "bottom" for string s => z_bottom_s
            z_bottom_s = z_center_s - (module_height/2)*np.sin(tilt)
            # Module "top" for string s => z_top_s
            z_top_s    = z_center_s + (module_height/2)*np.sin(tilt)

            # We'll place the rails rail_offset below each edge
            front_rail_z = z_bottom_s - rail_offset  # "front" or "low" edge
            rear_rail_z  = z_top_s    - rail_offset  # "rear" or "high" edge

            # The x-limits
            x_left = 0.0
            x_right = (cols - 1)*(module_width + spacing)

            # For string s => front rail y = (lowest edge in Y) for that string
            # Actually the module's "lowest Y edge" is y_s - (module_height/2)*cos(tilt).
            # But we only physically build that rail for s==0. Otherwise we share with s-1's top.
            front_y = y_s - (module_height/2)*np.cos(tilt)
            rear_y  = y_s + (module_height/2)*np.cos(tilt)

            # If s==0 => build front rail
            if s == 0:
                front_rail_start = np.array([x_left,  front_y, front_rail_z])
                front_rail_end   = np.array([x_right, front_y, front_rail_z])
                # Build that rail + posts
                build_rail(ax, front_rail_start, front_rail_end,
                           rail_width, rail_thickness,
                           post_width, post_thickness,
                           footing_level, max_post_spacing,
                           rail_color, post_color)

            # We ALWAYS build the "rear rail" for string s. This is the top edge of string s,
            # which also acts as front edge for string s+1, so it is shared.
            # That ensures no "floating" top corners for the final string.
            rear_rail_start = np.array([x_left,  rear_y, rear_rail_z])
            rear_rail_end   = np.array([x_right, rear_y, rear_rail_z])
            build_rail(ax, rear_rail_start, rear_rail_end,
                       rail_width, rail_thickness,
                       post_width, post_thickness,
                       footing_level, max_post_spacing,
                       rail_color, post_color)

def build_rail(ax, rail_start, rail_end,
               rail_width, rail_thickness,
               post_width, post_thickness,
               footing_level, max_post_spacing,
               rail_color, post_color):
    """
    Helper to build a single rail + posts at each end (+ optional mid-post for long spans).
    """
    # 1) The rail beam
    v_rail, f_rail = create_beam(rail_start, rail_end, rail_width, rail_thickness)
    rail_poly = Poly3DCollection([[v_rail[v] for v in face] for face in f_rail],
                                 facecolor=rail_color, alpha=0.9, edgecolor='k')
    ax.add_collection3d(rail_poly)

    # 2) End posts
    for end_pt in [rail_start, rail_end]:
        bottom = np.array([end_pt[0], end_pt[1], footing_level])
        v_post, f_post = create_beam(end_pt, bottom, post_width, post_thickness)
        post_poly = Poly3DCollection([[v_post[v] for v in face] for face in f_post],
                                     facecolor=post_color, alpha=0.9, edgecolor='k')
        ax.add_collection3d(post_poly)

    # 3) Mid-post if the rail is very long
    rail_len = np.linalg.norm(rail_end - rail_start)
    if rail_len > max_post_spacing:
        mid_pt = (rail_start + rail_end)/2.0
        bottom_mid = np.array([mid_pt[0], mid_pt[1], footing_level])
        v_mid, f_mid = create_beam(mid_pt, bottom_mid, post_width, post_thickness)
        mid_poly = Poly3DCollection([[v_mid[v] for v in face] for face in f_mid],
                                    facecolor=post_color, alpha=0.9, edgecolor='k')
        ax.add_collection3d(mid_poly)


#---------------- MAIN CODE --------------
if __name__ == "__main__":

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Example color
    flamingo_pink = '#FC8EAC'

    # Layout parameters
    rows        = 1
    cols        = 8
    numStrings  = 2
    spacing     = 0.05
    module_w    = 1.134
    module_h    = 2.462
    tilt_deg    = 10.0
    tilt        = degrees_to_radians(tilt_deg)

    # 1) Plot modules
    mods = plot_solar_layout(ax, rows, cols, numStrings, spacing,
                             module_w, module_h,
                             tilt=tilt, rotation=0)

    # 2) Plot shared interface structure
    plot_structures_with_shared_interface(
        ax, rows, cols, numStrings, spacing,
        module_w, module_h,
        tilt, rotation=0,
        rail_width=0.12, rail_thickness=0.08,
        post_width=0.08, post_thickness=0.08,
        rail_offset=0.03,
        footing_level=-0.3,
        rail_color=flamingo_pink,
        post_color=flamingo_pink,
        max_post_spacing=4.0
    )

    # 3) Set axis, limits
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # Some axis bounds
    z_max = (numStrings*module_h*math.sin(tilt)) + module_h
    ax.set_xlim(-1, cols*(module_w+spacing))
    ax.set_ylim(-1, (numStrings*module_h*math.cos(tilt))+spacing)
    ax.set_zlim(-0.5, z_max + 0.1)

    plt.title("Carport with Shared Rails: No Floating Second String")
    plt.tight_layout()
    plt.show()
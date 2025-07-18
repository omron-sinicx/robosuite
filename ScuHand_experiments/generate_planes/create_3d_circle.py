import numpy as np
import trimesh

def generate_cut_disc_3d(radius=1.0, height=0.05, segments=64):
    # Top circle
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    top_z = height / 2
    bottom_z = -height / 2

    # Top and bottom circle points
    top_circle = np.stack([x, y, np.full_like(x, top_z)], axis=1)
    bottom_circle = np.stack([x, y, np.full_like(x, bottom_z)], axis=1)

    # Duplicate first point to create the radial cut
    top_circle = np.vstack([top_circle, top_circle[0]])
    bottom_circle = np.vstack([bottom_circle, bottom_circle[0]])

    # Add center points
    center_top = np.array([[0, 0, top_z]])
    center_bottom = np.array([[0, 0, bottom_z]])

    vertices = np.vstack([center_top, center_bottom, top_circle, bottom_circle])

    center_top_idx = 0
    center_bottom_idx = 1
    top_start_idx = 2
    bottom_start_idx = 2 + len(top_circle)

    faces = []

    # Top faces (fan from center top)
    for i in range(len(top_circle) - 1):
        faces.append([center_top_idx, top_start_idx + i, top_start_idx + i + 1])

    # Bottom faces (fan from center bottom)
    for i in range(len(bottom_circle) - 1):
        faces.append([center_bottom_idx, bottom_start_idx + i + 1, bottom_start_idx + i])

    # Side walls
    for i in range(len(top_circle) - 1):
        t0 = top_start_idx + i
        t1 = top_start_idx + i + 1
        b0 = bottom_start_idx + i
        b1 = bottom_start_idx + i + 1
        faces.append([t0, b0, b1])
        faces.append([t0, b1, t1])

    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export('cut_disc_3d.obj')
    return mesh

# Run it
generate_cut_disc_3d(radius=0.05, height=0.001, segments=64)

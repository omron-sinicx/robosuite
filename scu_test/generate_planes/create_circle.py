import numpy as np
import trimesh

def generate_clipped_circle(radius=0.05, segments=64, clip_degrees=10):
    # Angle to clip in radians
    clip_angle_rad = np.deg2rad(clip_degrees)

    # How many segments to remove
    clip_segments = int(np.ceil(segments * clip_degrees / 360))

    # Generate angles, excluding clipped portion
    theta = np.linspace(clip_angle_rad, 2 * np.pi, segments - clip_segments + 1)

    # Outer ring points
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = 0.0001 + np.zeros_like(x)
    circle_pts = np.stack([x, y, z], axis=1)

    # Add center vertex
    center = np.array([[0.0, 0.0, 0.0]])
    vertices = np.vstack([center, circle_pts])

    # Build triangle faces
    faces = []
    for i in range(1, len(vertices) - 1):
        faces.append([0, i, i + 1])

    # Optionally, leave out the last triangle to exaggerate the gap
    # (it's already skipped by reducing `theta`)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export('clipped_circle.obj')
    return mesh

# Run it
generate_clipped_circle(radius=0.05, segments=64, clip_degrees=1)

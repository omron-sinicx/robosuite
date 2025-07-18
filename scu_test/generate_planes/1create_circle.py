import numpy as np
import trimesh

def generate_clipped_concentric_circle_mesh(radius=0.05, rings=5, segments=64, clip_degrees=10):
    vertices = []
    faces = []

    # Clip in radians
    clip_rad = np.deg2rad(clip_degrees)
    usable_angle = 2 * np.pi - clip_rad
    clipped_segments = int(segments * (360 - clip_degrees) / 360)

    # Add center vertex
    vertices.append([0.0, 0.0, 0.0])

    # Generate concentric ring vertices
    for r in range(1, rings + 1):
        curr_radius = radius * r / rings
        for s in range(clipped_segments + 1):  # +1 to close last segment
            angle = clip_rad + usable_angle * s / clipped_segments
            x = curr_radius * np.cos(angle)
            y = curr_radius * np.sin(angle)
            vertices.append([x, y, 0.0])

    # Build triangle faces
    for r in range(rings):
        ring_start = 1 + r * (clipped_segments + 1)
        next_ring_start = ring_start + (clipped_segments + 1)

        for s in range(clipped_segments):
            curr = ring_start + s
            next = ring_start + s + 1

            if r == 0:
                # Connect to center
                faces.append([0, curr, next])
            else:
                prev_ring = ring_start - (clipped_segments + 1)
                prev_curr = prev_ring + s
                prev_next = prev_ring + s + 1
                faces.append([prev_curr, curr, prev_next])
                faces.append([curr, next, prev_next])

    # Create mesh
    vertices = np.array(vertices)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export("clipped_concentric_circle.obj")

    print(f"✅ Exported clipped mesh with {len(vertices)} vertices and {len(faces)} faces.")

    return mesh

# Run it
# generate_clipped_concentric_circle_mesh(radius=0.05, rings=3, segments=16, clip_degrees=0.1)
generate_clipped_concentric_circle_mesh(radius=0.05, rings=1, segments=30, clip_degrees=0.1)

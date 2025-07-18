import numpy as np
import trimesh

def create_half_truncated_conical_ring(inner_bottom_radius=0.02,
                                       outer_bottom_radius=0.05,
                                       inner_top_radius=0.015,
                                       outer_top_radius=0.03,
                                       height=0.03,
                                       arc_angle=np.pi,  # Half ring
                                       num_segments=50,
                                       filename='half_funnel.obj'):
    """
    Create a half funnel shape: truncated conical ring cut vertically in half.
    """
    vertices = []
    faces = []

    angles = np.linspace(-arc_angle/2, arc_angle/2, num_segments)

    for i in range(num_segments-1):
        theta0 = angles[i]
        theta1 = angles[i+1]

        # Define 8 vertices for each quad strip (4 outer, 4 inner)

        # Bottom outer
        v0 = [outer_bottom_radius * np.cos(theta0),
              outer_bottom_radius * np.sin(theta0),
              0]
        v1 = [outer_bottom_radius * np.cos(theta1),
              outer_bottom_radius * np.sin(theta1),
              0]

        # Top outer
        v2 = [outer_top_radius * np.cos(theta1),
              outer_top_radius * np.sin(theta1),
              height]
        v3 = [outer_top_radius * np.cos(theta0),
              outer_top_radius * np.sin(theta0),
              height]

        # Bottom inner
        v4 = [inner_bottom_radius * np.cos(theta0),
              inner_bottom_radius * np.sin(theta0),
              0]
        v5 = [inner_bottom_radius * np.cos(theta1),
              inner_bottom_radius * np.sin(theta1),
              0]

        # Top inner
        v6 = [inner_top_radius * np.cos(theta1),
              inner_top_radius * np.sin(theta1),
              height]
        v7 = [inner_top_radius * np.cos(theta0),
              inner_top_radius * np.sin(theta0),
              height]

        idx_base = len(vertices)
        vertices.extend([v0, v1, v2, v3, v4, v5, v6, v7])

        # Define faces (outer wall)
        faces.append([idx_base, idx_base+1, idx_base+2])
        faces.append([idx_base, idx_base+2, idx_base+3])

        # Inner wall (reverse order for normal direction)
        faces.append([idx_base+4, idx_base+7, idx_base+6])
        faces.append([idx_base+4, idx_base+6, idx_base+5])

        # Top face
        faces.append([idx_base+3, idx_base+2, idx_base+6])
        faces.append([idx_base+3, idx_base+6, idx_base+7])

        # Bottom face
        faces.append([idx_base, idx_base+4, idx_base+5])
        faces.append([idx_base, idx_base+5, idx_base+1])

    # Create mesh and export
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(filename)
    print(f"Mesh saved as {filename}")

# Usage example
create_half_truncated_conical_ring(inner_bottom_radius=0.049,
                                       outer_bottom_radius=0.05,
                                       inner_top_radius=0.029,
                                       outer_top_radius=0.03,
                                       height=0.05,
                                       arc_angle=np.pi,  # Half ring
                                       num_segments=6,)

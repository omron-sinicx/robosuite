"""
Generator for 3D meshes from 2D polygons.
"""
from typing import Dict, List, Optional, Union
import numpy as np
import trimesh
from shapely.geometry import Polygon

from geometry_utils import (
    vertices_to_polygon,
    vertices_to_Path2D,
)


class MeshGenerator:
    def __init__(self, config: dict):
        """
        Initialize the mesh generator with configuration parameters.

        Args:
            config: Dictionary containing configuration parameters:
                - peg_height: Height of peg in mm
                - hole_height: Height of hole block in mm
                - hole_size: Width of hole block in mm
                - hole_margin: Extra margin around hole in mm
                - peg_center_height_offset: Z-offset for peg center
                - hole_center_height_offset: Z-offset for hole center
        """
        self._validate_config(config)
        self.peg_height = config['peg_height']
        self.hole_height = config['hole_height']
        self.hole_size = config['hole_size']
        self.hole_margin = config['hole_margin']
        self.peg_z_offset = config['peg_center_height_offset']
        self.hole_center_height_offset = config['hole_center_height_offset']

    @staticmethod
    def _validate_config(config: dict) -> None:
        """Validate configuration parameters."""
        required_params = [
            'peg_height', 'hole_height', 'hole_size',
            'hole_margin', 'peg_center_height_offset',
            'hole_center_height_offset'
        ]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required parameter: {param}")

    def create_3d_mesh(self, poly_dict: Dict[str, np.ndarray], scale: float = 0.001) -> List[trimesh.Trimesh]:
        """
        Convert 2D polygon to 3D mesh by extruding it to specified height.

        Args:
            poly_dict: Dictionary containing polygon data with keys:
                - vertices: np.ndarray of vertex coordinates
                - type: str, either 'peg' or 'hole'
            scale: Scaling factor for vertices in meters (default: 0.001 for mm to m)

        Returns:
            List[trimesh.Trimesh]: List of 3D mesh objects

        Raises:
            ValueError: If polygon type is invalid or polygon is too large for hole
        """
        vertices_2d = poly_dict['vertices']
        poly_type = poly_dict.get('type', 'base')

        if poly_type not in ['peg', 'hole', 'base']:
            raise ValueError(f"Invalid polygon type: {poly_type}")

        if poly_type == 'hole':
            return self._create_hole_mesh(vertices_2d, scale)
        else:
            return self._create_peg_mesh(vertices_2d, scale)

    def _create_hole_mesh(self, vertices_2d: np.ndarray, scale: float) -> List[trimesh.Trimesh]:
        """Create mesh for hole type polygon."""
        # Create fixed-size box for hole
        box_vertices = np.array([
            [-self.hole_size / 2, -self.hole_size / 2],  # min x, min y
            [self.hole_size / 2, -self.hole_size / 2],   # max x, min y
            [self.hole_size / 2, self.hole_size / 2],    # max x, max y
            [-self.hole_size / 2, self.hole_size / 2],   # min x, max y
        ])

        # Validate hole size
        vertices_max = np.max(np.abs(vertices_2d))
        if vertices_max > (self.hole_size - self.hole_margin):
            raise ValueError(
                f"Hole too large for specified hole_size. "
                f"Need hole_size {self.hole_size} >= {vertices_max + self.hole_margin}"
            )

        # Create polygon with hole
        polygon = Polygon(shell=box_vertices.tolist(), holes=[vertices_2d.tolist()])

        # Triangulate the polygon
        tri_vertices, tri_faces = trimesh.creation.triangulate_polygon(polygon, triangle_args='pq5')

        # Create meshes for each triangle
        meshes = []
        for face in tri_faces:
            # Get triangle vertices
            triangle_vertices = tri_vertices[face]
            # Create path and extrude
            path = vertices_to_Path2D(triangle_vertices)
            extrude_mesh = path.extrude(self.hole_height)

            # Position mesh
            vertices = extrude_mesh.vertices.copy()
            vertices[:, 2] -= self.hole_height / 2
            vertices[:, 2] += self.hole_center_height_offset

            # Scale to meters
            vertices *= scale

            # Create final mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=extrude_mesh.faces)

            # Fixes common mesh issues
            mesh.process(validate=True)

            # Check for Degenerate or Sliver Triangles
            bad_faces = mesh.faces_sparse.shape[0] < 3
            if bad_faces:
                raise ValueError("Degenerate faces found in mesh")

            # Check the Inertia Tensor for Instability
            inertia = mesh.moment_inertia
            eigenvalues = np.linalg.eigvals(inertia)
            # Check stability condition: A + B >= C
            eigenvalues.sort()
            if not (eigenvalues[0] + eigenvalues[1] >= eigenvalues[2]):
                raise ValueError("Inertia tensor is unstable")

            meshes.append(mesh)

        # Return polygon information for visualization outside this class
        mesh_data = {
            'meshes': meshes,
            'polygon': polygon,
            'tri_vertices': tri_vertices,
            'tri_faces': tri_faces
        }

        return mesh_data

    def _create_peg_mesh(self, vertices_2d: np.ndarray, scale: float) -> List[trimesh.Trimesh]:
        """Create mesh for peg type polygon."""
        # Create base polygon
        polygon = vertices_to_polygon(vertices_2d)

        # Triangulate the polygon
        tri_vertices, tri_faces = trimesh.creation.triangulate_polygon(polygon, triangle_args='pq5')

        # Create meshes for each triangle
        meshes = []
        for face in tri_faces:
            # Get triangle vertices
            triangle_vertices = tri_vertices[face]
            # Create path and extrude
            path = vertices_to_Path2D(triangle_vertices)
            extrude_mesh = path.extrude(self.peg_height)

            # Position mesh
            vertices = extrude_mesh.vertices.copy()
            vertices[:, 2] -= self.peg_height / 2
            vertices[:, 2] += self.peg_z_offset

            # Scale to meters
            vertices *= scale

            # Create final mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=extrude_mesh.faces)
            meshes.append(mesh)

        # Return polygon information for visualization outside this class
        mesh_data = {
            'meshes': meshes,
            'polygon': polygon,
            'tri_vertices': tri_vertices,
            'tri_faces': tri_faces
        }

        return mesh_data

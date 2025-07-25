"""
Generator for creating random polygons and peg-hole pairs.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from tqdm.auto import tqdm

from geometry_utils import (
    point_in_polygon,
    point_to_line_distance,
    is_convex,
    inset_outset_polygon
)

logger = logging.getLogger(__name__)


class PolygonGenerator:
    def __init__(self, config: dict):
        """
        Initialize the polygon generator.

        Args:
            config: Dictionary containing:
                be_convex (bool): Whether to generate only convex polygons
                radius_range (Tuple[float, float]): (min_radius, max_radius) in mm
                tolerance (float): Tolerance for peg/hole differences in mm
                max_attempts (int): Maximum attempts to generate a valid polygon
                min_angle (float): Minimum allowed angle in degrees
                colinear_angle_tolerance (float): Tolerance for colinear angles
                convex_tolerance (float): Tolerance for convexity checks
                use_indent (bool): Whether to use indentations
                indent_factor (float): Controls how deep indentations can be
                indent_prob (float): Probability of having indentations
        """
        self.be_convex = config['be_convex']
        self.radius_range = config['radius_range']
        self.tolerance = config['tolerance']
        self.max_attempts = config['max_attempts']
        self.min_angle = config['min_angle']
        self.colinear_angle_tolerance = config['colinear_angle_tolerance']
        self.convex_tolerance = config['convex_tolerance']
        self.use_indent = config['use_indent']

        if self.use_indent:
            self.indent_factor = config['indent_factor']
            self.indent_prob = config['indent_prob']
            assert 0 <= self.indent_factor <= 1, "Indentation factor must be in range [0, 1]"
            assert 0 <= self.indent_prob <= 1, "Indentation probability must be in range [0, 1]"
        else:
            self.indent_factor = 0
            self.indent_prob = -1

        assert self.tolerance > 0, "Tolerance must be positive"
        self._validate_radius_range(self.radius_range)

    @staticmethod
    def _validate_radius_range(radius_range: Tuple[float, float]):
        """Validate radius range parameters."""
        if not isinstance(radius_range, tuple) or len(radius_range) != 2:
            raise ValueError("radius_range must be a tuple of (min, max)")
        if radius_range[0] >= radius_range[1]:
            raise ValueError("min radius must be less than max radius")
        if radius_range[0] <= 0:
            raise ValueError("radius must be positive")

    def generate_angles(self, n: int) -> np.ndarray:
        """Generate n random angles that sum to 360 degrees."""
        random_vals = np.random.rand(n)
        return random_vals / random_vals.sum() * 360

    def check_consecutive_angles(self, angles: np.ndarray) -> bool:
        """
        Check if any three consecutive angles sum to approximately 180 degrees.
        Returns True if the polygon is valid (no 180-degree sums).
        """
        n = len(angles)
        for i in range(n):
            angle_sum = sum(angles[(i + j) % n] for j in range(3))
            if abs(angle_sum - 180) < self.colinear_angle_tolerance:
                return False
        return True

    def check_angle_constraints(self, angles: np.ndarray) -> bool:
        """Check if angles meet minimum angle and non-collinearity constraints."""
        if np.any(angles < self.min_angle) or np.any(angles > 180 - self.min_angle):
            return False
        return self.check_consecutive_angles(angles)

    def check_collision(self, peg: Dict[str, np.ndarray], hole: Dict[str, np.ndarray]) -> bool:
        """
        Check if peg can fit into hole without collision.
        Returns True if insertion is possible.
        """
        peg_vertices = peg['vertices']
        hole_vertices = hole['vertices']
        safety_margin = self.tolerance * 0.1

        for vertex in peg_vertices:
            inside = point_in_polygon(vertex, hole_vertices)
            if not inside:
                return False

            for i in range(len(hole_vertices)):
                segment_start = hole_vertices[i]
                segment_end = hole_vertices[(i + 1) % len(hole_vertices)]
                dist = point_to_line_distance(vertex, segment_start, segment_end)
                if dist < safety_margin:
                    return False

        return True

    def generate_polygon(self, n: int) -> Dict[str, np.ndarray]:
        """
        Generate a random polygon with n vertices.

        Args:
            n: Number of vertices

        Returns:
            Dict containing:
                vertices: np.ndarray of vertex coordinates

        Raises:
            ValueError: If n < 3
            RuntimeError: If failed to generate valid polygon
        """
        if n < 3:
            raise ValueError("Number of vertices must be at least 3")

        for attempt in range(self.max_attempts):
            # Generate angles
            angles = self.generate_angles(n)
            if not self.check_angle_constraints(angles):
                continue

            # Generate radii
            radii = np.random.uniform(self.radius_range[0], self.radius_range[1], n)

            # Add indentations if enabled
            if np.random.random() < self.indent_prob:
                num_indentations = np.random.randint(1, max(2, n // 3))
                indentation_indices = np.random.choice(n, num_indentations, replace=False)

                for idx in indentation_indices:
                    reduction = np.random.uniform(0.3, self.indent_factor)
                    radii[idx] *= (1 - reduction)

            # Calculate cumulative angles and convert to coordinates
            cum_angles = np.cumsum(angles)
            vertices = np.zeros((n, 2))
            for i, (angle, r) in enumerate(zip(cum_angles, radii)):
                rad = np.deg2rad(angle)
                vertices[i] = [r * np.cos(rad), r * np.sin(rad)]

            # Check convexity if required
            if self.be_convex:
                if is_convex(vertices, self.convex_tolerance):
                    return {'vertices': vertices}
            else:
                return {'vertices': vertices}

        raise RuntimeError(f"Failed to generate valid polygon after {self.max_attempts} attempts")

    def generate_peg_hole_pair(self, n: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate a matching peg and hole using proper 2D offsetting.

        Args:
            n: Number of vertices

        Returns:
            Tuple of (peg_data, hole_data) dictionaries

        Raises:
            RuntimeError: If failed to generate valid pair
        """
        base_polygon = self.generate_polygon(n)

        for attempt in range(self.max_attempts):
            try:
                peg_verts, hole_verts = inset_outset_polygon(
                    base_polygon['vertices'],
                    self.tolerance
                )

                peg_data = {
                    'type': 'peg',
                    'vertices': peg_verts,
                }
                hole_data = {
                    'type': 'hole',
                    'vertices': hole_verts,
                }

                if self.check_collision(peg_data, hole_data):
                    return peg_data, hole_data

            except (ValueError, RuntimeError) as e:
                logger.debug(f"Attempt {attempt} failed: {str(e)}")
                continue

        raise RuntimeError(f"Failed to generate valid peg-hole pair after {self.max_attempts} attempts")

    def generate_multiple_polygons(
        self,
        vertex_counts: List[int],
        shapes_per_count: int = 20
    ) -> Dict[int, List[Dict[str, np.ndarray]]]:
        """
        Generate multiple peg-hole pairs for each vertex count.

        Args:
            vertex_counts: List of vertex counts to generate
            shapes_per_count: Number of shapes to generate per vertex count

        Returns:
            Dictionary mapping vertex counts to lists of polygon data
        """
        progress_bar = tqdm(
            total=len(vertex_counts) * shapes_per_count,
            desc="Generating polygons"
        )
        result = {}

        for n in vertex_counts:
            logger.info(f"Generating {shapes_per_count} peg-hole pairs with {n} vertices")
            polygons = []

            for _ in range(shapes_per_count):
                peg, hole = self.generate_peg_hole_pair(n)
                polygons.extend([peg, hole])
                progress_bar.update(1)

            result[n] = polygons

        return result

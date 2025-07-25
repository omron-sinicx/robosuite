"""
Utility functions for geometric calculations and conversions.
"""
from typing import Tuple, Optional, Union
import numpy as np
from shapely.geometry import Polygon
import trimesh
from trimesh.path import Path2D
import logging

logger = logging.getLogger(__name__)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Check if a point lies inside a polygon using ray casting algorithm.

    Args:
        point: Point coordinates as [x, y]
        polygon: Array of polygon vertices as [[x1, y1], [x2, y2], ...]

    Returns:
        bool: True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1

    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y) and
            (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
             (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
            inside = not inside
        j = i

    return inside


def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """
    Calculate minimum distance from a point to a line segment.

    Args:
        point: Point coordinates as [x, y]
        line_start: Start point of line segment as [x, y]
        line_end: End point of line segment as [x, y]

    Returns:
        float: Minimum distance from point to line segment
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return np.linalg.norm(point_vec)

    # Project point onto line
    t = max(0, min(1, np.dot(point_vec, line_vec) / (line_len * line_len)))
    projection = line_start + t * line_vec

    return np.linalg.norm(point - projection)


def vertices_to_polygon(vertices: np.ndarray) -> Polygon:
    """
    Convert numpy array of vertices to Shapely polygon.

    Args:
        vertices: Array of polygon vertices as [[x1, y1], [x2, y2], ...]

    Returns:
        Polygon: Shapely polygon object
    """
    return Polygon(vertices)


def polygon_to_array(poly: Polygon) -> np.ndarray:
    """
    Convert Shapely polygon to numpy array of vertices.

    Args:
        poly: Shapely polygon object

    Returns:
        np.ndarray: Array of polygon vertices

    Raises:
        ValueError: If polygon is empty
    """
    if poly.is_empty:
        raise ValueError("Cannot convert empty polygon")
    exterior_coords = np.array(poly.exterior.coords)
    return exterior_coords[:-1]  # skip repeated last vertex


def vertices_to_Path2D(vertices: np.ndarray) -> Path2D:
    """
    Convert numpy array of vertices to trimesh Path2D object.

    Args:
        vertices: Array of polygon vertices as [[x1, y1], [x2, y2], ...]

    Returns:
        Path2D: Trimesh Path2D object
    """
    vertices_closed = np.vstack((vertices, vertices[0]))
    lines = np.column_stack((
        np.arange(len(vertices_closed) - 1),
        np.arange(1, len(vertices_closed))
    ))
    return trimesh.path.Path2D(
        entities=[trimesh.path.entities.Line(points=line) for line in lines],
        vertices=vertices_closed
    )


def polygon_to_Path2D(polygon: Polygon) -> Path2D:
    """
    Convert Shapely polygon to trimesh Path2D object.

    Args:
        polygon: Shapely polygon object

    Returns:
        Path2D: Trimesh Path2D object
    """
    vertices = np.array(polygon.exterior.coords)
    return vertices_to_Path2D(vertices)


def inset_outset_polygon(vertices: np.ndarray,
                         tolerance: float,
                         join_style: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an inset (peg) polygon and an outset (hole) polygon using proper 2D offset operations.

    Args:
        vertices: Array of polygon vertices
        tolerance: Offset distance
        join_style: Style of line joins (1=round, 2=miter, 3=bevel)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Inset and outset polygon vertices

    Raises:
        ValueError: If tolerance is too large for polygon size
    """
    base_poly = vertices_to_polygon(vertices)

    # Create inset polygon
    peg_poly = base_poly.buffer(-tolerance / 2, join_style=join_style)
    if peg_poly.is_empty:
        raise ValueError("Cannot create peg - tolerance too large for polygon size")

    # Create outset polygon
    hole_poly = base_poly.buffer(tolerance / 2, join_style=join_style)

    return polygon_to_array(peg_poly), polygon_to_array(hole_poly)


def is_convex(vertices: np.ndarray, convex_tolerance: float = 1e-10) -> bool:
    """
    Check if polygon is strictly convex.

    Args:
        vertices: Array of polygon vertices
        convex_tolerance: Tolerance for convexity check

    Returns:
        bool: True if polygon is convex, False otherwise
    """
    n = len(vertices)
    if n < 3:
        return False

    sign = 0
    for i in range(n):
        v1 = vertices[(i + 1) % n] - vertices[i]
        v2 = vertices[(i + 2) % n] - vertices[(i + 1) % n]
        cross_product = np.cross(v1, v2)

        if abs(cross_product) < convex_tolerance:
            return False  # Collinear points not allowed

        if sign == 0:
            sign = np.sign(cross_product)
        elif sign * cross_product <= 0:
            return False

    return True

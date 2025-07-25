import random
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from tqdm import tqdm

try:
    from robosuite.scripts.generate_polygon_mesh.geometry_utils import (
        point_in_polygon,
        point_to_line_distance,
        vertices_to_Path2D,
    )
except ImportError:
    from geometry_utils import (
        point_in_polygon,
        point_to_line_distance,
        vertices_to_Path2D,
    )

# Constants
DEFAULT_TRIANGLE_ARGS = "pq10"
DEFAULT_JOIN_STYLE = 2  # miter
MIN_ANGLE_DEGREES = 15
COLINEAR_TOLERANCE = 5
DEFAULT_MAX_ATTEMPTS = 10000


def create_figure_and_axes(figsize=(8, 8)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    return fig, ax


def visualize_polygon(
    polygon: Dict[str, np.ndarray],
    ax: Optional[plt.Axes] = None,
    color: str = "k",
    fill: bool = False,
    annotate: bool = True,
    show: bool = False,
) -> Optional[plt.Axes]:
    """
    Visualize a single polygon.

    Args:
        polygon: Dictionary containing polygon data with 'vertices' key
        ax: Optional matplotlib axes for plotting
        color: Color for the polygon (default to 'k' for black)
        fill: Whether to fill the polygon (default is False)
        annotate: Whether to annotate vertices with their indices
        show: Whether to display the plot

    Returns:
        matplotlib.axes.Axes if ax was provided, None otherwise
    """
    vertices = polygon["vertices"]
    if vertices is None or len(vertices) == 0:
        raise ValueError("Invalid polygon vertices")

    # Create new figure if no axes provided
    if ax is None:
        _, ax = create_figure_and_axes()

    # Plot vertices and edges
    ax.plot(
        np.append(vertices[:, 0], vertices[0, 0]),
        np.append(vertices[:, 1], vertices[0, 1]),
        (f"{color}-o" if annotate else f"{color}-"),
    )

    # Fill the internal region
    if fill:
        ax.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=1)

    if annotate:
        # Add vertex numbers
        for i, (x, y) in enumerate(vertices):
            ax.annotate(f"{i}", (x, y), xytext=(5, 5), textcoords="offset points")
    else:
        # Turn off axis
        ax.axis("off")

    ax.grid(True)
    ax.set_aspect("equal")
    return ax


def visualize_polygons(
    polys: list,
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        _, ax = create_figure_and_axes()
    for poly in polys:
        out_vs = {"vertices": np.array(poly.exterior.coords.xy).T}
        ax = visualize_polygon(out_vs, ax=ax, fill=False, color="b", show=False)

        if poly.interiors and len(poly.interiors) > 0:
            in_vs = {"vertices": np.array(poly.interiors[0].xy).T}
            ax = visualize_polygon(in_vs, ax=ax, fill=False, color="r", show=False)
    return ax


def visualize_triangulation(
    polygon: Polygon,
    tri_vertices: np.ndarray,
    tri_faces: np.ndarray,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Axes:
    """
    Visualize the triangulation of a polygon.

    Args:
        polygon: Original Shapely polygon
        tri_vertices: Triangulation vertices
        tri_faces: Triangulation faces
        show: Whether to display the plot
        save_path: Optional path to save the plot

    Returns:
        matplotlib.figure.Figure: The matplotlib figure
    """
    if ax is None:
        _, ax = create_figure_and_axes()

    # Plot exterior polygon
    x, y = polygon.exterior.xy
    plt.plot(x, y, "k-", label="Exterior boundary")

    # Plot holes if any
    for hole in polygon.interiors:
        x, y = hole.xy
        plt.plot(x, y, "r--", label="Interiors boundary")

    # Plot triangles
    for face in tri_faces:
        triangle = tri_vertices[face]
        x = np.append(triangle[:, 0], triangle[0, 0])
        y = np.append(triangle[:, 1], triangle[0, 1])
        plt.plot(x, y, "b--", alpha=0.5)
        plt.fill(x, y, alpha=0.5)

    plt.axis("equal")
    plt.grid(True)
    plt.title("Polygon Triangulation")
    plt.legend()

    return ax


def visualize_meshes(meshes):
    scene = trimesh.Scene()

    def get_color():
        a = 255
        rgb = np.random.randint(0, 255, size=3)
        return [*rgb, a]

    for i, mesh in enumerate(meshes):
        mesh.visual.face_colors = get_color()
        scene.add_geometry(mesh, node_name=f"mesh_{i}")
    return scene


def get_box_vectrices(size=120):
    box_vertices = np.array(
        [
            [-size / 2, -size / 2],  # min x, min y
            [size / 2, -size / 2],  # max x, min y
            [size / 2, size / 2],  # max x, max y
            [-size / 2, size / 2],  # min x, max y
        ]
    )
    return box_vertices


def generate_angles(n: int) -> np.ndarray:
    """Generate n random angles that sum to 360 degrees."""
    random_vals = np.random.rand(n)
    return random_vals / random_vals.sum() * 360


def check_angle_constraints(
    angles: np.ndarray,
    min_angle: int = MIN_ANGLE_DEGREES,
    colinear_angle_tolerance=COLINEAR_TOLERANCE,
) -> bool:
    """Check if angles meet minimum angle and non-collinearity constraints."""
    if np.any(angles < min_angle) or np.any(angles > 180 - min_angle):
        return False

    # non-collinearity constraints
    n = len(angles)
    for i in range(n):
        angle_sum = sum(angles[(i + j) % n] for j in range(3))
        if abs(angle_sum - 180) < colinear_angle_tolerance:
            return False

    return True


def check_collision(
    inner_vertices: np.ndarray,
    outer_vertices: np.ndarray,
    tolerance: float,
) -> bool:
    """
    Check if peg can fit into hole without collision.
    Returns True if insertion is possible.
    """

    for vertex in inner_vertices:
        inside = point_in_polygon(vertex, outer_vertices)
        if not inside:
            return False

        for i in range(len(outer_vertices)):
            segment_start = outer_vertices[i]
            segment_end = outer_vertices[(i + 1) % len(outer_vertices)]
            dist = point_to_line_distance(vertex, segment_start, segment_end)
            if dist < tolerance:
                return False

    return True


def inset_outset_polygon(
    vertices: np.ndarray,
    tolerance: float,
    join_style: int = DEFAULT_JOIN_STYLE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    join_style: Style of line joins (1=round, 2=miter, 3=bevel)
    """
    # Convert vertices to polygon
    base_poly = Polygon(vertices)

    # Create inset polygon
    peg_poly = base_poly.buffer(-tolerance / 2, join_style=join_style)
    if peg_poly.is_empty:
        raise ValueError("Cannot create peg - tolerance too large for polygon size")

    # Create outset polygon
    hole_poly = base_poly.buffer(tolerance / 2, join_style=join_style)
    if hole_poly.is_empty:
        raise ValueError("Cannot create hole - tolerance too large for polygon size")

    def polygon_to_array(poly):
        exterior_coords = np.array(poly.exterior.coords)
        return exterior_coords[:-1]  # skip repeated last vertex

    return polygon_to_array(peg_poly), polygon_to_array(hole_poly)


def generate_polygon(
    n: int,
    radius_range: List[float],
    outer_vertices: Optional[np.ndarray] = None,
    safe_margin: float = 2,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    min_angle: float = MIN_ANGLE_DEGREES,
) -> np.ndarray:
    if n < 3:
        raise ValueError("Number of vertices must be at least 3")

    for attempt in tqdm(
        range(max_attempts), leave=False, desc="Attempts Generating polygon"
    ):
        # Generate angles
        angles = generate_angles(n)
        if not check_angle_constraints(angles, min_angle=min_angle):
            continue

        # Generate radii
        radii = np.random.uniform(radius_range[0], radius_range[1], n)

        # Calculate cumulative angles and convert to coordinates
        cum_angles = np.cumsum(angles)
        vertices = np.zeros((n, 2))
        for i, (angle, r) in enumerate(zip(cum_angles, radii)):
            rad = np.deg2rad(angle)
            vertices[i] = [r * np.cos(rad), r * np.sin(rad)]

        # Check is not too close to the outer edges
        if outer_vertices is not None and not check_collision(
            vertices, outer_vertices, tolerance=safe_margin
        ):
            continue

        # Check convexity if required
        return vertices

    raise RuntimeError(f"Failed to generate valid polygon after {max_attempts} attempts")


def create_polygons_from_description(desc):
    """
    Create a list of Shapely Polygon objects from a list of polygon descriptions.

    Args:
        desc: List of dictionaries, each containing 'outer_vertices' and optional 'inner_vertices'

    Returns:
        List of Shapely Polygon objects
    """
    polys = []
    for poly_desc in desc:
        outer_vertices = poly_desc.get("outer_vertices")
        inner_vertices = poly_desc.get("inner_vertices", None)

        # Convert numpy arrays to lists for Shapely
        outer_coords = outer_vertices.tolist()

        if inner_vertices is not None:
            inner_coords = [inner_vertices.tolist()]
            polygon = Polygon(shell=outer_coords, holes=inner_coords)
        else:
            polygon = Polygon(shell=outer_coords)

        polys.append(polygon)
    return polys


def triangulate_polygons(polys, triangle_args=DEFAULT_TRIANGLE_ARGS):
    pairs = []
    for poly in polys:
        tri_vertices, tri_faces = trimesh.creation.triangulate_polygon(
            poly, triangle_args=triangle_args
        )
        pairs.append({"vertices": tri_vertices, "faces": tri_faces})

    return pairs


def tri_pairs_to_meshes(
    tri_pairs,
    height: float,
    z_offset: float,
    scale=0.001,
):
    meshes = []
    for pair in tri_pairs:
        tri_vertices, tri_faces = pair["vertices"], pair["faces"]

        for face in tri_faces:
            triangle_vertices = tri_vertices[face]
            # Create path and extrude
            path = vertices_to_Path2D(triangle_vertices)
            extrude_mesh = path.extrude(height)

            mesh = trimesh.Trimesh(
                vertices=extrude_mesh.vertices.copy(), faces=extrude_mesh.faces
            )

            # Position mesh
            mesh.apply_translation([0, 0, -height / 2])
            mesh.apply_translation([0, 0, z_offset])
            mesh.apply_scale(scale)

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

    return meshes


def create_peg_hole_description(
    n: int,
    radius_ranges: List[List[float]],
    inset_outset_tolerance: float = 1,
    safe_margin: float = 5,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
):
    """
    Creates a description of a nested peg and matching hole with multiple layers.

    The function generates a series of nested polygon layers that form the cross-sections of a


    Args:
        n (int): Number of vertices in each polygon layer.
        radius_ranges (List[List[float]]): List of radius ranges [min, max] for each nested layer.
            Each element defines a new nesting level, with the first being the outermost shape.
            Example: [[30, 40], [10, 20]] creates a peg with outer and inner layers.
        inset_outset_tolerance (float, optional): Clearance between matching peg and hole layers.
            Higher values create more space between peg and hole. Defaults to 1.
        safe_margin (float, optional): Minimum distance between a nested layer and its parent layer.
            Prevents layers from being too close to each other. Defaults to 5.
        max_attempts (int, optional): Maximum attempts to generate valid polygon layers.
            Defaults to 10000.

    Returns:
        Tuple[List[Dict], List[Dict]]: A tuple containing:
            - peg_desc: A list of dictionaries describing the peg's nested layers:
                - "outer_vertices": Numpy array of vertex coordinates for layer's outer boundary
                - "inner_vertices": Optional numpy array for inner holes within this layer
            - hole_desc: A list of dictionaries describing the hole's nested layers with same format

    Raises:
        ValueError: If collision is detected between layers or if a layer cannot be
            created with the given parameters
        RuntimeError: If valid polygons cannot be generated after max_attempts
    """

    def create_description(layers):
        # pad number of layers to even to prevent zip dropping the last layer
        if len(layers) % 2 == 1:
            layers.append(None)
        desc = []
        for out_vs, in_vs in zip(layers[::2], layers[1::2]):
            if in_vs is None:
                desc.append({"outer_vertices": out_vs})
            else:
                desc.append({"outer_vertices": out_vs, "inner_vertices": in_vs})
        return desc

    def validate_layers(layers, tolerance):
        # Check for collisions between layers
        for i in range(len(layers) - 1):
            out_vs = layers[i]
            in_vs = layers[i + 1]
            if not check_collision(
                inner_vertices=in_vs, outer_vertices=out_vs, tolerance=tolerance
            ):
                return False
        return True

    sqrt_max_attempts = int(max_attempts**0.5)
    for i_attempt in range(sqrt_max_attempts):
        try:
            peg_layers = []
            hole_layers = [get_box_vectrices()]

            vertices = generate_polygon(
                n,
                radius_ranges[0],
                max_attempts=sqrt_max_attempts,
                min_angle=args.min_angle,
            )
            inset_vertices, outset_vertices = inset_outset_polygon(
                vertices, tolerance=inset_outset_tolerance
            )
            peg_layers.append(inset_vertices)
            hole_layers.append(outset_vertices)
            is_peg_inset = False

            for radius_range in radius_ranges[1:]:
                vertices = generate_polygon(
                    3,
                    radius_range,
                    outer_vertices=inset_vertices,
                    safe_margin=safe_margin,
                    max_attempts=sqrt_max_attempts,
                    min_angle=args.min_angle,
                )
                inset_vertices, outset_vertices = inset_outset_polygon(
                    vertices, tolerance=inset_outset_tolerance
                )

                if is_peg_inset:
                    peg_layers.append(inset_vertices)
                    hole_layers.append(outset_vertices)
                else:
                    peg_layers.append(outset_vertices)
                    hole_layers.append(inset_vertices)
                is_peg_inset = not is_peg_inset

            if not validate_layers(peg_layers, inset_outset_tolerance):
                raise RuntimeError("Collision detected between peg layers")
            if not validate_layers(hole_layers, inset_outset_tolerance):
                raise RuntimeError("Collision detected between hole layers")

            peg_desc = create_description(peg_layers)
            hole_desc = create_description(hole_layers)
            return peg_desc, hole_desc
        except RuntimeError:
            # print(f"Failed to generate valid polygon {e}, Try {i_attempt}-th attempt")
            continue

    raise ValueError(f"Failed to generate valid polygon after {max_attempts} attempts")


def ensure_dir(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        print(f"Output directory {output_dir} already exists. Overwriting.")
        shutil.rmtree(output_dir)
    triangulation_dir = output_dir / "triangulation"
    meshes_dir = output_dir / "meshes"
    triangulation_dir.mkdir(parents=True, exist_ok=True)
    (meshes_dir / "peg").mkdir(parents=True, exist_ok=True)
    (meshes_dir / "hole").mkdir(parents=True, exist_ok=True)
    return triangulation_dir, meshes_dir


def export_meshes(
    meshes: List[trimesh.Trimesh],
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for idx, mesh in enumerate(meshes):
        filename = output_dir / f"part_{idx}.stl"
        mesh.export(str(filename))
        saved_paths.append(filename)
    return saved_paths


def export_point_cloud(
    meshes: List[trimesh.Trimesh],
    output_dir: Path,
    output_name: str = "pcd.npy",
    n_pts: int = 1024,
) -> Path:
    merged_mesh = trimesh.util.concatenate(meshes)
    n_vertices = len(merged_mesh.vertices)
    pcd = np.concatenate(
        [merged_mesh.vertices, merged_mesh.sample(max(0, n_pts - n_vertices))],
        axis=0,
    )
    output_path = output_dir / output_name
    np.save(str(output_path), pcd)
    return output_path


def main(args):
    # Setup random seed and directories
    random.seed(args.seed)
    np.random.seed(args.seed)
    tri_dir, mesh_dir = ensure_dir(args.output_dir)

    for n_vert in tqdm(range(args.min_vertices, args.max_vertices + 1)):
        for c in tqdm(range(args.shapes_per_vertex_count), leave=False):
            peg_desc, hole_desc = create_peg_hole_description(
                n=n_vert,
                radius_ranges=args.radius_ranges,
                inset_outset_tolerance=args.tolerance,
                safe_margin=args.safe_margin,
                max_attempts=args.max_attempts,
            )

            # Create polygons
            peg_polys = create_polygons_from_description(peg_desc)
            hole_polys = create_polygons_from_description(hole_desc)

            # Triangulate polygons
            peg_tri_pairs = triangulate_polygons(peg_polys)
            hole_tri_pairs = triangulate_polygons(hole_polys)

            # Create meshes
            peg_meshes = tri_pairs_to_meshes(
                peg_tri_pairs, height=args.peg_height, z_offset=args.peg_z_offset
            )
            hole_meshes = tri_pairs_to_meshes(
                hole_tri_pairs, height=args.hole_height, z_offset=args.hole_z_offset
            )

            # Save meshes
            export_meshes(peg_meshes, mesh_dir / "peg" / f"poly{n_vert}_{c}")
            export_meshes(hole_meshes, mesh_dir / "hole" / f"poly{n_vert}_{c}")

            # Save point clouds
            if args.point_cloud:
                export_point_cloud(
                    peg_meshes,
                    mesh_dir / "peg" / f"poly{n_vert}_{c}",
                    n_pts=args.n_point_cloud_points,
                )

            # print("Visualizing triangulation")
            _, ax = create_figure_and_axes(figsize=(12, 12))
            for poly, tri_pair in zip(
                [*peg_polys, *hole_polys], [*peg_tri_pairs, *hole_tri_pairs]
            ):
                ax = visualize_triangulation(
                    poly,
                    tri_pair["vertices"],
                    tri_pair["faces"],
                    ax=ax,
                    show=False,
                )
            plt.savefig(tri_dir / f"poly{n_vert}_{c}.png")
            plt.close()

            # print("Visualizing polygons")
            # _, ax = create_figure_and_axes()
            # ax = visualize_polygons(peg_polys, ax=ax)
            # ax = visualize_polygons(hole_polys, ax=ax)
            # plt.savefig(tri_dir / f"poly{n_vert}_{c}_line.png")
            # plt.close()

            # print("Visualizing meshes")
            # scene = visualize_meshes([*peg_meshes, *hole_meshes])
            # scene.show()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated files (default: output)",
    )
    parser.add_argument(
        "-rs",
        "--radius-ranges",
        type=float,
        nargs="+",
        default=[[10.0, 20.0]],
        help="List of radius ranges [min, max] for each nested layer in descening order (default: [[10.0, 20.0]])",
    )
    parser.add_argument(
        "-minv",
        "--min-vertices",
        type=int,
        default=3,
        help="Minimum number of vertices (default: 3)",
    )
    parser.add_argument(
        "-maxv",
        "--max-vertices",
        type=int,
        default=7,
        help="Maximum number of vertices (default: 7)",
    )
    parser.add_argument(
        "-c",
        "--shapes-per-vertex-count",
        type=int,
        default=20,
        help="Number of shapes to generate per vertex count (default: 20)",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=1.0,
        help="Tolerance between peg and hole in mm (default: 1.0)",
    )
    parser.add_argument(
        "-m",
        "--safe-margin",
        type=float,
        default=5.0,
        help="Minimum distance between nested layers in mm (default: 5.0)",
    )
    parser.add_argument(
        "-pcd",
        "--point-cloud",
        action="store_true",
        help="Generate point cloud data instead of meshes",
    )
    parser.add_argument(
        "-n-pts",
        "--n-point-cloud-points",
        type=int,
        default=1024,
        help="Number of points in point cloud (default: 1024)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1,
        help="Random seed",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10000,
        help="Maximum attempts to generate valid polygon layers (default: 10000)",
    )

    # Generally fixed values
    parser.add_argument(
        "--min-angle",
        type=float,
        default=MIN_ANGLE_DEGREES,
        help=f"Minimum angle between edges in degrees (default: {MIN_ANGLE_DEGREES})",
    )
    parser.add_argument(
        "--peg-height",
        type=float,
        default=75.0,
        help="Height of peg in mm (default: 75.0)",
    )
    parser.add_argument(
        "--hole-height",
        type=float,
        default=140.0,
        help="Height of hole block in mm (default: 140.0)",
    )
    parser.add_argument(
        "--peg-z-offset",
        type=float,
        default=37.5,
    )
    parser.add_argument(
        "--hole-z-offset",
        type=float,
        default=-70.0,
    )
    args = parser.parse_args()

    # Convert radius ranges to list of lists
    if len(args.radius_ranges) % 2 != 0:
        raise ValueError("Radius ranges must be provided in pairs.")
    args.radius_ranges = [
        [args.radius_ranges[i], args.radius_ranges[i + 1]]
        for i in range(0, len(args.radius_ranges), 2)
    ]
    print(f"Radius ranges: {args.radius_ranges}")

    main(args)

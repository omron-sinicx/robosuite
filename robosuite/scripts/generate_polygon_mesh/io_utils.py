"""
File I/O functions for handling polygon data and meshes.
"""
from pathlib import Path
import json
from typing import Dict, List, Union
import numpy as np
import trimesh
from tqdm.auto import tqdm
import logging

from mesh_generator import MeshGenerator
from visualizer import visualize_mesh_triangulation

logger = logging.getLogger(__name__)


def validate_polygon_data(polygons: Dict[int, List[Dict[str, np.ndarray]]]) -> None:
    """
    Validate polygon data structure.

    Args:
        polygons: Dictionary of polygon data

    Raises:
        ValueError: If data structure is invalid
    """
    if not isinstance(polygons, dict):
        raise ValueError("Polygons must be a dictionary")

    for vertex_count, poly_list in polygons.items():
        if not isinstance(vertex_count, int) or vertex_count < 3:
            raise ValueError(f"Invalid vertex count: {vertex_count}")
        if not isinstance(poly_list, list):
            raise ValueError(f"Value for vertex count {vertex_count} must be a list")

        for poly in poly_list:
            if not isinstance(poly, dict):
                raise ValueError("Each polygon must be a dictionary")
            if 'vertices' not in poly:
                raise ValueError("Each polygon must have 'vertices' key")
            if not isinstance(poly['vertices'], np.ndarray):
                raise ValueError("Vertices must be numpy array")


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        directory: Directory path

    Returns:
        Path: Path object of the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_polygons(polygons: Dict[int, List[Dict[str, np.ndarray]]],
                  output_dir: Union[str, Path],
                  indent: int = 2) -> None:
    """
    Save polygon data to JSON files.

    Args:
        polygons: Dictionary of polygon data
        output_dir: Output directory
        indent: JSON indentation level

    Raises:
        ValueError: If polygon data is invalid
    """
    validate_polygon_data(polygons)
    output_path = ensure_directory(output_dir)

    for n, poly_list in tqdm(polygons.items(), desc="Saving polygons"):
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        for poly in poly_list:
            serializable_poly = {
                **{k: v.tolist() for k, v in poly.items() if isinstance(v, np.ndarray)},
                **{k: v for k, v in poly.items() if not isinstance(v, np.ndarray)}
            }
            serializable_data.append(serializable_poly)

        filename = output_path / f"polygons_{n}_vertices.json"
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=indent)

        logger.info(f"Saved polygons with {n} vertices to {filename}")


def load_polygons(file_path: Union[str, Path]) -> List[Dict[str, np.ndarray]]:
    """
    Load polygon data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        List[Dict[str, np.ndarray]]: List of polygon dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Convert lists back to numpy arrays
    polygons = []
    for poly in data:
        converted_poly = {
            **{k: np.array(v) for k, v in poly.items() if isinstance(v, list)},
            **{k: v for k, v in poly.items() if not isinstance(v, list)}
        }
        polygons.append(converted_poly)

    return polygons


def export_mesh_stl(meshes: List[trimesh.Trimesh], output_dir: Union[str, Path]) -> List[str]:
    """
    Export meshes to STL files.

    Args:
        meshes: List of trimesh objects
        output_dir: Output directory

    Returns:
        List[str]: List of saved file paths
    """
    output_path = ensure_directory(output_dir)
    saved_paths = []

    for idx, mesh in enumerate(meshes):
        filename = output_path / f"part_{idx}.stl"
        mesh.export(str(filename))
        logger.info(f"Saved mesh to {filename}")
        saved_paths.append(str(filename))

    return saved_paths


def export_point_cloud(meshes: List[trimesh.Trimesh],
                       output_path: Union[str, Path],
                       n_pcd_points: int = 1024) -> str:
    """
    Create and export point cloud from meshes.

    Args:
        meshes: List of trimesh objects
        output_path: Output file path
        n_pcd_points: Number of points to sample

    Returns:
        str: Path to saved point cloud file
    """
    merged_mesh = trimesh.util.concatenate(meshes)
    n_vertices = len(merged_mesh.vertices)
    pcd = np.concatenate([
        merged_mesh.vertices,
        merged_mesh.sample(max(0, n_pcd_points - n_vertices))
    ], axis=0)

    # trimesh.points.plot_points(pcd) # visualize
    np.save(str(output_path), pcd)
    logger.info(f"Saved point cloud to {output_path}")
    return str(output_path)


def save_3d_meshes(mesh_generator: MeshGenerator,
                   polygons: Dict[int, List[Dict[str, np.ndarray]]],
                   output_dir: Union[str, Path],
                   save_pcd: bool = True,
                   n_pcd_points: int = 1024,
                   ) -> None:
    """
    Save 3D meshes to STL files.

    Args:
        mesh_generator: MeshGenerator object
        polygons: Dictionary of polygon data
        output_dir: Output directory
        n_pcd_points: Number of points for point cloud sampling
    """
    try:
        validate_polygon_data(polygons)
    except ValueError as e:
        raise IOError("Invalid polygon data") from e

    output_path = ensure_directory(output_dir)
    mesh_path = output_path / "meshes"
    ensure_directory(mesh_path)
    viz_path = output_path / "triangulation"
    ensure_directory(viz_path)

    step_cnt = sum([len(poly_list) for poly_list in polygons.values()])
    pbar = tqdm(total=step_cnt, desc="Saving 3D meshes")
    for n, poly_list in polygons.items():
        for i, poly in enumerate(poly_list):
            try:
                meshes = []
                # Generate mesh from polygon
                mesh_data = mesh_generator.create_3d_mesh(poly)
                mesh_type = poly['type']

                # Output directory
                dirname = mesh_path / f"{mesh_type}" / f"poly{n}_{i//2}"
                ensure_directory(dirname)
                plot_dir = viz_path / f"{mesh_type}"
                ensure_directory(plot_dir)

                # Generate visualization
                plot_path = plot_dir / f"poly{n}_{i//2}.png"
                visualize_mesh_triangulation(
                    mesh_data,
                    show=False,
                    save_path=plot_path
                )
                logger.info(f"Saved visualization to {plot_path}")

                # Export mesh STL files
                meshes = mesh_data['meshes']
                export_mesh_stl(meshes, dirname)

                # Export point cloud files
                if save_pcd and mesh_type == 'peg':
                    pcd_filename = dirname / f"pcd.npy"
                    export_point_cloud(meshes, pcd_filename, n_pcd_points)

            except Exception as e:
                logger.error(f"Failed to process polygon {i} with {n} vertices: {e}")

            pbar.update(1)

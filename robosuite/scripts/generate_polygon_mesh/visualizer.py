"""
Functions for visualizing polygons and meshes.
"""
from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from tqdm.auto import tqdm


def visualize_polygon(
    polygon: Dict[str, np.ndarray],
    ax: Optional[plt.Axes] = None,
    color: str = 'k',
    fill: bool = False,
    annotate: bool = True,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None
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
        save_path: Optional path to save the plot

    Returns:
        matplotlib.axes.Axes if ax was provided, None otherwise
    """
    vertices = polygon['vertices']
    poly_type = polygon.get('type', 'base')

    # Create new figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        standalone = True
    else:
        standalone = False

    # Plot vertices and edges
    ax.plot(
        np.append(vertices[:, 0], vertices[0, 0]),
        np.append(vertices[:, 1], vertices[0, 1]),
        (f'{color}-o' if annotate else f'{color}-'),
        label=poly_type.capitalize()
    )

    # Fill the internal region
    if fill:
        ax.fill(
            vertices[:, 0],
            vertices[:, 1],
            color=color,
            alpha=1
        )

    if annotate:
        # Add vertex numbers
        for i, (x, y) in enumerate(vertices):
            ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    else:
        # Turn off axis
        ax.axis('off')

    ax.grid(True)
    ax.set_aspect('equal')

    if standalone:
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
    return ax


def visualize_pair(
    peg: Dict[str, np.ndarray],
    hole: Dict[str, np.ndarray],
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Visualize a peg-hole pair on the same plot.

    Args:
        peg: Dictionary containing peg polygon data
        hole: Dictionary containing hole polygon data
        show: Whether to display the plot
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot both peg and hole
    visualize_polygon(peg, color='b', ax=ax, show=False)
    visualize_polygon(hole, color='g', ax=ax, show=False)

    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend()
    plt.title(f'Peg-Hole Pair with {len(peg["vertices"])} vertices')

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def visualize_triangulation(
    polygon: Polygon,
    tri_vertices: np.ndarray,
    tri_faces: np.ndarray,
    show: bool = False,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
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
    fig = plt.figure(figsize=(10, 10))

    # Plot original polygon
    x, y = polygon.exterior.xy
    plt.plot(x, y, 'k-', label='Original polygon')

    # Plot holes if any
    for hole in polygon.interiors:
        x, y = hole.xy
        plt.plot(x, y, 'r--', label='Hole boundary')

    # Plot triangles
    for face in tri_faces:
        triangle = tri_vertices[face]
        x = np.append(triangle[:, 0], triangle[0, 0])
        y = np.append(triangle[:, 1], triangle[0, 1])
        plt.plot(x, y, 'b--', alpha=0.5)
        plt.fill(x, y, alpha=0.1)

    plt.axis('equal')
    plt.grid(True)
    plt.title('Polygon Triangulation')
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def visualize_mesh_triangulation(
    mesh_data: Dict,
    show: bool = False,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Visualize the triangulation from mesh data.

    Args:
        mesh_data: Dictionary containing polygon, tri_vertices, and tri_faces
        show: Whether to display the plot
        save_path: Optional path to save the plot

    Returns:
        matplotlib.figure.Figure: The matplotlib figure
    """
    return visualize_triangulation(
        mesh_data['polygon'],
        mesh_data['tri_vertices'],
        mesh_data['tri_faces'],
        show=show,
        save_path=save_path
    )


def visualize_multiple_pairs(
    polygons: Dict[int, List[Dict[str, np.ndarray]]],
    output_dir: Union[str, Path],
    show: bool = False
) -> None:
    """
    Visualize multiple peg-hole pairs and save to files.

    Args:
        polygons: Dictionary mapping vertex counts to lists of polygon data
        output_dir: Directory to save visualization files
        show: Whether to display the plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for n, poly_list in tqdm(polygons.items(), desc="Saving visualization pairs"):
        for i in range(0, len(poly_list), 2):  # Step by 2 to get pairs
            peg, hole = poly_list[i], poly_list[i + 1]
            save_path = output_path / f"poly{n}_pair_{i//2}.png"
            visualize_pair(
                peg,
                hole,
                show=show,
                save_path=save_path
            )


def visualize_clean_pegs(
    polygons: Dict[int, List[Dict[str, np.ndarray]]],
    output_dir: Union[str, Path],
    show: bool = False
) -> None:
    """
    Visualize peg without any grid, title, etc.

    Args:
        polygons: Dictionary mapping vertex counts to lists of polygon data
        output_dir: Directory to save visualization files
        show: Whether to display the plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for n, poly_list in tqdm(polygons.items(), desc="Saving visualization pegs"):
        for i in range(0, len(poly_list), 2):  # Step by 2 to get pairs
            peg = poly_list[i]
            save_path = output_path / f"poly{n}_{i//2}.png"
            fig, ax = plt.subplots(figsize=(2, 2))

            ax = visualize_polygon(
                peg,
                ax=ax,
                fill=True,
                annotate=False,
                show=False,
                save_path=save_path
            )
            plt.tight_layout(pad=0)
            plt.savefig(save_path)
            if show:
                plt.show()
            plt.close()

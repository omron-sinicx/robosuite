"""
Main script for generating polygon peg-hole pairs.
"""
import logging
import argparse

import numpy as np
from pathlib import Path

from polygon_generator import PolygonGenerator
from mesh_generator import MeshGenerator
from io_utils import save_polygons, save_3d_meshes
from visualizer import visualize_multiple_pairs, visualize_clean_pegs

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate random polygon peg-hole pairs for robotic manipulation.'
    )

    # Generation parameters
    parser.add_argument('-o', '--output-dir', type=str, default='output',
                        help='Output directory for generated files (default: output)')
    parser.add_argument('-minv', '--min-vertices', type=int, default=4,
                        help='Minimum number of vertices (default: 4)')
    parser.add_argument('-maxv', '--max-vertices', type=int, default=10,
                        help='Maximum number of vertices (default: 10)')
    parser.add_argument('-c', '--shapes-per-count', type=int, default=20,
                        help='Number of shapes to generate per vertex count (default: 20)')
    parser.add_argument('-img', '--image', action='store_true', help='Generate 2D images of polygons')
    parser.add_argument('-pcd', '--point-cloud', action='store_true',
                        help='Generate point cloud data instead of meshes')
    parser.add_argument('-n-pts', '--n-point-cloud-points', type=int, default=1024)

    # Geometry parameters
    parser.add_argument('--be-convex', action='store_true',
                        help='Generate only convex polygons')
    parser.add_argument('--min-radius', type=float, default=10.0,
                        help='Minimum radius of polygons in mm (default: 10.0)')
    parser.add_argument('--max-radius', type=float, default=20.0,
                        help='Maximum radius of polygons in mm (default: 20.0)')
    parser.add_argument('--peg-height', type=float, default=75.0,
                        help='Height of peg in mm (default: 75.0)')
    parser.add_argument('--hole-height', type=float, default=140.0,
                        help='Height of hole block in mm (default: 140.0)')
    parser.add_argument('--hole-size', type=float, default=120.0,
                        help='Width of the hole block in mm (default: 120.0)')
    parser.add_argument('--hole-margin', type=float, default=4.0,
                        help='Margin around hole in mm (default: 4.0)')
    parser.add_argument('--tolerance', type=float, default=1.0,
                        help='Tolerance between peg and hole in mm (default: 1.0)')
    parser.add_argument('--min-angle', type=float, default=15.0,
                        help='Minimum allowed angle in degrees (default: 15.0)')
    parser.add_argument('--use-indent', action='store_true',
                        help='Generate polygons with indentations')
    parser.add_argument('--indent-factor', type=float, default=0.6,
                        help='Factor controlling depth of indentations in [0-1] (default: 0.6)')
    parser.add_argument('--indent-prob', type=float, default=1.0,
                        help='Probability of generating polygons with indentations [0-1] (default: 1.0)')

    # Output parameters
    parser.add_argument('--max-attempts', type=int, default=10000,
                        help='Maximum attempts to generate each polygon (default: 10000)')
    parser.add_argument('-viz', '--visualization', action='store_true',
                        help='Enable visualization generation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    return parser.parse_args()


def create_config(args):
    """Create configuration dictionary from parsed arguments."""
    return {
        # Polygon generation config
        'be_convex': args.be_convex,
        'radius_range': (args.min_radius, args.max_radius),
        'tolerance': args.tolerance,
        'max_attempts': args.max_attempts,
        'min_angle': args.min_angle,
        'use_indent': args.use_indent,
        'indent_factor': args.indent_factor,
        'indent_prob': args.indent_prob,
        'colinear_angle_tolerance': 1.0,
        'convex_tolerance': 1e-10,

        # Mesh generation config
        'peg_height': args.peg_height,
        'hole_height': args.hole_height,
        'hole_size': args.hole_size,
        'hole_margin': args.hole_margin,
        'peg_center_height_offset': 37.5,
        'hole_center_height_offset': -70.0,
        'debug_visualization': args.visualization
    }


def main():
    """Main function to generate polygon peg-hole pairs."""
    # Parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Create configuration
    config = create_config(args)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    polygon_dir = output_dir / 'polygon_data'
    polygon_dir.mkdir(exist_ok=True)

    try:
        # Initialize generators
        polygon_generator = PolygonGenerator(config)
        mesh_generator = MeshGenerator(config)

        # Generate polygons for each vertex count
        vertex_counts = list(range(args.min_vertices, args.max_vertices + 1))
        polygons = polygon_generator.generate_multiple_polygons(
            vertex_counts,
            args.shapes_per_count
        )

        # Save raw 2D polygon data
        save_polygons(polygons, polygon_dir)

        # Generate and save 2D visualizations if requested
        if args.visualization:
            visualize_multiple_pairs(polygons, output_dir / 'polygon_2d')

        if args.image:
            visualize_clean_pegs(polygons, output_dir / 'clean_pegs')

        save_3d_meshes(mesh_generator, polygons, output_dir,
                       save_pcd=args.point_cloud,
                       n_pcd_points=args.n_point_cloud_points)

        logger.info(f"Successfully generated meshes in {output_dir}")

    except Exception as e:
        logger.error(f"Error during polygon generation: {e}")
        raise


if __name__ == "__main__":
    main()

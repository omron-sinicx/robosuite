from argparse import ArgumentParser
from pathlib import Path
import random
import trimesh
import numpy as np


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, required=True, help="Path to the input STL file")
    parser.add_argument('-npts', type=int, default=1024, help="Number of points to sample from the mesh")
    parser.add_argument('-seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('-s', '--show', action='store_true', help="Show the mesh and point cloud")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load the mesh
    mesh: trimesh.Trimesh = trimesh.load(str(args.input), force='mesh')
    n_vertices = len(mesh.vertices)

    if args.show:
        mesh.show()

    # Sample points from the mesh
    pcd = np.concatenate(
        [mesh.vertices, mesh.sample(max(0, args.npts - n_vertices))],
        axis=0,
    )
    if args.show:
        trimesh.PointCloud(pcd).show()

    output_path = args.input.with_suffix('.pcd.npy')
    np.save(str(output_path), pcd)
    print(f'Saved to {output_path}')

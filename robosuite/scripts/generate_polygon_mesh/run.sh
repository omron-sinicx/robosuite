#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Check for required Python packages and install if missing
if ! python3 -c "import trimesh" &> /dev/null || ! python3 -c "import shapely" &> /dev/null; then
    echo "Installing required packages: trimesh[all], shapely"
    pip install --user 'trimesh[all]' shapely
fi

# Generate easy shapes (before 6/10)
# python3 main.py -c 20 -pcd -minv 3 -maxv 7 --seed 1 -o train_easy
# python3 main.py -c 20 -pcd -minv 3 -maxv 7 --seed 2 -o eval_easy

# Generate easy shapes (40mm)
tol=1

echo "Generating train_easy40 dataset..."
python3 gen_nested_poly.py -pcd -c 20 -minv 3 -maxv 7 -t $tol --safe-margin 3 -rs 10 40 --max-attempts 100000 --seed 1 -o train_easy40

echo "Generating eval_easy40 dataset..."
python3 gen_nested_poly.py -pcd -c 20 -minv 3 -maxv 7 -t $tol --safe-margin 3 -rs 10 40 --max-attempts 100000 --seed 2 -o eval_easy40

echo "Generating ood_easy40 dataset..."
python3 gen_nested_poly.py -pcd -c 20 -minv 8 -maxv 10 -t $tol --safe-margin 3 -rs 10 40 --max-attempts 100000 --seed 1 -o ood_easy40

echo "All dataset generation scripts completed successfully."

# Generate topo1 shapes
# tol=2
# python3 gen_nested_poly.py -pcd -c 20 -minv 3 -maxv 7 -t $tol --safe-margin 3 -rs 20 30 10 15 --max-attempts 100000 --seed 1 -o train_topo_tol${tol}
# python3 gen_nested_poly.py -pcd -c 20 -minv 3 -maxv 7 -t $tol --safe-margin 3 -rs 20 30 10 15 --max-attempts 100000 --seed 2 -o eval_topo_tol${tol}

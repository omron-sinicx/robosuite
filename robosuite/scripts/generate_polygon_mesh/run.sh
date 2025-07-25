#!/bin/bash

# Generate easy shapes (before 6/10)
# python3 main.py -c 20 -pcd -minv 3 -maxv 7 --seed 1 -o train_easy
# python3 main.py -c 20 -pcd -minv 3 -maxv 7 --seed 2 -o eval_easy

# Generate easy shapes (40mm)
tol=1
python3 gen_nested_poly.py -pcd -c 20 -minv 3 -maxv 7 -t $tol --safe-margin 3 -rs 10 40 --max-attempts 100000 --seed 1 -o train_easy40
python3 gen_nested_poly.py -pcd -c 20 -minv 3 -maxv 7 -t $tol --safe-margin 3 -rs 10 40 --max-attempts 100000 --seed 2 -o eval_easy40
python3 gen_nested_poly.py -pcd -c 20 -minv 8 -maxv 10 -t $tol --safe-margin 3 -rs 10 40 --max-attempts 100000 --seed 1 -o ood_easy40

# Generate topo1 shapes
# tol=2
# python3 gen_nested_poly.py -pcd -c 20 -minv 3 -maxv 7 -t $tol --safe-margin 3 -rs 20 30 10 15 --max-attempts 100000 --seed 1 -o train_topo_tol${tol}
# python3 gen_nested_poly.py -pcd -c 20 -minv 3 -maxv 7 -t $tol --safe-margin 3 -rs 20 30 10 15 --max-attempts 100000 --seed 2 -o eval_topo_tol${tol}

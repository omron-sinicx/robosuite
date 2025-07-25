#!/bin/bash

set -x;

# shapes=(H I K L N T V W X Y)
shapes=()

for shape in ${shapes[@]}; do
    peg_file="${shape}-peg.stl"
    ./decompose_mesh.sh -t peg -i $peg_file -o $peg_file
done

shapes=(M)

for shape in ${shapes[@]}; do
    hole_file="${shape}-hole.stl"
    ./decompose_mesh.sh -t hole -i $hole_file -o $hole_file
done
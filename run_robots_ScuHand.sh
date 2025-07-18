#!/bin/bash

# List all of the robots you’ve mapped (keys of ROBOT_CLASS_MAPPING)
robots=(
  "Baxter"
  "IIWA"
  "Jaco"
  "Kinova3"
  "Panda"
  "Sawyer"
  "UR5e"
  "SpotWithArm"
  "SpotWithArmFloating"
  "PandaOmron"
  "Tiago"
  "GR1"
  "GR1FixedLowerBody"
  "GR1ArmsOnly"
  "GR1FloatingBody"
  "PandaDexRH"
  "PandaDexLH"
  "XArm7"
)

for robot in "${robots[@]}"; do
  echo "=== Running $robot + ScuHand ==="
  python robosuite/demos/demo_composite_robot_ScuHand.py \
    --robot "$robot" \
    --grippers "ScuHand"
  echo "=== Done with $robot ==="
  echo
done

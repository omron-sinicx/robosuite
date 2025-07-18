#!/bin/bash

# Define the list of grippers (excluding None)
grippers=(
  "RethinkGripper"
  "PandaGripper"
  "JacoThreeFingerGripper"
  "JacoThreeFingerDexterousGripper"
  "WipingGripper"
  "Robotiq85Gripper"
  "Robotiq140Gripper"
  "RobotiqThreeFingerGripper"
  "RobotiqThreeFingerDexterousGripper"
  "BDGripper"
  "InspireLeftHand"
  "InspireRightHand"
  "FourierLeftHand"
  "FourierRightHand"
  "XArm7Gripper"
  "ScuHand"
)

# Loop through each gripper and run the demo
for gripper in "${grippers[@]}"; do
  echo "Running demo with gripper: $gripper"
  python robosuite/demos/demo_composite_robot.py --robot UR5e --grippers "$gripper"
  echo "Done with $gripper"
  echo "-----------------------------"
done

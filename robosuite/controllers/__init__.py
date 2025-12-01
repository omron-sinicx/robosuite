from .parts.controller_factory import controller_factory, load_part_controller_config
from .composite import composite_controller_factory, ALL_COMPOSITE_CONTROLLERS
from .composite.composite_controller_factory import load_composite_controller_config


PART_CONTROLLER_INFO = {
    "JOINT_VELOCITY": "Joint Velocity",
    "JOINT_TORQUE": "Joint Torque",
    "JOINT_POSITION": "Joint Position",
    "COMPLIANCE": "Operational Space Control with Compliance",
    "FDCC": "Forward Dynamics Compliance Control",
    "OSC_POSITION": "Operational Space Control (Position Only)",
    "OSC_POSITION_CB": "UPDATED: Operational Space Control (Position Only)",
    "OSC_POSE": "Operational Space Control (Position + Orientation)",
    "OSC_POSITION_FT": "Operational Space Control with force reference (Position Only)",
    "OSC_POSE_FT": "Operational Space Control with force reference (Position + Orientation)",

    "IK_POSE": "Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)",
}

ALL_PART_CONTROLLERS = PART_CONTROLLER_INFO.keys()

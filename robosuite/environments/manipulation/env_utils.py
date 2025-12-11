import logging
from pathlib import Path
import random
from typing import Tuple, List, Dict, Optional, Union
import os

import xml.etree.ElementTree as ET
import numpy as np
import yaml

import robosuite
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.objects.objects import MujocoXMLObject
from robosuite.models.objects.xml_objects import HoleObject
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.transform_utils import *
from robosuite.models.objects import (
    GuriguriLargeSquareHoleObject,
    GuriguriLargeRoundHoleObject,
    GuriguriLargeRectangleHoleObject,
    GuriguriLargeTriangleHoleObject,
)
import xml.dom.minidom as minidom

log = logging.getLogger(__name__)


BASIC_HOLES = {
    'round': GuriguriLargeRoundHoleObject,
    'square': GuriguriLargeSquareHoleObject,
    'triangle': GuriguriLargeTriangleHoleObject,
    'rectangle': GuriguriLargeRectangleHoleObject,
}

# Load peg configurations from YAML file


def load_peg_config():
    """
    Load peg configurations from YAML file.

    Returns:
        dict: Configuration dictionary for pegs and holes
    """
    config_path = os.path.join('/root/robosim/learning', 'config', 'pegs.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Global peg configuration
PEG_HOLE_CONFIG = load_peg_config()

# Cache for storing modified XML models
XML_MODEL_CACHE = {}

PEG_MESH_NAME_KEYWORD = 'peg_part'
HOLE_MESH_NAME_KEYWORD = 'hole_part'


def get_available_meshes(mesh_dir: Optional[str] = None) -> Dict[str, Dict[str, Union[List[str], str, None]]]:
    """
    Get available mesh directories for different shape categories.

    Args:
        mesh_dir: Optional path to look for mesh directories

    Returns:
        Dictionary with mesh categories and their available shapes
    """
    # Initialize with basic shapes
    available_meshes = {
        'basic': {'names': list(BASIC_HOLES.keys()), 'dir': None},
        'custom': {'names': [], 'dir': mesh_dir},
    }

    # Add custom shapes if directory is provided
    if mesh_dir:
        custom_dir = Path(mesh_dir) / 'peg'
        if custom_dir.exists() and custom_dir.is_dir():
            # Get all subdirectories as shape names
            shape_names = [d.name for d in custom_dir.iterdir() if d.is_dir()]
            if shape_names:
                available_meshes['custom']['names'] = shape_names
            else:
                available_meshes['custom']['dir'] = None  # No valid shapes found

    return available_meshes


def get_random_shape_from_dir(path: str) -> str:
    """
    Get a random shape name from a directory.

    Args:
        path: Path to look for shape directories

    Returns:
        str: Randomly selected shape name from the directory

    Raises:
        ValueError: If the directory doesn't exist or contains no subdirectories
    """
    path = Path(path)

    # Validate directory exists
    if not path.exists() or not path.is_dir():
        raise ValueError(f'Invalid directory path: {path}')

    # Get all subdirectories (each represents a shape)
    shape_dirs = [d.name for d in path.iterdir() if d.is_dir()]

    # Ensure there are shapes available
    if not shape_dirs:
        raise ValueError(f'No shape directories found in: {path}')

    # Return random choice
    return random.choice(shape_dirs)


class DynamicHoleObject(HoleObject):
    """
    Dynamically generated hole object based on mesh parts in the hole directory.
    """

    def __init__(self, name, shape, shape_type, mesh_dir=None, hole_pose="0 0 0"):
        """
        Initialize the dynamic hole object.

        Args:
            name: The name of the object
            shape: The shape name
            shape_type: The shape type (basic or custom)
            mesh_dir: Optional directory to look for meshes
        """
        # Create the XML model dynamically
        xml_string = self._generate_xml_model(shape, shape_type, mesh_dir)

        # Save the XML to a file
        asset_dir = Path(robosuite.models.assets_root)
        xml_path = asset_dir / "objects" / "custom-hole" / f"{os.getpid()}.xml"
        os.makedirs(os.path.dirname(xml_path), exist_ok=True)

        with open(xml_path, 'w') as f:
            f.write(xml_string)

        # Initialize the MujocoXMLObject with the saved XML file
        super().__init__(
            xml_path,
            name=name,
            pos=hole_pose,
        )

    def _generate_xml_model(self, shape, shape_type, mesh_dir=None):
        """
        Generate the XML model dynamically based on the mesh parts in the hole directory.

        Args:
            shape: The shape name
            mesh_dir: Optional directory to look for meshes

        Returns:
            str: The XML model as a string
        """
        # Get hole configuration from YAML
        hole_defaults = PEG_HOLE_CONFIG['hole']['defaults_params']
        hole_sets = PEG_HOLE_CONFIG['hole']['sets']

        # Determine the hole directory
        if mesh_dir:
            if isinstance(mesh_dir, str):
                hole_dir = Path(mesh_dir) / shape
            else:
                hole_dir = mesh_dir / shape
        else:
            hole_dir = Path(hole_sets[shape_type]['mesh_dir']) / shape

        # Get the mesh paths for the hole
        hole_parts = self._get_hole_parts(hole_dir)

        # Create the XML document
        doc = minidom.getDOMImplementation().createDocument(None, "mujoco", None)
        root = doc.documentElement
        root.setAttribute("model", "hole")

        # Add default section
        default = doc.createElement("default")
        root.appendChild(default)

        default_collision = doc.createElement("default")
        default_collision.setAttribute("class", "collision")
        default.appendChild(default_collision)

        geom = doc.createElement("geom")
        geom.setAttribute("group", "0")
        geom.setAttribute("type", "mesh")
        geom.setAttribute("solimp", array_to_string(hole_defaults['solimp']))
        geom.setAttribute("density", str(hole_defaults['density']))
        geom.setAttribute("solref", array_to_string(hole_defaults['solref']))
        geom.setAttribute("condim", str(hole_defaults['condim']))
        geom.setAttribute("friction", array_to_string(hole_defaults['friction']))
        default_collision.appendChild(geom)

        # Add asset section
        asset = doc.createElement("asset")
        root.appendChild(asset)

        # Add mesh assets for hole parts
        for i, part_path in enumerate(hole_parts):
            mesh = doc.createElement("mesh")
            mesh.setAttribute("name", f"hole_part_{i}")
            mesh.setAttribute("file", str(part_path))
            mesh.setAttribute("refpos", array_to_string(hole_defaults['refpos']))
            mesh.setAttribute("refquat", array_to_string(hole_defaults['refquat']))
            mesh.setAttribute("scale", "0.01 0.01 0.01")
            asset.appendChild(mesh)

        # Add worldbody section
        worldbody = doc.createElement("worldbody")
        root.appendChild(worldbody)

        body = doc.createElement("body")
        worldbody.appendChild(body)

        object_body = doc.createElement("body")
        object_body.setAttribute("name", "object")
        object_body.setAttribute("class", "collision")
        body.appendChild(object_body)

        # Add geoms for hole parts
        for i in range(len(hole_parts)):
            geom = doc.createElement("geom")
            geom.setAttribute("mesh", f"hole_part_{i}")
            geom.setAttribute("class", "collision")
            object_body.appendChild(geom)

        # Add sites
        bottom_site = doc.createElement("site")
        bottom_site.setAttribute("name", "bottom_site")
        bottom_site.setAttribute("rgba", "0 0 0 0")
        bottom_site.setAttribute("size", "0.005")
        bottom_site.setAttribute("pos", "0 0 -0.14")
        body.appendChild(bottom_site)

        top_site = doc.createElement("site")
        top_site.setAttribute("name", "top_site")
        top_site.setAttribute("rgba", "0 0 0 0")
        top_site.setAttribute("size", "0.005")
        top_site.setAttribute("pos", "0 0 0")
        body.appendChild(top_site)

        horizontal_radius_site = doc.createElement("site")
        horizontal_radius_site.setAttribute("name", "horizontal_radius_site")
        horizontal_radius_site.setAttribute("rgba", "0 0 0 0")
        horizontal_radius_site.setAttribute("size", "0.005")
        horizontal_radius_site.setAttribute("pos", "0.06 0.06 0")
        body.appendChild(horizontal_radius_site)

        # Convert the XML document to a string
        xml_string = doc.toprettyxml(indent="  ")

        # Remove extra whitespace and empty lines
        lines = [line for line in xml_string.split('\n') if line.strip()]
        xml_string = '\n'.join(lines)

        return xml_string

    def _get_hole_parts(self, hole_dir):
        """Get the hole parts from the specified directory with fallback options."""
        # Try to use the specified directory
        if not hole_dir.exists():
            raise ValueError(f"Hole directory not found: {hole_dir}")

        hole_parts = sorted([f for f in hole_dir.iterdir() if f.suffix == '.stl'])

        if not hole_parts:
            raise ValueError(f"No hole parts found in {hole_dir}")

        return hole_parts


class PegObject(MujocoXMLObject):
    def __init__(self, shape, shape_type, name='peg'):
        xml_string = self._generate_xml_model(shape, shape_type)

        # Save the XML to a file
        asset_dir = Path(robosuite.models.assets_root)
        xml_path = asset_dir / "objects" / "custom-peg" / f"{os.getpid()}.xml"
        os.makedirs(os.path.dirname(xml_path), exist_ok=True)

        with open(xml_path, 'w') as f:
            f.write(xml_string)

        super().__init__(
            xml_path,
            name=name,
            joints=None,  # [dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    def _generate_xml_model(self, shape, shape_type):
        peg_config = PEG_HOLE_CONFIG['pegs']['sets'][shape_type]
        peg_default_params = PEG_HOLE_CONFIG['pegs']['default_params']

        # Create new peg body
        peg_root = ET.Element('mujoco', {'model': 'peg'})
        asset = ET.SubElement(peg_root, 'asset')
        worldbody = ET.SubElement(peg_root, 'worldbody')
        body = ET.SubElement(worldbody, 'body')
        peg_body = ET.SubElement(body, 'body', {
            'name': 'object',
            'quat': array_to_string(peg_default_params['quat']),
        })

        if shape_type == 'basic':
            self._add_basic_peg(asset, peg_body, shape, peg_config, peg_default_params)
        else:  # custom shape
            self._add_custom_peg(asset, peg_body, shape, peg_config, peg_default_params)

        return ET.tostring(peg_root, encoding='unicode', method='xml')

    def _add_basic_peg(self, asset, peg_body, shape, peg_config, peg_default_params):
        """Add a basic peg shape to the gripper."""
        # Get mesh information
        mesh_name = peg_config['shapes'][shape]['mesh_name']
        mesh_file = peg_config['shapes'][shape]['mesh_file']
        refpos = peg_default_params['refpos']
        refquat = peg_default_params['refquat']

        # Check if the mesh file exists
        asset_dir = Path(robosuite.models.assets_root)
        full_mesh_path = asset_dir / 'grippers' / mesh_file
        if not full_mesh_path.exists():
            raise ValueError(f"Mesh file not found: {full_mesh_path}")

        # Add mesh to assets if it doesn't exist
        if asset.find(f"./mesh[@name='{mesh_name}']") is None:
            ET.SubElement(asset, 'mesh', {
                'name': mesh_name,
                'file': str(full_mesh_path),
                'refpos': array_to_string(refpos),
                'refquat': array_to_string(refquat)
            })

        # Add visual geom
        ET.SubElement(peg_body, 'geom', {
            'type': 'mesh',
            'mesh': mesh_name,
            'group': '1',
            'contype': '0',
            'conaffinity': '0',
            'rgba': array_to_string(peg_default_params['rgba'])
        })

        # Add collision geom
        ET.SubElement(peg_body, 'geom', {
            'type': 'mesh',
            'mesh': mesh_name,
            'group': '0',
            'contype': '0',
            'conaffinity': '1',
            'solref': array_to_string(peg_default_params['solref']),
            'friction': array_to_string(peg_default_params['friction']),
            'condim': str(peg_default_params['condim'])
        })

    def _add_custom_peg(self, asset, peg_body, shape, peg_config, peg_default_params):
        """Add a custom peg shape to the gripper."""
        peg_dir = Path(peg_config['mesh_dir']) / shape

        # Get mesh paths for the custom shape
        try:
            peg_parts = self.get_mesh_paths(peg_dir)
        except ValueError as e:
            raise ValueError(f"Error getting mesh paths: {e}")

        # Clear existing peg_part meshes from assets
        for mesh in list(asset.findall('mesh')):
            if PEG_MESH_NAME_KEYWORD in mesh.get('name', ''):
                asset.remove(mesh)

        # Add new mesh assets for peg parts
        for i, part_path in enumerate(peg_parts):
            mesh_name = f"{PEG_MESH_NAME_KEYWORD}_{i}"

            refpos = peg_config.get('refpos', peg_default_params['refpos'])
            refquat = peg_config.get('refquat', peg_default_params['refquat'])

            # Add mesh to assets
            ET.SubElement(asset, 'mesh', {
                'name': mesh_name,
                'file': str(part_path),
                'refpos': array_to_string(refpos),
                'refquat': array_to_string(refquat)
            })

            # Add visual geom
            ET.SubElement(peg_body, 'geom', {
                'type': 'mesh',
                'mesh': mesh_name,
                'group': '1',
                'contype': '0',
                'conaffinity': '0',
                'rgba': array_to_string(peg_default_params['rgba'])
            })

            # Add collision geom
            ET.SubElement(peg_body, 'geom', {
                'type': 'mesh',
                'mesh': mesh_name,
                'group': '0',
                'contype': '0',
                'conaffinity': '1',
                'solref': array_to_string(peg_default_params['solref']),
                'friction': array_to_string(peg_default_params['friction']),
                'condim': str(peg_default_params['condim'])
            })

    def get_mesh_paths(self, peg_dir) -> List[Path]:
        """
        Get the mesh paths for the peg.

        Args:
            peg_dir: Directory containing peg mesh files

        Returns:
            List[Path]: List of paths to peg mesh files
        """
        if not peg_dir.exists():
            raise ValueError(f'Mesh directory not found: {peg_dir}')
        peg_parts = sorted([f for f in peg_dir.iterdir() if f.suffix == '.stl'])
        if not peg_parts:
            raise ValueError(f'No peg parts found in {peg_dir}')
        return peg_parts


def get_camera_pose(view_direction, table_offset):
    """
    Get camera pose based on the specified view direction.

    Args:
        view_direction: Direction to view from ('real_calib', 'close', 'front', 'left', 'right', 'top')
        table_offset: Offset from the table center

    Returns:
        Tuple[np.ndarray, np.ndarray]: Camera position and quaternion
    """
    # FIXME: use comprehensive camera poses calculation
    # Initialize with identity quaternion (WXYZ format for Mujoco)
    cam_quat = np.array([0, 0, 0, 1])

    if view_direction == 'real_0801':
        cam_pos = np.array([0.497684, 0.660777, 0.541988])
        cam_quat = np.array([-0.272603, 0.637648, 0.661787, -0.284834])
        cam_quat = quat_multiply(cam_quat, np.array([0, 1, 0, 0]))
    elif view_direction == 'real_0807_d405':
        cam_pos = np.array([0.271233, 0.461096, 0.292791])
        cam_quat = np.array([-0.461784, 0.779215, 0.359905, -0.223714])
        cam_quat = quat_multiply(np.array([0, 1, 0, 0]), cam_quat)
        cam_quat = quat_multiply(np.array([1, 0, 0, 0]), cam_quat)
    elif view_direction == 'real_0807_d435':
        cam_pos = np.array([0.425442, 0.626776, 0.408199])
        cam_quat = np.array([-0.296421, 0.625465, 0.652314, -0.308892])
        cam_quat = quat_multiply(np.array([0, 1, 0, 0]), cam_quat)
        cam_quat = quat_multiply(np.array([1, 0, 0, 0]), cam_quat)
    elif view_direction == 'real_0904_d415':
        cam_pos = np.array([0.443744, 0.650226, 0.423445])
        cam_quat = np.array([-0.306215, 0.623489, 0.659918, -0.28636])
        cam_quat = quat_multiply(np.array([0, 1, 0, 0]), cam_quat)
        cam_quat = quat_multiply(np.array([1, 0, 0, 0]), cam_quat)
    elif view_direction == "close":
        # A close view from the side (the robot base at left)
        cam_quat = quat_multiply(axisangle2quat([0, 0, np.pi / 2]), cam_quat)
        cam_quat = quat_multiply(axisangle2quat([0, np.pi / 2, 0]), cam_quat)
        cam_quat = quat_multiply(axisangle2quat([0, 0, -np.pi / 6]), cam_quat)
        cam_pos = np.array([0.3, 0, 0.3]) + table_offset
    elif view_direction == "front":
        # Front view
        cam_quat = quat_multiply(axisangle2quat([0, 0, np.pi / 2]), cam_quat)
        cam_quat = quat_multiply(axisangle2quat([0, 0, -np.pi / 6]), cam_quat)
        cam_pos = np.array([0, 0.6, 0.5]) + table_offset
    elif view_direction == "left":
        # Look from side (the robot base at left)
        cam_quat = quat_multiply(axisangle2quat([0, 0, np.pi / 2]), cam_quat)
        cam_quat = quat_multiply(axisangle2quat([0, np.pi / 2, 0]), cam_quat)
        cam_quat = quat_multiply(axisangle2quat([0, 0, -np.pi / 6]), cam_quat)
        cam_pos = np.array([0.6, 0, 0.5]) + table_offset
    elif view_direction == "right":
        # Look from side (the robot base at right)
        cam_quat = quat_multiply(axisangle2quat([0, 0, np.pi / 2]), cam_quat)
        cam_quat = quat_multiply(axisangle2quat([0, -np.pi / 2, 0]), cam_quat)
        cam_quat = quat_multiply(axisangle2quat([0, 0, -np.pi / 6]), cam_quat)
        cam_pos = np.array([-0.6, 0, 0.5]) + table_offset
    elif view_direction == "top":
        # Look from top
        cam_pos = np.array([0, 0, 2]) + table_offset
    else:
        raise ValueError(
            f"Invalid view direction: {view_direction}. Expected one of [real_calib, close, front, left, right, top]")

    return cam_pos, cam_quat


def randomize_spring_params(robot_body, spring_cfg):
    rxry_stiffness = spring_cfg['rxry_stiffness']
    rxry_damping = spring_cfg['rxry_damping']
    z_stiffness = spring_cfg['z_stiffness']
    z_damping = spring_cfg['z_damping']
    assert len(rxry_stiffness) == 2 and len(rxry_damping) == 2, \
        "rxry_stiffness and rxry_damping must be lists of two values each"
    assert len(z_stiffness) == 2 and len(z_damping) == 2, \
        "z_stiffness and z_damping must be lists of two values each"

    joints = robot_body.findall('.//joint')
    rxy_joints = [joint for joint in joints
                  if any(keyword in joint.attrib.get('name', '') for keyword in ['flex_wrist_rx', 'flex_wrist_ry'])]
    z_joints = [joint for joint in joints if 'flex_wrist_rz' in joint.attrib.get('name', '')]
    assert rxy_joints, "No flex_wrist_rx or flex_wrist_ry joints found in the robot body"
    assert z_joints, "No flex_wrist_rz joint found in the robot body"
    z_joint = z_joints[0]

    stiffness = np.random.uniform(rxry_stiffness[0], rxry_stiffness[1])
    damping = np.random.uniform(rxry_damping[0], rxry_damping[1])
    for joint in rxy_joints:
        if 'stiffness' in joint.attrib:
            joint.attrib['stiffness'] = f'{stiffness:.4f}'
        if 'damping' in joint.attrib:
            joint.attrib['damping'] = f'{damping:.4f}'

    z_joint.attrib['stiffness'] = f'{np.random.uniform(z_stiffness[0], z_stiffness[1]):.4f}'
    z_joint.attrib['damping'] = f'{np.random.uniform(z_damping[0], z_damping[1]):.4f}'


def scale_peg_and_hole(peg_wrapper, asset, hole, peg_size_range, curriculum_variance_coef):
    """
    Scale the peg and hole meshes based on curriculum parameters.

    Args:
        peg_wrapper (ET.Element): The XML element for the peg wrapper
        asset (ET.Element): The XML asset element containing meshes
        hole (MujocoXMLObject): The hole object
        peg_size_range (np.ndarray): Range of allowed peg sizes [min, max]
        curriculum_variance_coef (float): Curriculum coefficient for variance (0-1)

    Returns:
        tuple: Scaled peg wrapper and hole object
    """

    # Compute randomized peg size
    min_peg_size, max_peg_size = compute_domain_randomization_range(
        'peg_size', peg_size_range[0], peg_size_range[1],
        curriculum_variance_coef, max_value_is_easier=True)

    # Generate random scale factors for x and y dimensions
    sx, sy = np.random.uniform(min_peg_size, max_peg_size, size=2)

    # Scale peg meshes
    for geom in peg_wrapper.findall('.//geom'):
        if 'type' in geom.attrib and geom.attrib['type'] == 'mesh':
            mesh_name = geom.attrib['mesh']
            mesh = asset.find(f"./mesh[@name='{mesh_name}']")
            if mesh is not None:
                mesh.attrib['scale'] = array_to_string([sx, sy, 1.0])

    # Scale hole with additional clearance based on curriculum
    # Add more clearance when curriculum_variance_coef is low
    hole_sx = sx * (1 + (1 - curriculum_variance_coef) * 0.10)
    hole_sy = sy * (1 + (1 - curriculum_variance_coef) * 0.10)

    for mesh in hole.asset.findall('mesh'):
        mesh.attrib['scale'] = array_to_string([hole_sx, hole_sy, 1.0])

    return peg_wrapper, hole


def set_peg_and_hole_color(peg_wrapper, hole: DynamicHoleObject, cfg):
    """
    Randomize the color of peg and hole meshes based on curriculum parameters.

    Args:
        peg_wrapper (ET.Element): The XML element for the peg wrapper
        hole (MujocoXMLObject): The hole object
        cfg (dict): config for peg and hole colors

    Returns:
        tuple: Scaled peg wrapper and hole object
    """
    # Generate colors
    alpha = 1  # default to opaque
    peg_default_clr = cfg['peg_default']
    hole_default_clr = cfg['hole_default']

    interp = cfg.get('interp', 0)
    if interp > 0:
        peg_clr = np.clip(peg_default_clr - np.random.uniform(-interp, interp, size=3), 0, 1)
        hole_clr = np.clip(hole_default_clr - np.random.uniform(-interp, interp, size=3), 0, 1)
    else:
        peg_clr = peg_default_clr
        hole_clr = hole_default_clr
    peg_clr = array_to_string([*peg_clr, alpha])
    hole_clr = array_to_string([*hole_clr, alpha])

    # peg
    for geom in peg_wrapper.findall('.//geom'):
        geom.attrib['rgba'] = peg_clr

    # hole
    for (_, geom) in hole._get_geoms(hole._obj):
        geom_name = geom.get('name', '')
        if 'hole' in geom_name:
            geom.set('rgba', hole_clr)

    return peg_wrapper, hole


def compute_domain_randomization_range(
    feature_name: str,
    min_value: float,
    max_value: float,
    curriculum_coefficient: float,
    max_value_is_easier: bool = False
) -> Tuple[float, float]:
    """
    Compute a dynamically adjusted range for domain randomization based on a curriculum learning approach.

    This function helps in gradually adjusting the range of a feature during training,
    making the task progressively easier or harder based on the curriculum coefficient.

    Parameters:
    -----------
    feature_name : str
        Name of the feature being adjusted (used for error messaging)
    min_value : float
        The smallest value in the original range
    max_value : float
        The largest value in the original range
    curriculum_coefficient : float
        A value between 0 and 1 that determines how much of the range to use
        - 0 means using the easiest part of the range
        - 1 means using the full original range
    max_value_is_easier : bool, optional
        Determines the direction of difficulty
        - If True: Higher values are considered easier
        - If False: Lower values are considered easier
        Default is False

    Returns:
    --------
    tuple[float, float]
        A tuple containing (updated_min_value, updated_max_value)

    Raises:
    -------
    ValueError
        If min_value is not less than max_value
    """
    # Validate input values
    if min_value > max_value:
        raise ValueError(
            f"Invalid range for {feature_name}: min_value ({min_value}) "
            f"must be less than max_value ({max_value})"
        )

    # Validate curriculum coefficient
    if not 0 <= curriculum_coefficient <= 1:
        raise ValueError(
            f"Curriculum coefficient must be between 0 and 1, "
            f"got {curriculum_coefficient}"
        )

    # Compute the total range
    value_range = max_value - min_value

    # Adjust range based on curriculum approach
    if max_value_is_easier:
        # If higher values are easier, shrink from the lower end
        updated_min_value = max_value - (curriculum_coefficient * value_range)
        updated_max_value = max_value
    else:
        # If lower values are easier, shrink from the higher end
        updated_min_value = min_value
        updated_max_value = min_value + (curriculum_coefficient * value_range)

    return updated_min_value, updated_max_value


def clear_xml_model_cache():
    """
    Clear the XML model cache to free up memory or for debugging purposes.

    Returns:
        int: The number of cached models that were cleared
    """
    count = len(XML_MODEL_CACHE)
    XML_MODEL_CACHE.clear()
    return count


def update_xml(
    xml_root, robot_configs, hole, peg_size_range, curriculum_variance_coef,
    color_cfg=None, spring_cfg=None
):
    """
    Update the MuJoCo model.

    Args:
        xml_root (ET.Element): The XML root element
        robot_configs (list): List of robot configurations
        hole (MujocoXMLObject): The hole object
        shape (str): The shape name
        shape_type (str): The shape type ('basic' or 'custom')
        peg_size_range (np.ndarray): Range of allowed peg sizes [min, max]
        curriculum_variance_coef (float): Curriculum coefficient for variance (0-1)
        color_cfg (dict, optional): Configuration for peg and hole colors, if None uses default colors
        spring_cfg (dict, optional): Configuration for spring parameters, if None no springs are randomized

    Returns:
        None
    """
    # Find the robot body in the merged model
    robot_body = xml_root.find(f".//body[@name='robot0_base']")
    if robot_body is None:
        raise ValueError("Robot body not found in the merged model")

    # Get the asset element
    asset = xml_root.find("./asset")
    if asset is None:
        raise ValueError("Asset element not found in the merged model")

    # Process each robot
    for idx, robot in enumerate(robot_configs):
        # Get the prefix for the current robot
        prefix = f"gripper{idx}_{list(robot['composite_controller_config']['body_parts'].keys())[0]}"

        # Find the peg wrapper
        peg_wrapper = robot_body.find(f".//body[@name='{prefix}_peg_wrapper']")
        if peg_wrapper is None:
            raise ValueError(f"Peg wrapper not found for prefix: {prefix}")

        scale_peg_and_hole(peg_wrapper, asset, hole, peg_size_range, curriculum_variance_coef)
        set_peg_and_hole_color(peg_wrapper, hole, color_cfg)

    if spring_cfg is not None:
        randomize_spring_params(robot_body, spring_cfg)


def get_peg_shape(shape_type, shape=None):
    peg_sets = PEG_HOLE_CONFIG.get('pegs', {}).get('sets', {})
    if shape_type in peg_sets:
        # Get the set configuration
        set_config = peg_sets[shape_type]
        set_type = set_config.get('type')

        if set_type == 'single_mesh':
            # Choose a random shape from the basic shapes defined in this set
            shapes = list(set_config.get('shapes', {}).keys())
            if shapes:
                shape = random.choice(shapes)
                return shape
        elif set_type == 'custom':
            # Choose a random shape from the custom mesh directory
            mesh_dir = set_config['mesh_dir']
            shape = get_random_shape_from_dir(mesh_dir)
            return shape
        else:
            raise ValueError(f'Invalid set type {set_type} for peg set {shape_type}')
    else:
        raise ValueError(f'Invalid shape type {shape_type}')


def get_peg_point_cloud(shape, shape_type) -> np.ndarray:
    peg_sets = PEG_HOLE_CONFIG.get('pegs', {}).get('sets', {})
    if shape_type in peg_sets:
        # Get the set configuration
        set_config = peg_sets[shape_type]
        set_type = set_config.get('type')

        if set_type == 'single_mesh':
            asset_dir = Path(robosuite.models.assets_root)
            mesh_file = set_config['shapes'][shape]['mesh_file']
            mesh_path = asset_dir / 'grippers' / mesh_file
            pcd_path = mesh_path.with_suffix('.pcd.npy')
            if not pcd_path.exists():
                raise ValueError(f'Point cloud file not found: {pcd_path}')
            pcd = np.load(pcd_path)
            return pcd
        elif set_type == 'custom':
            mesh_dir = set_config['mesh_dir']
            shape_dir = Path(mesh_dir) / shape
            pcd_path = shape_dir / 'pcd.npy'
            if not pcd_path.exists():
                raise ValueError(f'Point cloud file not found: {pcd_path}')
            pcd = np.load(pcd_path)
            return pcd
        else:
            raise ValueError(f'Invalid set type {set_type} for peg set {shape_type}')
    else:
        raise ValueError(f'Invalid shape type {shape_type}')


def get_hole_object(shape, shape_type, hole_pose) -> MujocoXMLObject:
    """
    Get the hole object for the given shape.

    Args:
        shape: The shape name
        shape_type: The shape type (basic or custom)

    Returns:
        MujocoXMLObject: The hole object
    """
    hole_sets = PEG_HOLE_CONFIG.get('hole', {}).get('sets', {})
    if shape_type in hole_sets:
        # Get the set configuration
        set_config = hole_sets[shape_type]
        set_type = set_config.get('type')

        if set_type == 'single_mesh':
            # Use basic hole objects from predefined dictionary
            hole_class = BASIC_HOLES.get(shape, BASIC_HOLES["round"])  # Default to round if shape not found
            return hole_class(name="hole", pos=array_to_string(hole_pose))
        elif set_type == 'custom':
            # For custom shape type, use the dynamically generated hole object
            mesh_dir = set_config['mesh_dir']
            return DynamicHoleObject(name="hole", shape=shape, shape_type=shape_type, mesh_dir=mesh_dir, hole_pose=array_to_string(hole_pose))
        else:
            raise ValueError(f'Invalid set type {set_type} for hole set {shape_type}')
    else:
        raise ValueError(f'Invalid shape type {shape_type}')

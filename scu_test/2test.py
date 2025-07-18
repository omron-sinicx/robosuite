import mujoco
from mujoco import viewer
import numpy as np
import time
from collections import defaultdict

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path("circular_cloth_funnel.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Identify all cloth geoms (by name prefix "G") and compute center from visible ones
cloth_geom_ids = []
visible_positions = []

for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name and name.startswith("G"):
        cloth_geom_ids.append(i)
        size = model.geom_size[i][0]
        body_id = model.geom_bodyid[i]
        mass = model.body_mass[body_id]
        if size > 1e-6 and mass > 1e-6:
            visible_positions.append(data.xpos[i][:2])

cloth_center = np.mean(np.array(visible_positions), axis=0)

# Parameters
visible_radius = 0.15
lift_height = 0.05

# Step 1: Mask geoms outside the visible circle and lift valid ones
valid_geom_ids = []
for i in cloth_geom_ids:
    pos = data.xpos[i]
    dist = np.linalg.norm(pos[:2] - cloth_center)
    model.geom_pos[i][2] = lift_height
    if dist <= visible_radius:
        valid_geom_ids.append(i)
        model.geom_rgba[i][:3] = np.array([1.0, 0.0, 0.0])  # red
    else:
        model.geom_size[i][:] = 0.0
        body_id = model.geom_bodyid[i]
        model.body_mass[body_id] = 0.0

# Step 2: Select a radius direction and mark geoms along it
radius_dir = np.array([1.0, 0.0])  # initial direction
radius_dir /= np.linalg.norm(radius_dir)
angle_offset = np.deg2rad(30)      # 10 degrees
threshold = 0.01

def rotate(vec, angle_rad):
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                           [np.sin(angle_rad),  np.cos(angle_rad)]])
    return rot_matrix @ vec

left_dir = rotate(radius_dir, +angle_offset)
right_dir = rotate(radius_dir, -angle_offset)

geom_groups = defaultdict(list)
geom_body_map = {}

for i in valid_geom_ids:
    pos = data.xpos[i][:2]
    rel = pos - cloth_center

    for name, direction in [('center', radius_dir), ('left', left_dir), ('right', right_dir)]:
        proj = np.dot(rel, direction)
        if proj < 0:
            continue
        perp = rel - proj * direction
        if np.linalg.norm(perp) < threshold:
            geom_groups[name].append(i)
            geom_body_map[i] = model.geom_bodyid[i]
            if name == 'center':
                model.geom_rgba[i] = np.array([0.0, 1.0, 0.0, 1.0])
            elif name == 'left':
                model.geom_rgba[i] = np.array([1.0, 1.0, 0.0, 1.0])
            elif name == 'right':
                model.geom_rgba[i] = np.array([0.0, 1.0, 1.0, 1.0])
            break

# Step 3: Assign opposite velocities along radius_dir
vel_mag = 0.1
body_velocities = {}

for name, sign in [('left', +1), ('right', -1)]:
    for geom_id in geom_groups[name]:
        body_id = geom_body_map[geom_id]
        body_velocities[body_id] = sign * radius_dir * vel_mag

# Step 4: Animate and apply velocities
num_steps = 3000
with viewer.launch_passive(model, data) as v:
    for step in range(num_steps):
        for body_id, vel in body_velocities.items():
            data.qvel[model.body_dofadr[body_id]:model.body_dofadr[body_id]+2] = vel  # x, y only

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)

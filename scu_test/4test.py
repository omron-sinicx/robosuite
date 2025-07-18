import mujoco
from mujoco import viewer
import numpy as np
import time
from collections import defaultdict
from mujoco import mj_id2name, mj_name2id, mjtObj

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path("spoon_cloth_scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Step 1: Identify spoon_head cloth geoms and compute their center
visible_positions = []
cloth_geom_ids = []


# print(model.ngeom, data.xpos.shape)

for name_id in range(model.ngeom):
    name = mj_id2name(model, mjtObj.mjOBJ_GEOM, name_id)
    print(name)
    if name and name.startswith("spoon_headG") and name not in cloth_geom_ids:
        geom_id = mj_name2id(model, mjtObj.mjOBJ_GEOM, name)
        body_id = model.geom_bodyid[geom_id]

        if model.body_mass[body_id] > 1e-6:
            cloth_geom_ids.append(name_id)
            visible_positions.append(data.xpos[geom_id][:2])


if visible_positions:
    cloth_center = np.mean(np.array(visible_positions), axis=0)
else:
    raise RuntimeError("No visible geoms found for spoon_head to compute center.")


# Step 2: Filter circular region
visible_radius = 0.05
lift_height = 0.05
valid_geom_ids = []

for i in cloth_geom_ids:
    pos = data.xpos[i]
    dist = np.linalg.norm(pos[:2] - cloth_center)
    model.geom_pos[i][2] = lift_height
    if dist <= visible_radius:
        valid_geom_ids.append(i)
        model.geom_rgba[i][:3] = np.array([1.0, 0.0, 0.0])  # red
    # else:
    #     model.geom_size[i][:] = 0.0
    #     body_id = model.geom_bodyid[i]
    #     model.body_mass[body_id] = 0.0

# Step 3: Pick center, left, right radius lines
radius_dir = np.array([1.0, 0.0])
radius_dir /= np.linalg.norm(radius_dir)
angle_offset = np.deg2rad(10)
threshold = 0.005

def rotate(vec, angle_rad):
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    return R @ vec

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

# Step 4: Compute velocity to move left/right geoms to center radius line
vel_mag = 1
body_velocities = {}

for side in ['left', 'right']:
    for geom_id in geom_groups[side]:
        body_id = geom_body_map[geom_id]
        current_pos = data.xpos[geom_id][:2]
        rel_vec = current_pos - cloth_center
        proj_len = np.dot(rel_vec, radius_dir)
        target_pos = cloth_center + proj_len * radius_dir
        move_vec = target_pos - current_pos
        norm = np.linalg.norm(move_vec)
        if norm > 1e-6:
            vel = move_vec / norm * vel_mag
            body_velocities[body_id] = vel

# Step 5: Animate with individual stopping per body
num_steps = 5000000
stopping_threshold = 0.0001  # Distance tolerance to stop
active_bodies = {}

# Precompute individual target positions
for geom_id in geom_groups['left'] + geom_groups['right']:
    body_id = geom_body_map[geom_id]
    current_pos = data.xpos[geom_id][:2]
    rel_vec = current_pos - cloth_center
    proj_len = np.dot(rel_vec, radius_dir)
    target_pos = cloth_center + proj_len * radius_dir
    active_bodies[body_id] = {
        "geom_id": geom_id,
        "target": target_pos
    }

with viewer.launch_passive(model, data) as v:
    for step in range(num_steps):
        to_remove = []

        for body_id, info in active_bodies.items():
            geom_id = info["geom_id"]
            target_pos = info["target"]
            current_pos = data.xpos[geom_id][:2]
            move_vec = target_pos - current_pos
            dist = np.linalg.norm(move_vec)

            if dist < stopping_threshold:
                data.qvel[model.body_dofadr[body_id]:model.body_dofadr[body_id]+2] = 0
                to_remove.append(body_id)
            else:
                vel = move_vec / dist * vel_mag
                data.qvel[model.body_dofadr[body_id]:model.body_dofadr[body_id]+2] = vel

        for b in to_remove:
            del active_bodies[b]

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)

        if step % 100 == 0:
            print(f"Step {step}: {len(active_bodies)} bodies still moving.")

        if not active_bodies:
            print(f"🎉 All geoms reached center radius line at step {step}")
            break



import mujoco
from mujoco import viewer
import numpy as np
import time

# Load model and data
model = mujoco.MjModel.from_xml_path("circular_cloth_funnel.xml")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

# Find cloth-related joint IDs
cloth_joint_ids = []
geom_positions = []
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and name.startswith("J"):  # composite cloth joints typically named like J0_...
        cloth_joint_ids.append(i)

for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name and name.startswith("G"):
        geom_positions.append(data.xpos[i][:2])

cloth_center = np.mean(np.array(geom_positions), axis=0)
print("Auto-detected cloth center:", cloth_center)

# Precompute per-joint body distances to center (used to compute deformation amount)
joint_dists = []
for jnt_id in cloth_joint_ids:
    body_id = model.jnt_bodyid[jnt_id]
    pos = model.body_pos[body_id][:2]
    dist = np.linalg.norm(pos - cloth_center)
    joint_dists.append(dist)

joint_dists = np.array(joint_dists)
max_dist = np.max(joint_dists)

# Animation loop
num_steps = 3000
with viewer.launch_passive(model, data) as v:
    for step in range(num_steps):
        for idx, jnt_id in enumerate(cloth_joint_ids):
            dof_addr = model.jnt_dofadr[jnt_id]
            # Deform upward along z in a smooth funnel shape
            height = 0.5 + 0.5 * (joint_dists[idx] / max_dist) * (step / num_steps)
            data.qpos[dof_addr] = height  # set z-position of the joint (main kind has 3 DOFs)

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)

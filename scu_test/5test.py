import mujoco
from mujoco import viewer
import numpy as np
import time
from mujoco import mj_name2id, mj_id2name, mjtObj
from scipy.spatial.transform import Rotation as R

# Load model and data
model = mujoco.MjModel.from_xml_path("spoon_2_cloth_scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Loop through all body names
for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    # print(body_name)
    if body_name and body_name.startswith("spoon_head2B"):
        model.body_mass[i] = 0

for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if body_name and body_name.startswith("spoon_head1B"):
        model.body_mass[i] = 0

# Collect all geoms starting with 'spoon_head2G'
geom_ids = []
body_ids = set()

target_geom_names = [f"spoon_head2G{i}_0" for i in range(5, 11)]

# handle_names = [f"handle{i}" for i in range(6)]


for i in range(model.ngeom):
    name = mj_id2name(model, mjtObj.mjOBJ_GEOM, i)
    # if name and name.startswith("spoon_head2G"):
    if name in target_geom_names:
        # print(name)
        geom_ids.append(i)
        body_ids.add(model.geom_bodyid[i])

body_ids = list(body_ids)
print("✅ Moving body IDs:", body_ids)


# 1. Get body id
spoon_base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon_base")
dofadr = model.body_dofadr[spoon_base_bid]
qpos_adr = model.jnt_qposadr[spoon_base_bid]
z_pos = 0.108



# Actuator indices (assuming order is: x, y, z, q1, q2, q3)
x_idx, y_idx, z_idx = 0, 1, 2
q1_idx, q2_idx, q3_idx = 3, 4, 5

# Initial + target z
z_current = 0.1
z_target = 0.05
z_step = -0.0005

# Target quaternion from Euler (Y axis rotation)
target_rot = R.from_euler('y', -45, degrees=True).as_quat()
# MuJoCo quaternion order: [w, x, y, z]
quat_mj = np.array([target_rot[3], target_rot[0], target_rot[1], target_rot[2]])

print(quat_mj)

with viewer.launch_passive(model, data) as v:
    time.sleep(2)

    for step in range(10000):


        for bid in body_ids:
            dofadr = model.body_dofadr[bid]
            if model.body_dofnum[bid] >= 2:
                data.qvel[dofadr + 1] = -0.2  # y-axis velocity


        # if z_current > z_target:
        #     z_current += z_step
        # else:
        #     z_current = z_target

        # Apply to position actuators
        data.ctrl[x_idx] = 0.0
        data.ctrl[y_idx] = 0.0
        data.ctrl[z_idx] = 0.004

        # Apply to orientation actuators (match q1, q2, q3 from quaternion)
        data.ctrl[q1_idx] = 0
        data.ctrl[q2_idx] = -0.3
        data.ctrl[q3_idx] = 0

        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)





# # Apply constant Y+ velocity
# vel_y = -0.1  # Adjust as needed
# with viewer.launch_passive(model, data) as v:
#     for step in range(5000):
#         # for bid in body_ids:
#         #     dofadr = model.body_dofadr[bid]
#         #     if model.body_dofnum[bid] >= 2:
#         #         data.qvel[dofadr + 1] = vel_y  # y-axis velocity
        
#         data.qvel[model.body_dofadr[spoon_base_bid] + 2] = 0.2  # z-axis
        
#         # z_pos += 0.0002  # Lift over time
#         # data.qpos[qpos_adr:qpos_adr+3] = np.array([0.0, 0.0, z_pos])
#         # quat = R.from_euler('y', -45, degrees=True).as_quat()
#         # data.qpos[qpos_adr+3:qpos_adr+7] = np.array([quat[3], quat[0], quat[1], quat[2]])
        
        
#         mujoco.mj_step(model, data)
#         v.sync()
#         time.sleep(0.01)

import mujoco
from mujoco import viewer
import numpy as np
import time
from mujoco import mj_name2id, mj_id2name, mjtObj
from scipy.spatial.transform import Rotation as R
import itertools

print("Current Mujoco Version is", mujoco.__version__)

# Load model and data
model = mujoco.MjModel.from_xml_path("ctrls_circles_cloth1.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Print flex info
print(f"Number of flexible vertices: {model.nflexvert}")
print(f"Number of flexible elements: {model.nflexelem}")

prefix = "c_"
max_check = 98  # Arbitrary upper limit, usually you won't need more than 300

for i in range(max_check):
    name = f"{prefix}{i}"
    try:
        body_id = mj_name2id(model, mjtObj.mjOBJ_BODY, name)
        pos = data.xpos[body_id]
        print(f"{name}: {pos}")
    except Exception:
        break  # Stops at the first missing name — efficient and safe

# print(f"Total flex vertices: {model.nflexvert}")
# for i in range(model.nflexvert):
#     print(f"c_{i}: xpos = {data.xpos[mj_name2id(model, mjtObj.mjOBJ_BODY, f'c_{i}')]} | flexvert = {data.flexvert_xpos[i]}")


# Get body IDs
left_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left")
right_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right")

# Print initial positions
print("\nInitial left position:", data.xpos[left_bid])
print("Initial right position:", data.xpos[right_bid])

# Actuator control indices
LEFT_CTRL_START = 0     # left has indices 0–5
RIGHT_CTRL_START = 6    # right has indices 6–11

# Control values
x_speed = 0.03
y_speed = 0.03

with viewer.launch_passive(model, data) as v:
    time.sleep(3)
    # v.opt.label = mujoco.mjtLabel.mjLABEL_BODY

    # # Make sure this is run after v.sync() and v is a mujoco.viewer.Handle
    # v.user_scn.ngeom = 0
    # i = 0

    # for j in range(100):
    #     name = f"c_{j}"
    #     try:
    #         bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    #         pos = data.xpos[bid]
    #         mujoco.mjv_initGeom(
    #             v.user_scn.geoms[i],
    #             type=mujoco.mjtGeom.mjGEOM_SPHERE,
    #             size=[0.001, 0, 0],  # Small sphere
    #             pos=pos,
    #             mat=np.eye(3).flatten(),
    #             rgba=[1, 0, 0, 0.1],  # Red
    #         )
    #         i += 1
    #     except Exception as e:
    #         print(f"Could not add marker for {name}: {e}")

    # v.user_scn.ngeom = i


    timestep = model.opt.timestep
    total_time = 3.0
    steps = int(total_time / timestep)


    # Actuator indices (assuming order is: x, y, z, q1, q2, q3)
    left_x, left_y, left_z = 0, 1, 2
    left_q1, left_q2, left_q3 = 3, 4, 5

    right_x, right_y, right_z = 6, 7, 8
    right_q1, right_q2, right_q3 = 9, 10, 11

    # # === Step 1: Lift right stick +0.03 ===
    for t in range(steps // 3):
        # data.ctrl[right_z] = 0.03
        data.ctrl[right_q2] = -0.003
        data.ctrl[left_q2] = -0.003  # lift left

        mujoco.mj_step(model, data)
        v.sync()

    print("lift left and right!")

    # # === Step 2: Hold right, lift left +0.015 ===
    # for t in range(steps // 3):
    #     # data.ctrl[right_z] = 0.03  # maintain right lift
    #     data.ctrl[left_q2] = -0.008  # lift left
    #     mujoco.mj_step(model, data)
    #     v.sync()

    # === Step 3: Funnel motion: move left inward along x ===
    target_x = -0.02  # example target inward position
    for t in range(steps // 3):
        data.ctrl[left_x] = -0.005
        data.ctrl[left_y] = -0.005

        mujoco.mj_step(model, data)
        v.sync()


    for step in range(10000):



        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)

    print("\nFinal left position:", data.xpos[left_bid])
    print("Final right position:", data.xpos[right_bid])



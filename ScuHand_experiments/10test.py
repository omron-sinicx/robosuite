import mujoco
from mujoco import viewer
import numpy as np
import time
from mujoco import mj_name2id, mj_id2name, mjtObj
from scipy.spatial.transform import Rotation as R
import itertools

print("Current Mujoco Version is", mujoco.__version__)

# Load model and data
model = mujoco.MjModel.from_xml_path("ctrls_circles_cloth.xml")
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
    time.sleep(10)
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


    for step in range(10000):
        # Move left in -x, -y
        # data.ctrl[LEFT_CTRL_START + 0] = -x_speed
        # data.ctrl[LEFT_CTRL_START + 1] = -y_speed
        data.ctrl[LEFT_CTRL_START + 2] = 0.025  # z

        # # # # Move right in +x, +y
        # data.ctrl[RIGHT_CTRL_START + 0] = +x_speed
        # data.ctrl[RIGHT_CTRL_START + 1] = +y_speed
        data.ctrl[RIGHT_CTRL_START + 2] = 0.025  # z

        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)

    print("\nFinal left position:", data.xpos[left_bid])
    print("Final right position:", data.xpos[right_bid])



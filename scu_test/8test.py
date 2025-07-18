import mujoco
from mujoco import viewer
import numpy as np
import time
from mujoco import mj_name2id, mj_id2name, mjtObj
from scipy.spatial.transform import Rotation as R
import itertools

print("Current Mujoco Version is", mujoco.__version__)

# Load model and data
model = mujoco.MjModel.from_xml_path("ctrl_circle_cloth.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Print flex info
print(f"Number of flexible vertices: {model.nflexvert}")
print(f"Number of flexible elements: {model.nflexelem}")

# print("\nInitial Cloth Vertex Positions:")
# for i in [0, 1, 33]:
#     print(f"Vertex {i}: {data.flexvert_xpos[i]}")


prefix = "cut_circle_"
max_check = 65  # Arbitrary upper limit, usually you won't need more than 300

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
#     print(f"cut_circle_{i}: xpos = {data.xpos[mj_name2id(model, mjtObj.mjOBJ_BODY, f'cut_circle_{i}')]} | flexvert = {data.flexvert_xpos[i]}")





















# Get spoon_base body ID
spoon_base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon_base")

# Print initial position
initial_pos = data.xpos[spoon_base_bid].copy()
print("\nInitial spoon_base position:", initial_pos)

# Indices for actuator control
x_idx, y_idx, z_idx = 0, 1, 2

# Set target z-position (lower)
z_target = 0.2
z_tolerance = 0.001  # Stop condition
z_speed = -0.0005    # Downward speed




with viewer.launch_passive(model, data) as v:
    time.sleep(1)

    v.opt.label = mujoco.mjtLabel.mjLABEL_BODY

    # Make sure this is run after v.sync() and v is a mujoco.viewer.Handle
    v.user_scn.ngeom = 0
    i = 0

    for j in range(65):
        name = f"cut_circle_{j}"
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            pos = data.xpos[bid]
            mujoco.mjv_initGeom(
                v.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.001, 0, 0],  # Small sphere
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1],  # Red
            )
            i += 1
        except Exception as e:
            print(f"Could not add marker for {name}: {e}")

    v.user_scn.ngeom = i




    for step in range(10000):
        current_z = data.xpos[spoon_base_bid][2]
        delta_z = z_target - current_z

        # print(current_z)



        if abs(delta_z) < z_tolerance:
            print(f"Target z reached: {current_z:.5f}")

            # break

        # data.ctrl[x_idx] = -0.05
        # data.ctrl[y_idx] = -0.05
        # data.ctrl[z_idx] = 0.25  # move down slightly

        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)

    print("Final spoon_base position:", data.xpos[spoon_base_bid])






        # # Add a 3x3x3 grid of variously colored spheres to the middle of the scene.
        # v.user_scn.ngeom = 0
        # i = 0
        # for x, y, z in itertools.product(*((range(-1, 2),) * 3)):
        #     mujoco.mjv_initGeom(
        #         v.user_scn.geoms[i],
        #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
        #         size=[0.02, 0, 0],
        #         pos=0.1*np.array([x, y, z]),
        #         mat=np.eye(3).flatten(),
        #         rgba=0.5*np.array([x + 1, y + 1, z + 1, 2])
        #     )
        #     i += 1
        # v.user_scn.ngeom = i
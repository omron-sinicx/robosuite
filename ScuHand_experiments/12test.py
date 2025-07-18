import mujoco
from mujoco import viewer
import numpy as np
import time
from mujoco import mj_name2id, mj_id2name, mjtObj
from scipy.spatial.transform import Rotation as R
import itertools

print("Current Mujoco Version is", mujoco.__version__)

# Load model and data
model = mujoco.MjModel.from_xml_path("ctrls_circles_cloth2.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)


### -------------------------------
### STEP 1. Get cloth center from c_0
### -------------------------------

center_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "c_0")
cloth_center = data.xpos[center_id][:2]
print("Detected cloth center from c_0:", cloth_center)


### -------------------------------
### Define IDs
### -------------------------------

left_indices = [1, 17, 33, 49, 65, 81]
left_names = [f"c_{i}" for i in left_indices]
left_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in left_names}

# right_indices = [16, 32, 48, 64, 80, 96]
# right_names = [f"c_{i}" for i in right_indices]
# right_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in right_names}

right_indices = [15, 31, 47, 63, 79, 95]
right_names = [f"c_{i}" for i in right_indices]
right_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in right_names}


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



    for step in range(100000):
        ### -------------------------------
        ### For each left-right pair
        ### -------------------------------

     # Actuator indices (assuming order is: x, y, z, q1, q2, q3)
        left_x, left_y, left_z = 0, 1, 2
        left_q1, left_q2, left_q3 = 3, 4, 5

        data.ctrl[left_q2] = -0.005  # lift left
        # data.ctrl[left_x] = -0.01
        time.sleep(0.01)

        # data.ctrl[left_z] = 0.0003
        # data.ctrl[left_y] = 0.01
        # data.ctrl[left_x] = -0.01

        left_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left")
        right_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right")

        target_pos = data.xpos[right_body_id]
        for step in range(500):
            current_pos = data.xpos[left_body_id]
            delta = target_pos - current_pos

            # Compute force direction and magnitude
            force_dir = delta[:3]
            force_mag = 10  # Tune force magnitude for desired speed
            if np.linalg.norm(force_dir) > 1e-4:
                force_dir = force_dir / np.linalg.norm(force_dir)
            force = 0.1*force_dir * force_mag

            # Apply external force to left body
            data.xfrc_applied[left_body_id][:3] = force



        ### -------------------------------
        ### Refresh
        ### -------------------------------
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)






























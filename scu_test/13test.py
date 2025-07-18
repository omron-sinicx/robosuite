import mujoco
from mujoco import viewer
import numpy as np
import time
from mujoco import mj_name2id, mj_id2name, mjtObj
from scipy.spatial.transform import Rotation as R
import itertools

print("Current Mujoco Version is", mujoco.__version__)

# Load model and data
model = mujoco.MjModel.from_xml_path("ctrls_circles_cloth3.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)


### -------------------------------
### STEP 1. Get cloth center from c_0
### -------------------------------

# Print flex info
print(f"Number of flexible vertices: {model.nflexvert}")
print(f"Number of flexible elements: {model.nflexelem}")

prefix = "c_"
max_check = 35  # Arbitrary upper limit, usually you won't need more than 300

for i in range(max_check):
    name = f"{prefix}{i}"
    try:
        body_id = mj_name2id(model, mjtObj.mjOBJ_BODY, name)
        pos = data.xpos[body_id]
        print(f"{name}: {pos}")
    except Exception:
        break  # Stops at the first missing name — efficient and safe

center_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "c_0")
cloth_center = data.xpos[center_id][:2]
print("Detected cloth center from c_0:", cloth_center)


with viewer.launch_passive(model, data) as v:
    time.sleep(6)
    v.opt.label = mujoco.mjtLabel.mjLABEL_BODY

    # Make sure this is run after v.sync() and v is a mujoco.viewer.Handle
    v.user_scn.ngeom = 0
    i = 0

    for j in range(100):
        name = f"c_{j}"
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            pos = data.xpos[bid]
            mujoco.mjv_initGeom(
                v.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.001, 0, 0],  # Small sphere
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 0.1],  # Red
            )
            i += 1
        except Exception as e:
            print(f"Could not add marker for {name}: {e}")

    v.user_scn.ngeom = i




    for step in range(1000000):

        # target_pos = np.array([-0.02903158, 0.14496864, 0.23230037])
        target_pos = np.array([-0.04584949 , 0.15360559, 0.22366343])

        max_step_size = 0.001  # adjust to 0.01 for faster motion
        c17_qpos_addr = model.joint('ctrl_17_joint').qposadr
        current_pos = np.array([data.qpos[c17_qpos_addr+ 0].item(), data.qpos[c17_qpos_addr+ 1].item(), data.qpos[c17_qpos_addr+ 2].item()])
    # Ensure current_pos is flat
        print("target pos:", target_pos)
        print("current_pos:", current_pos)

        # Simulation loop
        for i in range(1000):

            # Calculate difference
            diff = target_pos - current_pos

            print("Need to move", diff)

            # Check if close enough to target
            if np.linalg.norm(diff) < 1e-5:
                print("Target reached")
                break

            # Calculate step (clip per axis)
            step = np.clip(diff, -max_step_size, max_step_size)

            # Update position
            new_pos = current_pos + step
            data.qpos[c17_qpos_addr + 0] = new_pos[0]
            data.qpos[c17_qpos_addr + 1] = new_pos[1]
            data.qpos[c17_qpos_addr + 2] = new_pos[2]

            # Zero linear velocity for stability
            data.qvel[c17_qpos_addr + 0] = 0
            data.qvel[c17_qpos_addr + 1] = 0
            data.qvel[c17_qpos_addr + 2] = 0

            # Step simulation
            mujoco.mj_step(model, data)

            # Optional: print current position each step
            print(f"Step {i}, position: {new_pos}")

        # Final confirmation
        print("Movement finished")







        # # print(data.qpos[right_qpos_addr + 0])

        # # # Set position directly c_31
        # data.qpos[c17_qpos_addr + 0] = -0.02903158  
        # data.qpos[c17_qpos_addr + 1] = 0.14496864  
        # data.qpos[c17_qpos_addr + 2] = 0.23230037

        # # Optionally reset linear velocity to zero for stability
        # data.qvel[c17_qpos_addr + 0] = 0
        # data.qvel[c17_qpos_addr + 1] = 0
        # data.qvel[c17_qpos_addr + 2] = 0


        # # Set position directly c_30
        # data.qpos[c17_qpos_addr + 0] = -0.04584949  
        # data.qpos[c17_qpos_addr + 1] = 0.15360559    
        # data.qpos[c17_qpos_addr + 2] = 0.22366343

        # data.qvel[c17_qpos_addr + 0] = 0
        # data.qvel[c17_qpos_addr + 1] = 0
        # data.qvel[c17_qpos_addr + 2] = 0


        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)






























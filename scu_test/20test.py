import mujoco
from mujoco import viewer
import numpy as np
import time
from mujoco import mj_name2id, mj_id2name, mjtObj
from scipy.spatial.transform import Rotation as R
import itertools
import hashlib

def index_to_color(index):
    """
    Generate a consistent RGB color for a given index using hashing.
    Returns a list [r,g,b] with each in [0,1].
    """
    # Hash the index to get consistent bytes
    h = hashlib.md5(str(index).encode()).hexdigest()
    # Use first 6 hex digits for RGB
    r = int(h[0:2], 16) / 255
    g = int(h[2:4], 16) / 255
    b = int(h[4:6], 16) / 255
    return [r, g, b]

print("Current Mujoco Version is", mujoco.__version__)

# Load model and data
model = mujoco.MjModel.from_xml_path("ctrls_circles_cloth9.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)


### -------------------------------
### STEP 1. Get cloth center from c_0
### -------------------------------

# Print flex info
print(f"Number of flexible vertices: {model.nflexvert}")
print(f"Number of flexible elements: {model.nflexelem}")

for i in range(model.nbody):
    raw = model.names[model.name_bodyadr[i]:]
    name = raw.split(b'\x00',1)[0].decode('utf-8')  # decode bytes to str
    pos = data.xpos[i]
    print(f"{name}: {pos}")

center_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "c_0")
cloth_center = data.xpos[center_id][:2]
print("Detected cloth center from c_0:", cloth_center)



with viewer.launch_passive(model, data) as v:
    time.sleep(10)

    # model.opt.gravity[:] = 0

    # Get actuator ID
    scroll_stick_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'scroll_stick_joint')

    # Initialize geom index for trail visualization
    geom_index = 0

    # Prepare previous positions dict if needed
    previous_positions = {}

    r_flag=True

    # Main simulation loop
    for i in range(10000000):


        # Example actuator control
        if r_flag:
            data.ctrl[scroll_stick_actuator_id] = 100.0
        else:
            data.ctrl[scroll_stick_actuator_id] = 0.0


        # Pair definitions (choose your active set)

        # pairs = [(1, 23), (2, 24)] # 15 degree
        # pairs = [(1, 22), (2, 23), (3, 24)] # 30 degree
        pairs = [(1, 21), (2, 22), (3, 23), (4, 24)] # 45 degree
        # pairs = [(1,20), (2,21), (3,22), (4, 23), (5, 24)] # 60 degree
        # pairs = [(1,19), (2,20), (3,21), (4, 22), (5, 23), (6, 24)] # 75 degree
        # pairs = [(1,18), (2,19), (3,20), (4, 21), (5, 22), (6, 23), (7, 24)] # 90 degree
        # pairs = [(1,17), (2,18), (3,19), (4, 20), (5, 21), (6, 22), (7, 23), (8, 24)] # 105 degree
        # pairs = [(1,16), (2,17), (3,18), (4, 19), (5, 20), (6, 21), (7, 22), (8, 23), (9, 24)] # 120 degree
        # pairs = [(1,15), (2,16), (3,17), (4, 18), (5, 19), (6, 20), (7, 21), (8, 22), (9, 23), (10, 24)] # 135 degree
        # pairs = [(1,14), (2,15), (3,16), (4, 17), (5, 18), (6, 19), (7, 20), (8, 21), (9, 22), (10, 23), (11, 24)] # 150 degree
        # pairs = [(1,13), (2,14), (3,15), (4, 16), (5, 17), (6, 18), (7, 19), (8, 20), (9, 21), (10, 22), (11, 23), (12, 24)] # 165 degree
        # pairs = [(1,12), (2,13), (3,14), (4, 15), (5, 16), (6, 17), (7, 18), (8, 19), (9, 20), (10, 21), (11, 22), (12, 23), (13, 24)] # 180 degree

        max_step_size = 0.001
        all_reached = True

        for (i, j) in pairs:
            # Get qpos addresses using your syntax
            c_i_qpos_addr = model.joint(f'ctrl_{i}_joint').qposadr
            c_j_qpos_addr = model.joint(f'ctrl_{j}_joint').qposadr

            # Current positions
            c_i_pos = np.array([data.qpos[c_i_qpos_addr+0],
                                data.qpos[c_i_qpos_addr+1],
                                data.qpos[c_i_qpos_addr+2]])
            
            c_j_pos = np.array([data.qpos[c_j_qpos_addr+0],
                                data.qpos[c_j_qpos_addr+1],
                                data.qpos[c_j_qpos_addr+2]])

            # Compute difference
            diff = c_j_pos - c_i_pos

            # if np.linalg.norm(diff) >= 1e-5:
            if np.linalg.norm(diff) >= 9e-4:

                all_reached = False

                # Clip step
                step = np.clip(diff, -max_step_size, max_step_size)

                # Update c_i position
                data.qpos[c_i_qpos_addr+0] += step[0]
                data.qpos[c_i_qpos_addr+1] += step[1]
                data.qpos[c_i_qpos_addr+2] += step[2]

                # Zero linear velocity
                c_i_qvel_addr = model.joint(f'ctrl_{i}_joint').dofadr
                data.qvel[c_i_qvel_addr+0] = 0
                data.qvel[c_i_qvel_addr+1] = 0
                data.qvel[c_i_qvel_addr+2] = 0

                # mujoco.mj_step(model, data)  # Advance simulation step




            # === Visualization: Drop sphere for trajectory ===
            name = f"ctrl_{i}"
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            pos = data.xpos[bid]
            color = index_to_color(i) + [0.5]  # RGBA, semi-transparent

            if geom_index < len(v.user_scn.geoms):
                mujoco.mjv_initGeom(
                    v.user_scn.geoms[geom_index],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.0005, 0, 0],
                    pos=pos,
                    mat=np.eye(3).flatten(),
                    rgba=color,
                )
                geom_index += 1
                v.user_scn.ngeom = geom_index
            else:
                geom_index = 0  # Reset if out of slots

        # If all targets reached, break
        if all_reached:
            print("All targets reached")
            r_flag=False

            # break
            # continue

        # Advance simulation
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)































    # for step in range(100000):

    #     # Get actuator ID
    #     scroll_stick_actuator_id = mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, 'scroll_stick_joint')
    #     data.ctrl[scroll_stick_actuator_id] = 100.0  # rotate at 50 rad/s

    #     for step in range(1000000):



    #         weld_1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'lock_1')
    #         weld_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'lock_2')
    #         weld_3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'lock_3')


    #         # Simulation loop
    #         for i in range(1000):


    #             pairs = [(1,21), (2,22), (3,23), (4, 24)]
    #             # pairs = [(1,20), (2,21), (3,22), (4, 23), (5, 24)]
    #             # pairs = [(1,19), (2,20), (3,21), (4, 22), (5, 23), (6, 24)]
    #             # pairs = [(1,18), (2,19), (3,20), (4, 21), (5, 22), (6, 23), (7, 24)]
    #             # pairs = [(1,17), (2,18), (3,19), (4, 20), (5, 21), (6, 22), (7, 23), (8, 24)]
    #             # pairs = [(1,16), (2,17), (3,18), (4, 19), (5, 20), (6, 21), (7, 22), (8, 23), (9, 24)]
    #             # pairs = [(1,15), (2,16), (3,17), (4, 18), (5, 19), (6, 20), (7, 21), (8, 22), (9, 23), (10, 24)]
    #             # pairs = [(1,14), (2,15), (3,16), (4, 17), (5, 18), (6, 19), (7, 20), (8, 21), (9, 22), (10, 23), (11, 24)]
    #             # pairs = [(1,13), (2,14), (3,15), (4, 16), (5, 17), (6, 18), (7, 19), (8, 20), (9, 21), (10, 22), (11, 23), (12, 24)]
    #             # pairs = [(1,12), (2,13), (3,14), (4, 15), (5, 16), (6, 17), (7, 18), (8, 19), (9, 20), (10, 21), (11, 22), (12, 23), (13, 24)]


    #             max_step_size = 0.001

    #             all_reached = True

    #             for (i, j) in pairs:
    #                 # Get qpos addresses using your syntax
    #                 c_i_qpos_addr = model.joint(f'ctrl_{i}_joint').qposadr
    #                 c_j_qpos_addr = model.joint(f'ctrl_{j}_joint').qposadr

    #                 # Current positions
    #                 c_i_pos = np.array([data.qpos[c_i_qpos_addr+0],
    #                                     data.qpos[c_i_qpos_addr+1],
    #                                     data.qpos[c_i_qpos_addr+2]])
                    
    #                 c_j_pos = np.array([data.qpos[c_j_qpos_addr+0],
    #                                     data.qpos[c_j_qpos_addr+1],
    #                                     data.qpos[c_j_qpos_addr+2]])

    #                 # Compute difference
    #                 diff = c_j_pos - c_i_pos

    #                 # if np.linalg.norm(diff) >= 1e-5:
    #                 if np.linalg.norm(diff) >= 9e-4:

    #                     all_reached = False

    #                     # Clip step
    #                     step = np.clip(diff, -max_step_size, max_step_size)

    #                     # Update c_i position
    #                     data.qpos[c_i_qpos_addr+0] += step[0]
    #                     data.qpos[c_i_qpos_addr+1] += step[1]
    #                     data.qpos[c_i_qpos_addr+2] += step[2]

    #                     # Zero linear velocity
    #                     c_i_qvel_addr = model.joint(f'ctrl_{i}_joint').dofadr
    #                     data.qvel[c_i_qvel_addr+0] = 0
    #                     data.qvel[c_i_qvel_addr+1] = 0
    #                     data.qvel[c_i_qvel_addr+2] = 0

    #                     # mujoco.mj_step(model, data)  # Advance simulation step


    #             if all_reached:
    #                 # print("All targets reached")
    #                 # model.eq_active0[weld_1] = 1  # Lock at timestep 200
    #                 # model.eq_active0[weld_2] = 1  # Lock at timestep 200
    #                 # model.eq_active0[weld_3] = 1  # Lock at timestep 200

    #                 break

    #         # data.ctrl[scroll_stick_actuator_id] = 0.01  # rotate at 50 rad/s

    #             # mujoco.mj_step(model, data)
                

    #             mujoco.mj_forward(model, data)
    #             mujoco.mj_step(model, data)
    #             v.sync()
    #             time.sleep(0.01)



    #         mujoco.mj_forward(model, data)
    #         mujoco.mj_step(model, data)
    #         v.sync()
    #         time.sleep(0.01)

    #         # if all_reached:
    #             # print("haha")
                
    #             # break


    #         data.ctrl[scroll_stick_actuator_id] = 0.0

    #     mujoco.mj_forward(model, data)
    #     mujoco.mj_step(model, data)
    #     v.sync()
    #     time.sleep(0.01)




























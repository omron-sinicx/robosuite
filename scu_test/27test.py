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
model = mujoco.MjModel.from_xml_path("ctrls_circles_cloth15.xml")
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

    # model.opt.gravity[:] = 0

    # Main simulation loop
    for i in range(10000000):


        # Example actuator control
        if r_flag:
            data.ctrl[scroll_stick_actuator_id] = 100.0
        else:
            data.ctrl[scroll_stick_actuator_id] = 0.0


        # pairs = [
        #     (1, 19), (2, 20),
        #     (21, 39), (22, 40),
        #     (41, 59), (42, 60),
        #     (61, 79), (62, 80)
        # ] # 18 degree

        # pairs = [
        #     (1, 18), (2, 19), (3, 20),
        #     (21, 38), (22, 39), (23, 40),
        #     (41, 58), (42, 59), (43, 60),
        #     (61, 78), (62, 79), (63, 80)
        # ] # 36 degree

        # pairs = [
        #     (1, 17), (2, 18), (3, 19), (4, 20),
        #     (21, 37), (22, 38), (23, 39), (24, 40),
        #     (41, 57), (42, 58), (43, 59), (44, 60),
        #     (61, 77), (62, 78), (63, 79), (64, 80)
        # ] # 54 degree

        # pairs = [
        #     (1, 16), (2, 17), (3, 18), (4, 19), (5, 20),
        #     (21, 36), (22, 37), (23, 38), (24, 39), (25, 40),
        #     (41, 56), (42, 57), (43, 58), (44, 59), (45, 60),
        #     (61, 76), (62, 77), (63, 78), (64, 79), (65, 80)
        # ] # 72 degree

        # pairs = [
        #     (1, 15), (2, 16), (3, 17), (4, 18), (5, 19), (6, 20),
        #     (21, 35), (22, 36), (23, 37), (24, 38), (25, 39), (26, 40),
        #     (41, 55), (42, 56), (43, 57), (44, 58), (45, 59), (46, 60),
        #     (61, 75), (62, 76), (63, 77), (64, 78), (65, 79), (66, 80)
        # ] # 90 degree

        # pairs = [
        #     (1, 14), (2, 15), (3, 16), (4, 17), (5, 18), (6, 19), (7, 20),
        #     (21, 34), (22, 35), (23, 36), (24, 37), (25, 38), (26, 39), (27, 40),
        #     (41, 54), (42, 55), (43, 56), (44, 57), (45, 58), (46, 59), (47, 60),
        #     (61, 74), (62, 75), (63, 76), (64, 77), (65, 78), (66, 79), (67, 80)
        # ] # 108 degree

        # pairs = [
        #     (1, 13), (2, 14), (3, 15), (4, 16), (5, 17), (6, 18), (7, 19), (8, 20),
        #     (21, 33), (22, 34), (23, 35), (24, 36), (25, 37), (26, 38), (27, 39), (28, 40),
        #     (41, 53), (42, 54), (43, 55), (44, 56), (45, 57), (46, 58), (47, 59), (48, 60),
        #     (61, 73), (62, 74), (63, 75), (64, 76), (65, 77), (66, 78), (67, 79), (68, 80)
        # ] # 126 degree

        # pairs = [
        #     (1, 12), (2, 13), (3, 14), (4, 15), (5, 16), (6, 17), (7, 18), (8, 19), (9, 20),
        #     (21, 32), (22, 33), (23, 34), (24, 35), (25, 36), (26, 37), (27, 38), (28, 39), (29, 40),
        #     (41, 52), (42, 53), (43, 54), (44, 55), (45, 56), (46, 57), (47, 58), (48, 59), (49, 60),
        #     (61, 72), (62, 73), (63, 74), (64, 75), (65, 76), (66, 77), (67, 78), (68, 79), (69, 80)
        # ] # 144 degree

        # pairs = [
        #     (1, 11), (2, 12), (3, 13), (4, 14), (5, 15), (6, 16), (7, 17), (8, 18), (9, 19), (10, 20),
        #     (21, 31), (22, 32), (23, 33), (24, 34), (25, 35), (26, 36), (27, 37), (28, 38), (29, 39), (30, 40),
        #     (41, 51), (42, 52), (43, 53), (44, 54), (45, 55), (46, 56), (47, 57), (48, 58), (49, 59), (50, 60),
        #     (61, 71), (62, 72), (63, 73), (64, 74), (65, 75), (66, 76), (67, 77), (68, 78), (69, 79), (70, 80)
        # ] # 162 degree

        # pairs = [
        #     (1, 10), (2, 11), (3, 12), (4, 13), (5, 14), (6, 15), (7, 16), (8, 17), (9, 18), (10, 19), (11, 20),
        #     (21, 30), (22, 31), (23, 32), (24, 33), (25, 34), (26, 35), (27, 36), (28, 37), (29, 38), (30, 39), (31, 40),
        #     (41, 50), (42, 51), (43, 52), (44, 53), (45, 54), (46, 55), (47, 56), (48, 57), (49, 58), (50, 59), (51, 60),
        #     (61, 70), (62, 71), (63, 72), (64, 73), (65, 74), (66, 75), (67, 76), (68, 77), (69, 78), (70, 79), (71, 80)
        # ] # 180 degree


        all_pairs = [

            # 18 degree
            [
                (1, 19), (2, 20),
                (21, 39), (22, 40),
                (41, 59), (42, 60),
                (61, 79), (62, 80)
            ],

            # 36 degree
            [
                (1, 18), (2, 19), (3, 20),
                (21, 38), (22, 39), (23, 40),
                (41, 58), (42, 59), (43, 60),
                (61, 78), (62, 79), (63, 80)
            ],

            # 54 degree
            [
                (1, 17), (2, 18), (3, 19), (4, 20),
                (21, 37), (22, 38), (23, 39), (24, 40),
                (41, 57), (42, 58), (43, 59), (44, 60),
                (61, 77), (62, 78), (63, 79), (64, 80)
            ],

            # 72 degree
            [
                (1, 16), (2, 17), (3, 18), (4, 19), (5, 20),
                (21, 36), (22, 37), (23, 38), (24, 39), (25, 40),
                (41, 56), (42, 57), (43, 58), (44, 59), (45, 60),
                (61, 76), (62, 77), (63, 78), (64, 79), (65, 80)
            ],

            # 90 degree
            [
                (1, 15), (2, 16), (3, 17), (4, 18), (5, 19), (6, 20),
                (21, 35), (22, 36), (23, 37), (24, 38), (25, 39), (26, 40),
                (41, 55), (42, 56), (43, 57), (44, 58), (45, 59), (46, 60),
                (61, 75), (62, 76), (63, 77), (64, 78), (65, 79), (66, 80)
            ],

            # 108 degree
            [
                (1, 14), (2, 15), (3, 16), (4, 17), (5, 18), (6, 19), (7, 20),
                (21, 34), (22, 35), (23, 36), (24, 37), (25, 38), (26, 39), (27, 40),
                (41, 54), (42, 55), (43, 56), (44, 57), (45, 58), (46, 59), (47, 60),
                (61, 74), (62, 75), (63, 76), (64, 77), (65, 78), (66, 79), (67, 80)
            ],

            # 126 degree
            [
                (1, 13), (2, 14), (3, 15), (4, 16), (5, 17), (6, 18), (7, 19), (8, 20),
                (21, 33), (22, 34), (23, 35), (24, 36), (25, 37), (26, 38), (27, 39), (28, 40),
                (41, 53), (42, 54), (43, 55), (44, 56), (45, 57), (46, 58), (47, 59), (48, 60),
                (61, 73), (62, 74), (63, 75), (64, 76), (65, 77), (66, 78), (67, 79), (68, 80)
            ],

            # 144 degree
            [
                (1, 12), (2, 13), (3, 14), (4, 15), (5, 16), (6, 17), (7, 18), (8, 19), (9, 20),
                (21, 32), (22, 33), (23, 34), (24, 35), (25, 36), (26, 37), (27, 38), (28, 39), (29, 40),
                (41, 52), (42, 53), (43, 54), (44, 55), (45, 56), (46, 57), (47, 58), (48, 59), (49, 60),
                (61, 72), (62, 73), (63, 74), (64, 75), (65, 76), (66, 77), (67, 78), (68, 79), (69, 80)
            ],

            # 162 degree
            [
                (1, 11), (2, 12), (3, 13), (4, 14), (5, 15), (6, 16), (7, 17), (8, 18), (9, 19), (10, 20),
                (21, 31), (22, 32), (23, 33), (24, 34), (25, 35), (26, 36), (27, 37), (28, 38), (29, 39), (30, 40),
                (41, 51), (42, 52), (43, 53), (44, 54), (45, 55), (46, 56), (47, 57), (48, 58), (49, 59), (50, 60),
                (61, 71), (62, 72), (63, 73), (64, 74), (65, 75), (66, 76), (67, 77), (68, 78), (69, 79), (70, 80)
            ],

            # 180 degree
            [
                (1, 10), (2, 11), (3, 12), (4, 13), (5, 14), (6, 15), (7, 16), (8, 17), (9, 18), (10, 19), (11, 20),
                (21, 30), (22, 31), (23, 32), (24, 33), (25, 34), (26, 35), (27, 36), (28, 37), (29, 38), (30, 39), (31, 40),
                (41, 50), (42, 51), (43, 52), (44, 53), (45, 54), (46, 55), (47, 56), (48, 57), (49, 58), (50, 59), (51, 60),
                (61, 70), (62, 71), (63, 72), (64, 73), (65, 74), (66, 75), (67, 76), (68, 77), (69, 78), (70, 79), (71, 80)
            ],

        ]




        # Choose your target step index, e.g. 2 for 54 degree
        target_index = 8

        # Iterate through each intermediate step up to and including the target
        for step_pairs in all_pairs[:target_index+1]:

            max_step_size = 0.001
            all_reached = False

            while not all_reached:
                all_reached = True

                for (i, j) in step_pairs:
                    c_i_qpos_addr = model.joint(f'ctrl_{i}_joint').qposadr
                    c_j_qpos_addr = model.joint(f'ctrl_{j}_joint').qposadr

                    c_i_pos = np.array([data.qpos[c_i_qpos_addr+0],
                                        data.qpos[c_i_qpos_addr+1],
                                        data.qpos[c_i_qpos_addr+2]])
                    c_j_pos = np.array([data.qpos[c_j_qpos_addr+0],
                                        data.qpos[c_j_qpos_addr+1],
                                        data.qpos[c_j_qpos_addr+2]])

                    diff = c_j_pos - c_i_pos

                    if np.linalg.norm(diff) >= 9e-4:
                        all_reached = False

                        step = np.clip(diff, -max_step_size, max_step_size)

                        data.qpos[c_i_qpos_addr+0] += step[0]
                        data.qpos[c_i_qpos_addr+1] += step[1]
                        data.qpos[c_i_qpos_addr+2] += step[2]

                        c_i_qvel_addr = model.joint(f'ctrl_{i}_joint').dofadr
                        data.qvel[c_i_qvel_addr+0] = 0
                        data.qvel[c_i_qvel_addr+1] = 0
                        data.qvel[c_i_qvel_addr+2] = 0

                    # === Visualization ===
                    name = f"ctrl_{i}"
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    pos = data.xpos[bid]
                    color = index_to_color(i) + [0.5]

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
                        geom_index = 0

                mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data)
                v.sync()
                time.sleep(0.01)

            print(f"Step {step_pairs} reached")
































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




























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
model = mujoco.MjModel.from_xml_path("ctrls_circles_cloth16.xml")
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


weld_ids = {}

for i in range(73):
    lock_name = f'lock_{i}'
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, lock_name)
    weld_ids[lock_name] = weld_id


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

        # 0-8
        all_pairs = [

            # 20 degree
            [
                (1, 17), (2, 18),
                (19, 35), (20, 36),
                (37, 53), (38, 54),
                (55, 71), (56, 72)
            ],

            # 40 degree
            [
                (1, 16), (2, 17), (3, 18),
                (19, 34), (20, 35), (21, 36),
                (37, 52), (38, 53), (39, 54),
                (55, 70), (56, 71), (57, 72)
            ],

            # 60 degree
            [
                (1, 15), (2, 16), (3, 17), (4, 18),
                (19, 33), (20, 34), (21, 35), (22, 36),
                (37, 51), (38, 52), (39, 53), (40, 54),
                (55, 69), (56, 70), (57, 71), (58, 72)
            ],

            # 80 degree
            [
                (1, 14), (2, 15), (3, 16), (4, 17), (5, 18),
                (19, 32), (20, 33), (21, 34), (22, 35), (23, 36),
                (37, 50), (38, 51), (39, 52), (40, 53), (41, 54),
                (55, 68), (56, 69), (57, 70), (58, 71), (59, 72)
            ],

            # 100 degree
            [
                (1, 13), (2, 14), (3, 15), (4, 16), (5, 17), (6, 18),
                (19, 31), (20, 32), (21, 33), (22, 34), (23, 35), (24, 36),
                (37, 49), (38, 50), (39, 51), (40, 52), (41, 53), (42, 54),
                (55, 67), (56, 68), (57, 69), (58, 70), (59, 71), (60, 72)
            ],

            # 120 degree
            [
                (1, 12), (2, 13), (3, 14), (4, 15), (5, 16), (6, 17), (7, 18),
                (19, 30), (20, 31), (21, 32), (22, 33), (23, 34), (24, 35), (25, 36),
                (37, 48), (38, 49), (39, 50), (40, 51), (41, 52), (42, 53), (43, 54),
                (55, 66), (56, 67), (57, 68), (58, 69), (59, 70), (60, 71), (61, 72)
            ],

            # 140 degree
            [
                (1, 11), (2, 12), (3, 13), (4, 14), (5, 15), (6, 16), (7, 17), (8, 18),
                (19, 29), (20, 30), (21, 31), (22, 32), (23, 33), (24, 34), (25, 35), (26, 36),
                (37, 47), (38, 48), (39, 49), (40, 50), (41, 51), (42, 52), (43, 53), (44, 54),
                (55, 65), (56, 66), (57, 67), (58, 68), (59, 69), (60, 70), (61, 71), (62, 72)
            ],

            # 160 degree
            [
                (1, 10), (2, 11), (3, 12), (4, 13), (5, 14), (6, 15), (7, 16), (8, 17), (9, 18),
                (19, 28), (20, 29), (21, 30), (22, 31), (23, 32), (24, 33), (25, 34), (26, 35), (27, 36),
                (37, 46), (38, 47), (39, 48), (40, 49), (41, 50), (42, 51), (43, 52), (44, 53), (45, 54),
                (55, 64), (56, 65), (57, 66), (58, 67), (59, 68), (60, 69), (61, 70), (62, 71), (63, 72)
            ],

            # 180 degree
            [
                (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16), (9, 17), (10, 18),
                (19, 27), (20, 28), (21, 29), (22, 30), (23, 31), (24, 32), (25, 33), (26, 34), (27, 35), (28, 36),
                (37, 45), (38, 46), (39, 47), (40, 48), (41, 49), (42, 50), (43, 51), (44, 52), (45, 53), (46, 54),
                (55, 63), (56, 64), (57, 65), (58, 66), (59, 67), (60, 68), (61, 69), (62, 70), (63, 71), (64, 72)
            ],

            # # 180 degree
            # [
            #     (1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13), (7, 14), (8, 15), (9, 16), (10, 17), (11, 18),
            #     (19, 26), (20, 27), (21, 28), (22, 29), (23, 30), (24, 31), (25, 32), (26, 33), (27, 34), (28, 35), (29, 36),
            #     (37, 44), (38, 45), (39, 46), (40, 47), (41, 48), (42, 49), (43, 50), (44, 51), (45, 52), (46, 53), (47, 54),
            #     (55, 62), (56, 63), (57, 64), (58, 65), (59, 66), (60, 67), (61, 68), (62, 69), (63, 70), (64, 71), (65, 72)
            # ]

        ]

        # Choose your target step index, e.g. 2 for 54 degree
        # # target_index = 1 #40 d
        # # target_index = 3 # 80 d
        # target_index = 5 # 120 d
        target_index = 6 # 140 d
        # target_index = 7 # 160 d
        # target_index = 8 # 180 d


        # if r_flag:
        #     aaa = all_pairs[:target_index+1]
        # else:
        #     aaa = all_pairs[target_index:target_index+1]


        half_len = len(all_pairs) / 2
        num_select = int(half_len) 

        if target_index < half_len:
            if r_flag:
                aaa = all_pairs[:target_index+1]
            else:
                aaa = all_pairs[target_index:target_index+1]
            print(f"Using full steps up to target_index={target_index}, selected indices: {list(range(target_index+1))}")
        else:
            if r_flag:
                # Determine evenly spaced indices between 0 and target_index
                indices = np.linspace(0, target_index, num_select, dtype=int).tolist()

                # Ensure first is 0 and last is target_index
                indices[0] = 0
                indices[-1] = target_index

                # Remove duplicates while keeping order
                seen = set()
                indices = [x for x in indices if not (x in seen or seen.add(x))]

                aaa = [all_pairs[i] for i in indices]
                print(f"Using reduced steps for target_index={target_index}, selected indices: {indices}")
            else:
                aaa = all_pairs[target_index:target_index+1]
                print(f"Selected indices: ", aaa)

        
        # Iterate through each intermediate step up to and including the target
        # for step_pairs in all_pairs[:target_index+1]:
        for step_pairs in aaa:

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


                    # First check the last pair in the last block
                    last_block = all_pairs[target_index]
                    i_last, j_last = last_block[-1]  # extract the last pair

                    lc_i_qpos_addr = model.joint(f'ctrl_{i_last}_joint').qposadr
                    lc_j_qpos_addr = model.joint(f'ctrl_{j_last}_joint').qposadr

                    lc_i_pos = np.array([
                        data.qpos[lc_i_qpos_addr + 0],
                        data.qpos[lc_i_qpos_addr + 1],
                        data.qpos[lc_i_qpos_addr + 2]
                    ])
                    lc_j_pos = np.array([
                        data.qpos[lc_j_qpos_addr + 0],
                        data.qpos[lc_j_qpos_addr + 1],
                        data.qpos[lc_j_qpos_addr + 2]
                    ])

                    diff_last = lc_j_pos - lc_i_pos


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


                # # If all targets reached, break
                # if all_reached:
                #     r_flag=False
                #     break

                mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data)
                v.sync()
                time.sleep(0.01)

            print(f"Step {step_pairs} reached")

            if np.linalg.norm(diff_last) < 9e-4:
                r_flag=False
                # continue

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)
        # for i in range(73):
        #     lock_name = f'lock_{i}'
        #     weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, lock_name)
        #     model.eq_active0[weld_id] = 1  # activate this weld
        #     # print("fix", i)
        # r_flag=False

































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




























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
model = mujoco.MjModel.from_xml_path("ctrls_circles_cloth13.xml")
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


        all_pairs = [

            # 18 degree
            [
                (1, 15), (2, 16),
                (17, 31), (18, 32),
                (33, 47), (34, 48),
            ],

            # 36 degree
            [
                (1, 14), (2, 15), (3, 16),
                (17, 30), (18, 31), (19, 32),
                (33, 46), (34, 47), (35, 48),
            ],

            # 54 degree
            [
                (1, 13), (2, 14), (3, 15), (4, 16),
                (17, 29), (18, 30), (19, 31), (20, 32),
                (33, 45), (34, 46), (35, 47), (36, 48),
            ],

            # 72 degree
            [
                (1, 12), (2, 13), (3, 14), (4, 15), (5, 16),
                (17, 28), (18, 29), (19, 30), (20, 31), (21, 32),
                (33, 44), (34, 45), (35, 46), (36, 47), (37, 48),
            ],

            # 90 degree
            [
                (1, 11), (2, 12), (3, 13), (4, 14), (5, 15), (6, 16),
                (17, 27), (18, 28), (19, 29), (20, 30), (21, 31), (22, 32),
                (33, 43), (34, 44), (35, 45), (36, 46), (37, 47), (38, 48),
            ],

            # 108 degree
            [
                (1, 10), (2, 11), (3, 12), (4, 13), (5, 14), (6, 15), (7, 16),
                (17, 26), (18, 27), (19, 28), (20, 29), (21, 30), (22, 31), (23, 32),
                (33, 42), (34, 43), (35, 44), (36, 45), (37, 46), (38, 47), (39, 48),
            ],

            # 126 degree
            [
                (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16),
                (17, 25), (18, 26), (19, 27), (20, 28), (21, 29), (22, 30), (23, 31), (24, 32),
                (33, 41), (34, 42), (35, 43), (36, 44), (37, 45), (38, 46), (39, 47), (40, 48),
            ],

            # 144 degree
            [
                (1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13), (7, 14), (8, 15), (9, 16),
                (17, 24), (18, 25), (19, 26), (20, 27), (21, 28), (22, 29), (23, 30), (24, 31), (25, 32),
                (33, 40), (34, 41), (35, 42), (36, 43), (37, 44), (38, 45), (39, 46), (40, 47), (41, 48),
            ],

            # 162 degree
            [
                (1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12), (7, 13), (8, 14), (9, 15), (10, 16),
                (17, 23), (18, 24), (19, 25), (20, 26), (21, 27), (22, 28), (23, 29), (24, 30), (25, 31), (26, 32),
                (33, 39), (34, 40), (35, 41), (36, 42), (37, 43), (38, 44), (39, 45), (40, 46), (41, 47), (42, 48),
            ],

            # 180 degree
            [
                (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15), (11, 16),
                (17, 22), (18, 23), (19, 24), (20, 25), (21, 26), (22, 27), (23, 28), (24, 29), (25, 30), (26, 31), (27, 32),
                (33, 38), (34, 39), (35, 40), (36, 41), (37, 42), (38, 43), (39, 44), (40, 45), (41, 46), (42, 47), (43, 48),
            ]

        ]

                # Choose your target step index, e.g. 2 for 54 degree
                
        target_index = 5

        # Iterate through each intermediate step up to and including the target
        for step_pairs in all_pairs[:target_index+1]:

            max_step_size = 0.0025
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

                    # # === Visualization ===
                    # name = f"ctrl_{i}"
                    # bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    # pos = data.xpos[bid]
                    # color = index_to_color(i) + [0.5]

                    # if geom_index < len(v.user_scn.geoms):
                    #     mujoco.mjv_initGeom(
                    #         v.user_scn.geoms[geom_index],
                    #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    #         size=[0.0005, 0, 0],
                    #         pos=pos,
                    #         mat=np.eye(3).flatten(),
                    #         rgba=color,
                    #     )
                    #     geom_index += 1
                    #     v.user_scn.ngeom = geom_index
                    # else:
                    #     geom_index = 0

                mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data)
                v.sync()
                time.sleep(0.01)

            print(f"Step {step_pairs} reached")


        # for i in range(73):
        #     lock_name = f'lock_{i}'
        #     weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, lock_name)
        #     model.eq_active0[weld_id] = 1  # activate this weld
        #     # print("fix", i)
        # r_flag=False

























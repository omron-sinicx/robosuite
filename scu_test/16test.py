import mujoco
from mujoco import viewer
import numpy as np
import time
from mujoco import mj_name2id, mj_id2name, mjtObj
from scipy.spatial.transform import Rotation as R
import itertools

print("Current Mujoco Version is", mujoco.__version__)

# Load model and data
model = mujoco.MjModel.from_xml_path("ctrls_circles_cloth6.xml")
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
    time.sleep(6)
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


    # Turn off gravity
    # model.opt.gravity[:] = 0


    for step in range(1000000):


        # target_pos15 = np.array([-0.01886579, 0.16111882, 0.21615018]) #15
        # target_pos14 = np.array([-0.02727475, 0.16543729, 0.21183172]) #14
        # target_pos30 = np.array([-0.04584949 , 0.15360559, 0.22366343]) #30
        # target_pos31 = np.array([-0.02903158, 0.14496864, 0.23230037]) #31


        c13_qpos_addr = model.joint('ctrl_13_joint').qposadr
        c13_current_pos = np.array([data.qpos[c13_qpos_addr+ 0].item(), data.qpos[c13_qpos_addr+ 1].item(), data.qpos[c13_qpos_addr+ 2].item()])

        c14_qpos_addr = model.joint('ctrl_14_joint').qposadr
        c14_current_pos = np.array([data.qpos[c14_qpos_addr+ 0].item(), data.qpos[c14_qpos_addr+ 1].item(), data.qpos[c14_qpos_addr+ 2].item()])
        
        c9_qpos_addr = model.joint('ctrl_9_joint').qposadr
        c9_current_pos = np.array([data.qpos[c9_qpos_addr+ 0].item(), data.qpos[c9_qpos_addr+ 1].item(), data.qpos[c9_qpos_addr+ 2].item()])

        c10_qpos_addr = model.joint('ctrl_10_joint').qposadr
        c10_current_pos = np.array([data.qpos[c10_qpos_addr+ 0].item(), data.qpos[c10_qpos_addr+ 1].item(), data.qpos[c10_qpos_addr+ 2].item()])

        c1_qpos_addr = model.joint('ctrl_1_joint').qposadr
        c1_current_pos = np.array([data.qpos[c1_qpos_addr+ 0].item(), data.qpos[c1_qpos_addr+ 1].item(), data.qpos[c1_qpos_addr+ 2].item()])


        max_step_size = 0.001  # adjust to 0.01 for faster motion

        c2_qpos_addr = model.joint('ctrl_2_joint').qposadr
        c2_current_pos = np.array([data.qpos[c2_qpos_addr+ 0].item(), data.qpos[c2_qpos_addr+ 1].item(), data.qpos[c2_qpos_addr+ 2].item()])

        c15_qpos_addr = model.joint('ctrl_15_joint').qposadr
        c15_current_pos = np.array([data.qpos[c15_qpos_addr+ 0].item(), data.qpos[c15_qpos_addr+ 1].item(), data.qpos[c15_qpos_addr+ 2].item()])
        
        # if step < 500:
        #     target_pos30 = np.array([c17_current_pos[0], c17_current_pos[1]+0.02, c17_current_pos[2]]) #14
        # else:
        #     print("start")
        #     target_pos30 = np.array([-0.04584949 , 0.15360559, 0.22366343]) #30


        # Get actuator ID
        scroll_stick_actuator_id = mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, 'scroll_stick_joint')

        weld_1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'lock_1')
        weld_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'lock_2')
        weld_3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'lock_3')

        data.ctrl[scroll_stick_actuator_id] = 100.0  # rotate at 50 rad/s


        # Simulation loop
        for i in range(100000):


            # Calculate difference
            # diff = c14_current_pos - c1_current_pos
            # diff = c14_current_pos - c1_current_pos

            # if np.linalg.norm(diff) < 1e-5:
            #     print("Target reached")
            #     break

            # step = np.clip(diff, -max_step_size, max_step_size)

            # # Update position
            # new_pos = c1_current_pos + step
            # data.qpos[c1_qpos_addr + 0] = new_pos[0]
            # data.qpos[c1_qpos_addr + 1] = new_pos[1]
            # data.qpos[c1_qpos_addr + 2] = new_pos[2]

            # # Zero linear velocity for stability [MUST]
            # data.qvel[c1_qpos_addr + 0] = 0
            # data.qvel[c1_qpos_addr + 1] = 0
            # data.qvel[c1_qpos_addr + 2] = 0


            pairs = [(81,92), (82,93), (83,94), (84, 95), (65,76), (66,77), (67,78), (68, 79)]
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

                if np.linalg.norm(diff) >= 1e-5:
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

            if all_reached:
                print("All targets reached")
                # model.eq_active0[weld_1] = 1  # Lock at timestep 200
                # model.eq_active0[weld_2] = 1  # Lock at timestep 200
                # model.eq_active0[weld_3] = 1  # Lock at timestep 200

                break

        # data.ctrl[scroll_stick_actuator_id] = 0.01  # rotate at 50 rad/s

            # mujoco.mj_step(model, data)  # Advance simulation step







            # if i == 900:
            #     model.eq_active0[weld_id] = 1  # Lock at timestep 200

        #     # Calculate difference
        #     diff = c31_current_pos - c17_current_pos

        #     if np.linalg.norm(diff) < 1e-5:
        #         print("Target reached")
        #         break

        #     step = np.clip(diff, -max_step_size, max_step_size)

        #     # Update position
        #     new_pos = c17_current_pos + step
        #     data.qpos[c17_qpos_addr + 0] = new_pos[0]
        #     data.qpos[c17_qpos_addr + 1] = new_pos[1]
        #     data.qpos[c17_qpos_addr + 2] = new_pos[2]

        #     # Zero linear velocity for stability [MUST]
        #     data.qvel[c17_qpos_addr + 0] = 0
        #     data.qvel[c17_qpos_addr + 1] = 0
        #     data.qvel[c17_qpos_addr + 2] = 0








            # if i == 999:
            #     data.ctrl[scroll_stick_actuator_id] = 0.0  # rotate at 50 rad/s


            # mujoco.mj_step(model, data)
            
        # data.ctrl[scroll_stick_actuator_id] = 0.0

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)































import mujoco
import mujoco.viewer
import numpy as np
import time

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path("circle_cloth.xml")
data = mujoco.MjData(model)

# # Get joint ID using .id
# joint1_id = model.joint("handle_left").id
# joint2_id = model.joint("handle_right").id

# # Get the starting index in qpos for each free joint (7 values: xyz + quat)
# idx1 = model.jnt_qposadr[joint1_id]
# idx2 = model.jnt_qposadr[joint2_id]

# # Spiral motion parameters
# t = 0.0
# dt = 0.01
# r0 = 1.0
# z0 = 0.5
# shrink_rate = 0.3
# drop_rate = 0.1
# rotation_speed = 3.0

# Run viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(5)
    while viewer.is_running():
        # t += dt

        # # Spiral motion
        # r = max(0.1, r0 - shrink_rate * t)
        # angle = rotation_speed * t
        # z = max(0.2, z0 - drop_rate * t)
        # x = r * np.cos(angle)
        # y = r * np.sin(angle)

        # # Set position (xyz) of free joint
        # data.qpos[idx1:idx1+3] = [x, y, z]
        # data.qpos[idx2:idx2+3] = [x, y, z]

        # # Set default orientation (unit quaternion)
        # data.qpos[idx1+3:idx1+7] = [1, 0, 0, 0]
        # data.qpos[idx2+3:idx2+7] = [1, 0, 0, 0]

        mujoco.mj_step(model, data)
        viewer.sync()

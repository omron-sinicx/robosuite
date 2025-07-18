import mujoco
from mujoco import viewer
import numpy as np
import time
from mujoco import mj_name2id, mj_id2name, mjtObj
from scipy.spatial.transform import Rotation as R

# Load model and data
print(mujoco.__version__)

model = mujoco.MjModel.from_xml_path("circle_2_cloth_scene.xml") #./env/mujoco_templates/flex/.xml
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)



with viewer.launch_passive(model, data) as v:
    time.sleep(2)

    for step in range(10000):


        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)




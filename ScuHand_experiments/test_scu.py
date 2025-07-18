import mujoco
from mujoco import viewer
import time
import os
import numpy as np
from template_renderer import TemplateRenderer

# Step 1: Setup template rendering
template_path = "env/mujoco_templates/arena.xml"
output_xml = "rendered_cloth.xml"
template_dir = "./"

renderer = TemplateRenderer(template_dir)

rendered_xml = renderer.render_template(
    template_path,
    timestep=0.002,
    impratio=1.0,
    cloth_texture_r_2=0.7,
    cloth_texture_g_2=0.7,
    cloth_texture_b_2=0.7,
    cloth_texture_a=0.9,
    cloth_texture_builtin="checker",
    cloth_texture_mark="edge", # only edge
    cloth_material_name="bath_real_material", # wipe_real_material, bath_real_material, bath_2_real_material, kitchen_real_material, kitchen_2_real_material, wipe_real_material, wipe_2_real_material, cloth_material, white_real_material, blue_real_material, orange_real_material
    table_material_name="table_material", # table_real_material, table_material
    cone_type="elliptic",
    cloth_texture_width=256.0,
    cloth_texture_height=256.0,
    cloth_texture_repeat_x=1.0,
    cloth_texture_repeat_y=1.0,
    cloth_texture_random=0.0,
    cloth_texture_r_1=0.5,
    cloth_texture_g_1=0.5,
    cloth_texture_b_1=0.5,
    floor_material_name="floor_real_material", #floor_material, floor_real_material
    table_texture_width=128.0,
    table_texture_height=128.0,
    table_texture_repeat_x=1.0,
    table_texture_repeat_y=1.0,
    table_texture_random=0.0,
    table_texture_r_1=0.1,
    table_texture_g_1=0.1,
    table_texture_b_1=0.1,
    table_texture_r_2=0.1,
    table_texture_g_2=0.1,
    table_texture_b_2=0.1,
    table_texture_builtin="flat",
    table_texture_mark="edge",
    floor_texture_width=128.0,
    floor_texture_height=128.0,
    floor_texture_repeat_x=1.0,
    floor_texture_repeat_y=1.0,
    floor_texture_random=0.0,
    floor_texture_r_1=0.3,
    floor_texture_g_1=0.3,
    floor_texture_b_1=0.3,
    floor_texture_r_2=0.4,
    floor_texture_g_2=0.4,
    floor_texture_b_2=0.4,
    floor_texture_builtin="flat",
    floor_texture_mark="edge",
    robot1_texture_width=64.0,
    robot1_texture_height=64.0,
    robot1_texture_repeat_x=1.0,
    robot1_texture_repeat_y=1.0,
    robot1_texture_random=0.0,
    robot1_texture_r_1=0.5,
    robot1_texture_g_1=0.5,
    robot1_texture_b_1=0.5,
    robot1_texture_r_2=0.7,
    robot1_texture_g_2=0.7,
    robot1_texture_b_2=0.7,
    robot1_texture_builtin="flat",
    robot1_texture_mark="edge",
    robot2_texture_width=64.0,
    robot2_texture_height=64.0,
    robot2_texture_repeat_x=1.0,
    robot2_texture_repeat_y=1.0,
    robot2_texture_random=0.0,
    robot2_texture_r_1=0.5,
    robot2_texture_g_1=0.5,
    robot2_texture_b_1=0.5,
    robot2_texture_r_2=0.7,
    robot2_texture_g_2=0.7,
    robot2_texture_b_2=0.7,
    robot2_texture_builtin="flat",
    robot2_texture_mark="edge",
    geom_spacing=0.025, # modify the size of cloth
    geom_size=0.005,
    offset=0.01,
    friction=0.8,
    geom_solimp_low=0.9,
    geom_solimp_high=0.95,
    geom_solimp_width=0.001,
    geom_solref_timeconst=0.01,
    geom_solref_dampratio=1.0,
    joint_solimp_low=0.8,
    joint_solimp_high=0.9,
    joint_solimp_width=0.001,
    joint_solref_timeconst=0.01,
    joint_solref_dampratio=1.0,
    tendon_shear_solimp_low=0.6,
    tendon_shear_solimp_high=0.9,
    tendon_shear_solimp_width=0.001,
    tendon_shear_solref_timeconst=0.01,
    tendon_shear_solref_dampratio=1.0,
    tendon_main_solimp_low=0.6,
    tendon_main_solimp_high=0.9,
    tendon_main_solimp_width=0.001,
    tendon_main_solref_timeconst=0.01,
    tendon_main_solref_dampratio=1.0,
    train_camera_fovy=45.0,
    lights_randomization=False,
    num_lights=2,
    materials_randomization=False,
)

# Step 3: Save to file
with open(output_xml, "w") as f:
    f.write(rendered_xml)

# Step 4: Load and view the scene
model = mujoco.MjModel.from_xml_path(output_xml)
data = mujoco.MjData(model)



# Set initial joint positions for robot to stand upright (adjust names/values as needed)
joint_stand_poses = {
    "joint1": 0.0,
    "joint2": -0.785,
    "joint3": 1.0,
    "joint4": -2.356,
    "joint5": 0.0,
    "joint6": 1.571,
    "joint7": 0.785,

}



with viewer.launch_passive(model, data) as v:
    t0 = time.time()
    while v.is_running():
        elapsed = time.time() - t0



        if int(elapsed * 10) % 10 == 0:
            for i in range(model.nv):
                joint_id = model.dof_jntid[i]
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

                if joint_name.startswith("J"):
                    # data.qpos[i] = 0.01
                    print(i)
                    # data.qvel[0] = -3

                    # data.qvel[i] += 0.02 * np.random.randn()

                if joint_name in joint_stand_poses:
                    data.qpos[joint_id] = joint_stand_poses[joint_name]
                    data.qvel[joint_id] = 0.0

        mujoco.mj_step(model, data)
        v.sync()

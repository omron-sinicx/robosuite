# FDCC vs OSC_POSITION Controller Config Comparison

| Parameter                | OSC_POSITION                                 | FDCC (Top-level)                                   |
|--------------------------|----------------------------------------------|----------------------------------------------------|
| type                     | OSC_POSITION                                 | FDCC                                               |
| input_max                | 1                                            | 1                                                  |
| input_min                | -1                                           | -1                                                 |
| output_max               | [0.0022, 0.0022, 0.0022]                     | [0.01, 0.01, 0.01, 0.1, 0.1, 0.1]                  |
| output_min               | [-0.0022, -0.0022, -0.0022]                  | [-0.01, -0.01, -0.01, -0.1, -0.1, -0.1]            |
| kp                       | 500                                          | [0.05, 0.05, 0.05, 0.1, 0.1:
, 0.1]                  |
| damping_ratio            | 1                                            | (not present, but has kd: 0.0)                     |
| impedance_mode           | fixed                                        | (not present, but has compliance_mode: fixed)      |
| kp_limits                | [0, 300]                                     | [0, 300]                                           |
| damping_ratio_limits     | [0, 10]                                      | [0, 100]                                           |
| position_limits          | null                                         | null                                               |
| control_delta            | true                                         | true                                               |
| interpolation            | null                                         | null                                               |
| ramp_ratio               | 0.2                                          | 0.2                                                |
| ft_buffer_size           | 25                                           | 25                                                 |
| default_orientation      | [0, 0, 0, 1]                                 | (not present)                                      |
| orientation_limits       | (not present)                                | null                                               |
| selection_matrix         | (not present)                                | [1, 1, 1, 1, 1, 1]                                 |
| enable_selection_matrix  | (not present)                                | false                                              |
| goal_update_mode         | (not present)                                | achieved                                           |
| frame_of_reference       | (not present)                                | eef                                                |
| stiffness                | (not present)                                | 500                                                |
| stiffness_limits         | (not present)                                | [50, 500]                                          |
| error_scale              | (not present)                                | 0.005                                              |
| force_limits             | (not present)                                | [-50.0, 50.0]                                      |
| torque_limits            | (not present)                                | [-10.0, 10.0]                                      |
| compliance_mode          | (not present)                                | fixed                                              |
| iterations               | (not present)                                | 1                                                  |
| inner_controller_config  | (not present)                                | (OSC_POSE config block)                            |

**Notes:**
- FDCC wraps an inner OSC_POSE controller (see `inner_controller_config`).
- FDCC has additional compliance, force/torque, and selection matrix parameters.
- Some parameters (like `damping_ratio`, `impedance_mode`) are present in OSC_POSITION but replaced or renamed in FDCC.

````markdown
# Diffusion Policies w/ Reinforcement Learning for Contact‑Rich Manipulation

This branch implements Diffusion Policies with RL for contact‑rich robotic manipulation. Below are the steps to get started and test the ScuHand gripper with various robots.

---

## 1. Modify `scu_hand.xml`

Due to MuJoCo’s current `flexcomp` limitations, you must use an **absolute path** to load the mesh. Open:

```xml
models/assets/grippers/scu_hand.xml
````

and update the `<flexcomp>` tag:

```xml
<flexcomp
    type="mesh"
    file="/home/yongliangwang/Projects/robosuite_yl/robosuite/models/assets/grippers/meshes/scu_hand/sheet.obj"
    pos="0 -0.177269 0.1"
    quat="0.270598 0.653281 0.653282 -0.270598"
    dim="2"
    origin="0 0 0"
    radius="0.0001"
    rgba="0 1 0 0.3"
    mass="0.001"
    name="c">
  <contact   condim="3" solimp="0.99 0.99 0.01" solref="0.002 1"/>
  <edge      equality="true" solref="1e-5 1" solimp="1 1 0 0 1" damping="0.0"/>
  <elasticity poisson="0.49" thickness="0.0002" young="9e7" damping="0.00" elastic2d="bend"/>
</flexcomp>
```

> **Tip:** Replace the `file` attribute with the absolute path to your local `sheet.obj`.

---

## 2. Run UR5e with Different Grippers

To quickly benchmark UR5e paired with all available gripper models (including ScuHand), execute:

```bash
./run_ur5e_grippers.sh
```

This script loops through each gripper in `robosuite/models/assets/grippers/` and launches the `demo_composite_robot.py` demo.

---

## 3. Test ScuHand on Multiple Robot Platforms

If you want to see ScuHand mounted on different robots, run:

```bash
./run_robots_ScuHand.sh
```

This script iterates through every robot class in your branch and attaches ScuHand for a quick sanity check.

---

## 4. Customize the Demo Script

Edit the demo script to control simulation length or make the robot static:

```python
# File: demos/demo_composite_robot_ScuHand.py

# Change the number of simulation steps (default: 100)
for i in range(100):
    start = time.time()

    # Sample a random action
    action = np.random.uniform(low, high)

    # Uncomment the next line to hold the robot static
    # action = np.zeros(7)

    obs, reward, done, _ = env.step(action)

    # Cap frame rate if desired
    if max_fr is not None:
        elapsed = time.time() - start
        delay   = 1.0 / max_fr - elapsed
        if delay > 0:
            time.sleep(delay)

env.close()
```

* **To change the simulation length**, update `range(100)` to your desired number of steps.
* **To freeze the robot**, comment out the random action and uncomment `action = np.zeros(7)`.

---

Happy experimenting!
Feel free to open an issue or PR if you run into any problems.

```
```

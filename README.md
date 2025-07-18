# robosuite

This branch is for Diffusion Policies with Reinforcement Learning for Contact-Rich Robotic Manipulation project.

If u wanna see how ur5e with different grippers u should run:

./run_ur5e_grippers.sh 

if u wanna test ScuHand in various robots:

./run_robots_ScuHand.sh 

And u could modify the codings in robosuite/robosuite/demos/demo_composite_robot_ScuHand.py to control steps of the simulation. [modify 100 to any number]. And u could deannotate the action = np.zeros(7) make simulation static. 

    # Runs a few steps of the simulation as a sanity check
    for i in range(100):
        start = time.time()

        action = np.random.uniform(low, high)
        # print(action)
        # action = np.zeros(7)
        obs, reward, done, _ = env.step(action)

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    env.close()

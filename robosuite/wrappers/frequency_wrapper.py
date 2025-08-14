import numpy as np
from robosuite.environments.manipulation.osx_grind import OSXGrind, DEFAULT_GRIND_CONFIG


class FrequencyWrapper:
    """
    A wrapper that separates the control frequency for actions from the trajectory target frequency.

    This wrapper allows the environment to:
    1. Accept actions at a reduced frequency (e.g., 20Hz)
    2. Update controller parameters at this reduced frequency
    3. Continue sending controller targets at a high frequency (500Hz)

    Args:
        env_class: The environment class to instantiate
        env_config (dict): Configuration for the environment
        action_control_freq (int): Frequency at which actions are processed (Hz)
        trajectory_target_freq (int): Frequency at which trajectory targets are sent (Hz)
    """

    def __init__(self, env_class, env_config=None, action_control_freq=20, trajectory_target_freq=500):
        # Store the frequencies
        self.action_control_freq = action_control_freq
        self.trajectory_target_freq = trajectory_target_freq
        self.timestep = 0

        # Calculate the number of inner steps per outer step
        self.steps_per_action = int(trajectory_target_freq / action_control_freq)
        print(f"steps_per_action: {self.steps_per_action}")

        # Create a copy of the environment config
        env_config_copy = env_config.copy() if env_config else {}

        # Set the control frequency to the high frequency
        env_config_copy["control_freq"] = trajectory_target_freq
        self.ignore_done = env_config_copy.get("ignore_done", False)
        env_config_copy["ignore_done"] = True
        env_config_copy["action_control_freq"] = action_control_freq

        # Create the environment with the high frequency, passing all arguments transparently
        self.env = env_class(**env_config_copy)

        # Store the last action
        self.last_action = None

    def reset(self, **kwargs):
        """Reset the environment and return the initial observation."""
        obs = self.env.reset(**kwargs)
        self.env.horizon = self.env.num_waypoints
        self.last_action = None
        self.timestep = 0
        return obs

    def step(self, action):
        """
        Take a step in the environment with the given action.

        This method will:
        1. Store the action
        2. Execute multiple steps in the inner environment at the high frequency
        3. Return the final observation, accumulated reward, and done flag

        Args:
            action (np.array): The action to take

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Store the action
        self.last_action = action

        # Initialize accumulated reward and done flag
        total_reward = 0
        done = False
        info = {}

        # Execute multiple steps in the inner environment
        for _ in range(self.steps_per_action):
            if done:
                break

            # Use the stored action for each inner step
            obs, reward, done, info = self.env.step(self.last_action)
            total_reward += reward

            is_truncated = (self.timestep >= self.env.horizon) and not self.ignore_done
            done = done or is_truncated

            if is_truncated:
                info['termination_reason'] = "TRUNCATED"

        self.timestep += 1
        wrapper_reward = total_reward / self.steps_per_action
        return obs, wrapper_reward, done, info

    def render(self, **kwargs):
        """Render the environment."""
        return self.env.render(**kwargs)

    def close(self):
        """Close the environment."""
        return self.env.close()

    # Forward all other attributes to the wrapped environment
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Cannot access private attribute '{name}'")
        return getattr(self.env, name)


# For backward compatibility
def OSXGrindFrequencyWrapper(env_config=None, action_control_freq=20, trajectory_target_freq=500): return FrequencyWrapper(
    env_class=OSXGrind,
    env_config=env_config,
    action_control_freq=action_control_freq,
    trajectory_target_freq=trajectory_target_freq
)

# SB3 DQN IMPLEMENTATION FOR BREAKOUT (PART 2)
#This script is used to train the DQN agent on the Breakout environment.

#%% IMPORTS
import os
from datetime import datetime
import gymnasium as gym
import numpy as np
import ale_py
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper


#%% FOLDER MANAGEMENT
def create_run_folder(base_folder="dqn"):
    """
    Create a run folder with timestamped subfolders for training outputs.
    """
    os.makedirs(base_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = os.path.join(base_folder, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(os.path.join(run_folder, "videos"), exist_ok=True)
    os.makedirs(os.path.join(run_folder, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_folder, "plots"), exist_ok=True)
    return run_folder

#%% ENVIRONMENT SETUP
def make_env(env_name="BreakoutNoFrameskip-v4", frame_stack=4):
    def make_env():
        gym.register_envs(ale_py)
        env = gym.make(env_name, render_mode="rgb_array") #Render mode is set to rgb_array to return RGB images
        env = AtariWrapper(env, clip_reward=False) #Atari wrapper to preprocess the environment
        return env

    env = DummyVecEnv([make_env]) #Create a dummy vectorized environment
    env = VecFrameStack(env, n_stack=frame_stack) #Stack frames together
    return env


#%% CUSTOM CALLBACK
class TrainingMetricsCallback(BaseCallback):
    """
    Simplified callback to support training. 
    Tracks absolute rewards passed from the training loop.
    """
    def __init__(self, env, n_episodes, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.env = env
        self.n_episodes = n_episodes
        self.abs_rewards = []  # Tracks absolute rewards per episode

    def _on_step(self) -> bool:
        """
        This method is a placeholder required by the BaseCallback class.
        """
        return True


#%% PLOTTING METRICS FUNCTION
def plot_metrics(episode_rewards, output_folder):
    plt.figure(figsize=(16, 8))

    # Episode Rewards plot
    plt.plot(episode_rewards, label="Episode Rewards", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Training Rewards per Episode")
    plt.grid(True)
    plt.legend()

    # Save and show the plot
    plot_path = os.path.join(output_folder, "training_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Metrics plot saved at: {plot_path}")

#%% TRAINING THE AGENT
def train_agent(n_episodes):
    """
    Trainning of the DQN agent for Breakout.
    """
    run_folder = create_run_folder()
    env = make_env()

    # Define model hyperparameters
    policy_kwargs = {
        "net_arch": [256, 256],
    }
    dqn_params = {
        "policy": "CnnPolicy",
        "env": env,
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "learning_starts": 5000,
        "batch_size": 64,
        "tau": 0.8,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.02,
        "max_grad_norm": 10,
        "verbose": 1,
        "tensorboard_log": os.path.join(run_folder, "tensorboard"),
        "policy_kwargs": policy_kwargs,
    }

    
    model = DQN(**dqn_params) # model initialization
    callback = TrainingMetricsCallback(env=env, n_episodes=n_episodes)

    # Train for each episode
    for episode in range(n_episodes):
        obs = env.reset()  
        done_count = 0
        episode_reward = 0
        abs_rewards = []
        rewards = []

        while done_count < 5: 
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _ = env.step(action)
            #print(reward)
            episode_reward += reward
            
            if done:
                print("done")
                done_count += 1
                rewards.append(episode_reward)
                #print(rewards)
                episode_reward += reward  # Reset for the next episode
                # Reset for next episode
                if done_count == 5:
                    abs_reward = sum(rewards) #Sum of rewards for 5 episodes to get the absolute reward
                    #print(abs_reward,"a")
                    episode_reward = 0
                    rewards = []
                    abs_rewards.extend(abs_reward)
                    #print(abs_rewards, "b")
                    
        #print(abs_reward)
        callback.abs_rewards.append(abs_reward)  # Append the absolute reward to the list of rewards for the callback
        #print(callback.abs_rewards.append(abs_reward))
        print(f"Episode {episode + 1}/{n_episodes} - Reward: {abs_reward}")

    # Save the model
    final_model_path = os.path.join(run_folder, "models", "final_model.zip")
    model.save(final_model_path)
    print(f"Final model saved at: {final_model_path}")

    plot_metrics(callback.abs_rewards, run_folder)
    env.close()

#%% MAIN FUNCTION
if __name__ == "__main__":
    train_agent(n_episodes=5000)
# SB3 PPO IMPLEMENTATION FOR BREAKOUT (PART 2)
#This script is used to train the PPO agent on the Breakout environment.

#%% IMPORTS
import os
from datetime import datetime
import gymnasium as gym
import numpy as np
import ale_py
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper


#%% FOLDER MANAGEMENT
def create_run_folder(base_folder="ppo"):
    """
    Create a run folder with timestamped subfolders for training outputs.
    """
    os.makedirs(base_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = os.path.join(base_folder, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(os.path.join(run_folder, "models"), exist_ok=True)
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
    Custom callback to track training rewards and total loss.
    """
    def __init__(self, env, n_episodes, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.env = env
        self.n_episodes = 1
        self.episode_rewards = []
        self.episode_losses = []
        self.abs_rewards = []   

    def _on_step(self) -> bool:
        logs = self.model.logger #Get the logger object
        if logs:
            loss = logs.name_to_value.get("train/loss", 0.0) #Get the loss value from the logs
            self.episode_losses.append(loss) 
        return True

    def _on_rollout_end(self) -> None:
        # Compute the rewards at the end of the rollout
        rewards = []
        for _ in range(self.n_episodes):
            obs = self.env.reset()
            done_count = 0
            episode_reward = 0
            while done_count < 5: #5 episodes (lives) per rollout
                action, _ = self.model.predict(obs, deterministic=True) 
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward 
                if done:
                    done_count += 1
                    rewards.append(episode_reward)
                    episode_reward += reward  
                    if done_count == 5:
                        abs_reward = sum(rewards) #Sum of rewards for 5 episodes to get the absolute reward
                        #print(abs_reward,"a")
                        episode_reward = 0
                        rewards = []
                        self.abs_rewards.extend(abs_reward) #Append the absolute reward to the list
                        #print(self.abs_rewards, "b")

#%% PLOTTING METRICS FUNCTION
def plot_metrics(abs_rewards, episode_losses, output_folder):
    """
    Plot rewards and losses during training.
    """
    plt.figure(figsize=(16, 8))

    # Episode Rewards plot
    plt.subplot(2, 1, 1) 
    plt.plot(abs_rewards, label="Episode Rewards", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Training Rewards per Episode")
    plt.grid(True)
    plt.legend()

    # Episode Losses plot
    plt.subplot(2, 1, 2)  
    plt.plot(episode_losses, label="Episode Loss", color="red")
    plt.xlabel("Time Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss per Episode")
    plt.grid(True)
    plt.legend()

    # Save and show the plot
    plot_path = os.path.join(output_folder, "training_metrics.png")
    plt.tight_layout()  # To prevent overlap of plots
    plt.savefig(plot_path)
    plt.close()
    print(f"Metrics plot saved at: {plot_path}")


#%% TRAINING THE AGENT
def train_agent(n_episodes):
    """
    Training of a SB3 PPO agent for Breakout Atari env.
    """
    run_folder = create_run_folder()
    env = make_env()
    
    # Initialize model
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=os.path.join(run_folder, "tensorboard"),
    )

    # Create callback
    callback = TrainingMetricsCallback(env, n_episodes)

    # Compute timesteps based on the number of episodes
    timesteps = n_episodes * 128  
    
    # Train the agent
    model.learn(total_timesteps=timesteps, callback=callback)

    # Save the model
    model_path = os.path.join(run_folder, "models", "ppo_breakout.zip")
    model.save(model_path)
    print(f"Model saved at: {model_path}")

    # Plot and save training metrics (both rewards and losses)
    plot_metrics(callback.abs_rewards, callback.episode_losses, run_folder)

#%% MAIN FUNCTION
if __name__ == "__main__":
    train_agent(n_episodes=5000)  
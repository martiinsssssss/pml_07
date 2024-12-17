# DUELING DQN IMPLEMENTATION FOR BEAM RIDER (PART 1)
#This script is used to train the Dueling DQN agent on the Beam Rider environment.

#%% IMPORTS
import gymnasium as gym
import wandb
import datetime
import torch
import torch.nn as nn        
import torch.optim as optim 
import torch.nn.functional as F
from torchsummary import summary
import collections
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import random
from PIL import Image
from IPython.display import HTML
from tqdm import tqdm
import ale_py
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
from collections import namedtuple, deque
import copy
import os

#%% ENVIRONMENT SETUP
gym.register_envs(ale_py)
ENV_NAME = "ALE/BeamRider-v5"

env = gym.make(ENV_NAME).unwrapped

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    print("Standard Env.        : {}".format(env.observation_space.shape))
    env = MaxAndSkipObservation(env, skip = 20)
    print("MaxAndSkipObservation: {}".format(env.observation_space.shape))
    #env = FireResetEnv(env)
    env = ResizeObservation(env, (84, 84))
    print("ResizeObservation    : {}".format(env.observation_space.shape))
    env = GrayscaleObservation(env, keep_dim=True)
    print("GrayscaleObservation : {}".format(env.observation_space.shape))
    env = ImageToPyTorch(env)
    print("ImageToPyTorch       : {}".format(env.observation_space.shape))
    env = ReshapeObservation(env, (84, 84))
    print("ReshapeObservation   : {}".format(env.observation_space.shape))
    env = FrameStackObservation(env, stack_size=4)
    print("FrameStackObservation: {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    
    return env

env = make_env(ENV_NAME)


#%% REPLAY BUFFER
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(self.experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)

#%% DUELING DQN ARCHITECTURE
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # Fully connected layers for value and advantage streams
        self.value_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Single value output for state value
        )
        self.advantage_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)  # Advantage for each action
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        value = self.value_fc(conv_out)
        advantage = self.advantage_fc(conv_out)
        # Combining value and advantage into Q-values
        q_values = value + (advantage - advantage.mean())
        return q_values

#%% DQN AGENT
class DQNAgent:
    def __init__(self, env, replay_size=5000, batch_size=8, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999, sync_target_frames=2000):
        self.env = env
        self.n_actions = env.action_space.n
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.sync_target_frames = sync_target_frames
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DuelingDQN(env.observation_space.shape, self.n_actions).to(self.device)
        self.target_net = DuelingDQN(env.observation_space.shape, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.frame_idx = 0

    def act(self, state):
        """ 
        Selects an action using epsilon-greedy policy.

        Parameters:
        - state: Current state of the environment.

        Returns:
        - action: Action to take in the environment.
        """

        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        q_values = self.policy_net(state)
        return q_values.max(1)[1].item()

    def learn(self):
        """
        Samples a batch from the replay buffer and performs a single step of optimization.

        Returns:
        - loss: Loss value from the optimization step.
        """
        
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        self.frame_idx += 1
        if self.frame_idx % self.sync_target_frames == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # Visualization function for reward plot
    def visualise(self, score, run_folder, show_avg=True):
        """
        Parameters:
        - score: List of rewards per episode.
        - run_folder: Folder to save the plot.
        - show_avg: Whether to show the moving average of rewards.

        Returns:
        - None, it saves the plot in the run folder.
        """

        score = np.array(score)

        plt.figure(figsize=(15, 7))
        plt.ylabel("Trajectory duration", fontsize=12)
        plt.xlabel("Training iterations", fontsize=12)
        plt.plot(score, color='gray', linewidth=1, label='Score')

        if show_avg:
            N = 100
            avg_score = np.convolve(score, np.ones(N) / N, mode='valid')
            plt.plot(avg_score, color='blue', linewidth=3, label='Moving Average (window=100)')

        plt.scatter(np.arange(score.shape[0]), score, color='green', linewidth=0.3)
        plt.legend(fontsize=12)

        # Save the plot
        rewards_plot_path = os.path.join(run_folder, "reward_plot.png")
        plt.savefig(rewards_plot_path)
        plt.close()  # Close the plot to avoid display in Jupyter/other environments
        print(f"Rewards plot saved at: {rewards_plot_path}")

    # Loss plot function
    def plot_loss(self, losses, run_folder):
        plt.figure(figsize=(15, 7))
        plt.ylabel("Loss", fontsize=12)
        plt.xlabel("Training iterations", fontsize=12)
        plt.plot(losses, color='gray', linewidth=1, label='Loss')

        # Save the plot
        losses_plot_path = os.path.join(run_folder, "loss_plot.png")
        plt.savefig(losses_plot_path)
        plt.close()  # Close the plot to avoid display in Jupyter/other environments
        print(f"Loss plot saved at: {losses_plot_path}")

#%% TRAINING FUNCTION
def train(env, agent, num_episodes=500, max_timesteps=500):
    # Create a folder for the run
    run_folder = f"dueling_dqn_runs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_folder, exist_ok=True)
    print(f"Saving results to: {run_folder}")
    
    episode_rewards = []
    losses = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for t in range(max_timesteps):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)

            loss = agent.learn()
            #if loss > 0:
            losses.append(loss)

            state = next_state
            total_reward += reward
            if done:
                break

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        episode_rewards.append(total_reward)

        # Print the reward, epsilon, and average loss after each episode
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        if len(losses) > 0:
            print(f"  Average loss so far: {np.mean(losses):.4f}")

    # Save the plots and model
    agent.visualise(episode_rewards, run_folder)
    agent.plot_loss(losses, run_folder)

    model_path = os.path.join(run_folder, "dqn_model.pth")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Model saved at: {model_path}")

    return episode_rewards, losses

#%% TRAINING
agent = DQNAgent(env)
rewards, losses = train(env, agent, num_episodes=15000)
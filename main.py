import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import warnings
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

# Define ReplayMemory class
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define DQN class
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Define functions for action selection and model optimization
def select_action(state, steps_done, policy_net, EPS_START, EPS_END, EPS_DECAY, device, env):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, device):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                        if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument('--modelPath', type=str, default="/Users/sathvikyechuri/Desktop/Inspirit AI Research/model.pth", help='Path to the saved model')
    parser.add_argument('--savePath', type=str, default="/Users/sathvikyechuri/Desktop/Inspirit AI Research", help='Path to save the model')
    parser.add_argument('--evaluation', dest='evaluation', action='store_true', help='Only do evaluation using pretrained model')
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    if not args.evaluation:
        env = gym.make("CartPole-v1")

        n_actions = env.action_space.n
        state, info = env.reset()
        n_observations = len(state)

        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)
        steps_done = 0

        # Enable interactive mode for live plotting
        plt.ion()

        episode_durations = []
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        def plot_durations(show_result=False):
            plt.figure(1)
            durations_t = torch.tensor(episode_durations, dtype=torch.float)
            if show_result:
                plt.title('Result')
            else:
                plt.clf()
                plt.title('Training...')
            plt.xlabel('Episode')
            plt.ylabel('Duration')
            plt.plot(durations_t.numpy())
            if len(durations_t) >= 100:
                means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                plt.plot(means.numpy())

            plt.pause(0.001)
            if is_ipython:
                if not show_result:
                    display.display(plt.gcf())
                    display.clear_output(wait=True)
                else:
                    display.display(plt.gcf())

        for i_episode in range(500):
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = select_action(state, steps_done, policy_net, EPS_START, EPS_END, EPS_DECAY, device, env)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                memory.push(state, action, next_state, reward)
                state = next_state
                optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, device)

                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break

        print('Training complete')
        plot_durations(show_result=True)
        plt.ioff()
        plt.show()

        # Save the model to the path specified by --savePath
        model_save_path = args.savePath + '/model.pth'
        torch.save(policy_net.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        env.close()
    else:
        def load_model(filepath, n_observations, n_actions):
            model = DQN(n_observations, n_actions).to(device)
            model.load_state_dict(torch.load(filepath))
            model.eval()
            return model

        def test_model(model, num_episodes=10):
            env = gym.make("CartPole-v1", render_mode='human')
            
            for i_episode in range(num_episodes):
                state, info = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                
                done = False
                while not done:
                    with torch.no_grad():
                        action = model(state).max(1).indices.view(1, 1)
                    
                    observation, reward, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated
                    
                    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    # Optional: Add a small delay to make the rendering visible
                    # This might be necessary if the environment is rendering too quickly.
                    time.sleep(0.02)  # Adjust as needed (in seconds)
                
                print(f"Episode {i_episode + 1} finished after {done} timesteps")
            
            env.close()


        state, info = gym.make("CartPole-v1").reset()
        n_observations = len(state)
        n_actions = gym.make("CartPole-v1").action_space.n
        loaded_model = load_model(args.modelPath, n_observations, n_actions)
        test_model(loaded_model)

if __name__ == "__main__":
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    main()

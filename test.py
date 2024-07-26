import numpy as np
import gym
import time

n_states = 40
iter_max = 10000

initial_lr = 0.1  # Higher initial learning rate
min_lr = 0.01
gamma = 0.99  # Discount factor
t_max = 5000
initial_eps = 1.0  # Initial exploration rate
min_eps = 0.01
decay_factor = 0.995  # Decay rate for exploration

def run_episode(env, policy=None, render=False):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract the observation array if obs is a tuple

    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a, b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _, _ = env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract the observation array if obs is a tuple
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

def obs_to_state(env, obs):
    """ Maps an observation to state """
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract the observation array if obs is a tuple

    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    
    # Ensure observations are within bounds
    obs = np.asarray(obs).flatten()
    obs = np.clip(obs, env_low, env_high)
    
    # Calculate discrete values
    a = int((obs[0] - env_low[0]) / env_dx[0])
    b = int((obs[1] - env_low[1]) / env_dx[1])
    
    return a, b

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    np.random.seed(0)
    print('----- using Improved Q Learning -----')
    q_table = np.zeros((n_states, n_states, 3))
    epsilon = initial_eps
    for i in range(iter_max):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract the observation array if obs is a tuple
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eta = max(min_lr, initial_lr * (0.85 ** (i // 100)))
        for j in range(t_max):
            a, b = obs_to_state(env, obs)
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                logits = q_table[a][b]
                logits_exp = np.exp(logits)
                probs = logits_exp / np.sum(logits_exp)
                action = np.random.choice(env.action_space.n, p=probs)
            obs, reward, done, _, _ = env.step(action)
            if isinstance(obs, tuple):
                obs = obs[0]  # Extract the observation array if obs is a tuple
            total_reward += reward
            # update q table
            a_, b_ = obs_to_state(env, obs)
            q_table[a][b][action] = q_table[a][b][action] + eta * (reward + gamma * np.max(q_table[a_][b_]) - q_table[a][b][action])
            if done:
                break
        epsilon = max(min_eps, epsilon * decay_factor)
        if i % 100 == 0:
            print('Iteration #%d -- Total reward = %d.' % (i + 1, total_reward))
    solution_policy = np.argmax(q_table, axis=2)
    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    print("Average score of solution =", np.mean(solution_policy_scores))
    
    # Animate it after training
    env = gym.make(env_name, render_mode="human")
    run_episode(env, solution_policy, True)
    time.sleep(2)
    # Close the environment after use
    env.close()
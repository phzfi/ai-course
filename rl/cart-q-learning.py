import gymnasium as gym
import numpy as np

env = gym.make('CartPole-v1')

alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000

# initialize q-table

state_space = env.observation_space.shape[0]
action_space = env.action_space.n
q_table = np.zeros((state_space, action_space))

state_bins = (10, 10, 10, 10)  # Number of bins for each state dimension
q_table = np.zeros(state_bins + (action_space,))


# discretize the space

def discretize_state(state):
    bins = np.array([np.linspace(-4.8, 4.8, 10), np.linspace(-4, 4, 10), np.linspace(-0.418, 0.418, 10), np.linspace(-4, 4, 10)])
    state_index = []
    for i in range(len(state)):
        state_index.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_index)

# Q-learning algorithm

for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        total_reward += reward

        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += alpha * td_error

        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# Test the trained agent

env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()
state = discretize_state(state)
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])
    next_state, reward, done, _, _ = env.step(action)
    next_state = discretize_state(next_state)
    total_reward += reward
    state = next_state
    env.render()

print(f"Total Reward: {total_reward}")
env.close()

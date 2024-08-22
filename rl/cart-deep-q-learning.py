import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from collections import deque
import random

# define Q-network

def build_q_network(state_size, action_size):
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=state_size, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))
    return model

# Initialize environment and hyperparameters

env = gym.make('CartPole-v1', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

qnetwork = build_q_network(state_size, action_size)

episodes = 1000
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
memory = deque(maxlen=2000)

# Define functions for experience replay and action selection 

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    state = np.reshape(state, [1, state_size])
    action_values = qnetwork.predict(state)
    return np.argmax(action_values[0])

# Train the Deep Q-network agent

def replay():
    global epsilon
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            next_state = np.reshape(next_state, [1, state_size])
            target += gamma * np.amax(qnetwork.predict(next_state)[0])
        state = np.reshape(state, [1, state_size])
        target_f = qnetwork.predict(state)
        target_f[0][action] = target
        qnetwork.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

for e in range(episodes):
    state = env.reset()
    state = state[0]  # Gymnasium returns a tuple (state, info)
    for time in range(500):
        if e % 100 == 0:
            env.render()  # Render every 100 episodes
        action = act(state)
        next_state, reward, done, _, _ = env.step(action)
        remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e}/{episodes}, Score: {time}")
            break
    replay()




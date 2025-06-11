from collections import deque
from game import SnakeGameAI, Direction, Block, BLOCK_SIZE
from model import make_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.998

        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = make_model(input_shape=[11], hidden_size=128, output_size=3)
        self.loss_fn = keras.losses.MeanSquaredError()
        self.optimizer = keras.optimizers.Adam(learning_rate=LR)

    def _epsilon_greedy_policy(self, state):
        # Decay epsilon after each action
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        new_action = [0, 0, 0]

        if np.random.rand() < self.epsilon:
            action_choice = np.random.randint(0, 3)
            new_action[action_choice] = 1
        else:
            state = np.expand_dims(state, axis=0)  # Ensure state is 2D for the model
            Q_values = self.model.predict(state)
            action_choice = np.argmax(Q_values[0])
            new_action[action_choice] = 1

        return new_action

    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.memory), size=batch_size)
        batch = [self.memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones

    def play_one_step(self, env, state):
        action = self._epsilon_greedy_policy(state)
        next_state, reward, done, info = env.play_step(action)
        self.memory.append((state, action, reward, next_state, done))
        return next_state, action, reward, done, info

    def training_step(self, batch_size):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences

        # Predict Q-values for the next states
        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        
        # Calculate the target Q-values
        target_Q_values = rewards + self.gamma * max_next_Q_values * (1 - dones)

        with tf.GradientTape() as tape:
            # Predict Q-values for the current states
            all_Q_values = self.model(states)

            # Convert actions to numpy array for correct indexing
            actions = np.array(actions)  # Ensure actions are in array format
            actions = np.expand_dims(actions, axis=1)  # Add an extra dimension so it matches the shape of all_Q_values

            # Get the Q-values for the chosen actions
            chosen_Q_values = tf.reduce_sum(all_Q_values * tf.one_hot(actions, 3), axis=1)

            # Compute the loss
            loss = self.loss_fn(target_Q_values, chosen_Q_values)

        # Apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

if __name__ == "__main__":
    agent = Agent()
    env = SnakeGameAI()
    env.reset()

    state = env.get_state()

    while True:
        next_state, action, reward, done, info = agent.play_one_step(env, state)
        state = next_state

        if done:
            agent.n_games += 1
            env.reset()

        if len(agent.memory) > BATCH_SIZE:
            agent.training_step(BATCH_SIZE)


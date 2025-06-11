import gym
import numpy as np
import pickle as pkl

# Create environment
cliffEnv = gym.make("CliffWalking-v0")

# Initialize Q-table: 48 states × 4 actions
q_table = np.zeros((48, 4))

# SARSA Policy: ε-greedy
def policy(state, explore=0.0):
    if np.random.random() <= explore:
        return np.random.randint(0, 4)  # Random action
    return int(np.argmax(q_table[state]))  # Greedy action

# Parameters
EPSILON = 0.1     # Exploration rate
ALPHA = 0.1       # Learning rate
GAMMA = 0.9       # Discount factor
NUM_EPISODES = 500

for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    steps = 0

    state, _ = cliffEnv.reset()
    action = policy(state, EPSILON)

    while not done:
        next_state, reward, terminated, truncated, _ = cliffEnv.step(action)
        done = terminated or truncated

        next_action = policy(next_state, EPSILON)

        # SARSA update rule
        q_table[state][action] += ALPHA * (
            reward + GAMMA * q_table[next_state][next_action] - q_table[state][action]
        )

        state = next_state
        action = next_action
        total_reward += reward
        steps += 1

    print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

cliffEnv.close()
pkl.dump(q_table,open("sarsaqtable.pkl",'wb'))
print("training complete")


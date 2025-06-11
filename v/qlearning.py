import gym
import numpy as np
import pickle as pkl

cliffEnv = gym.make("CliffWalking-v0")

q_table = np.zeros((48, 4))  # 4x12 grid = 48 states, 4 actions

# Parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500

def policy(state, explore=0.0):
    if np.random.random() < explore:
        return np.random.randint(4)
    return int(np.argmax(q_table[state]))

for episode in range(NUM_EPISODES):
    state, _ = cliffEnv.reset()  # compatible with Gym >=0.26
    done = False
    total_reward = 0
    eplen = 0

    while not done:
        action = policy(state, EPSILON)
        next_state, reward, terminated, truncated, _ = cliffEnv.step(action)
        done = terminated or truncated

        # Q-learning update
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += ALPHA * (
            reward + GAMMA * q_table[next_state][best_next_action] - q_table[state][action]
        )

        state = next_state
        total_reward += reward
        eplen += 1

    print(f"Episode {episode+1}: Reward = {total_reward}, Length = {eplen}")

cliffEnv.close()
pkl.dump(q_table, open("qlearning_qtable.pkl", "wb"))

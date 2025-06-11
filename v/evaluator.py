
import gym
import cv2
import pickle as pkl
import numpy as np

cliffEnv = gym.make("CliffWalking-v0", render_mode="rgb_array")

#q_table = pkl.load(open("sarsaqtable.pkl", "rb"))

q_table = pkl.load(open("qlearning_qtable.pkl", "rb"))


# SARSA Policy: ε-greedy
def policy(state, explore=0.0):
    if np.random.random() <= explore:
        return np.random.randint(0, 4)
    return int(np.argmax(q_table[state]))

# Frame setup
def initialize_frame():
    width, height = 600, 200
    img = np.ones((height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2

    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), (0, 0, 0), 1)

    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), (0, 0, 0), 1)

    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), (255, 0, 255), -1)
    img = cv2.putText(img, "Cliff", (49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    img = cv2.putText(img, "G", (49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img

def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(state, (4, 12))
    return cv2.putText(img, "A", (49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

NUM_EPISODES = 5

for ep in range(NUM_EPISODES):
    done = False
    frame = initialize_frame()

    # Compatible with Gym v0.26+
    state, _ = cliffEnv.reset()
    total_reward = 0
    ep_len = 0

    while not done:
        frame2 = put_agent(frame.copy(), state)
        cv2.imshow("Cliff Walking", frame2)
        cv2.waitKey(250)

        action = policy(state)

        state, reward, terminated, truncated, _ = cliffEnv.step(action)
        done = terminated or truncated
        ep_len += 1
        total_reward += reward

    print(f"Episode {ep + 1}: Length = {ep_len}, Total Reward = {total_reward}")

cliffEnv.close()
cv2.destroyAllWindows()


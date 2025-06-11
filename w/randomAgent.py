
import gym
import cv2
import tensorflow as tf

# Set render_mode here
env = gym.make("CartPole-v1", render_mode="rgb_array")

for ep in range(5):
    done = False
    state, _ = env.reset()
    
    while not done:
        frame = env.render()
        cv2.imshow("CartPole", frame)
        cv2.waitKey(100)
        
        # Corrected random action generation
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32).numpy()
        
        # Correct unpacking for Gym v0.26+
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

env.close()
cv2.destroyAllWindows()


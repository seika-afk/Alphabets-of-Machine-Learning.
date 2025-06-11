import gym
import tensorflow as tf
from tensorflow import keras
from keras import Model, Input
from keras.layers import Dense

env = gym.make('CartPole-v1')

# Q-Network
net_input = Input(shape=(4,))  # Corrected shape
x = Dense(64, activation="relu")(net_input)
x = Dense(32, activation="relu")(x)
output = Dense(2, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)

# Optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Parameters
ALPHA = 0.001
EPSILON = 0.1
GAMMA = 0.99
NUM_EP = 500

# Policy function
def policy(state, explore=0.0):
    if tf.random.uniform(shape=(), maxval=1.0) <= explore:
        return tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    q_values = q_net(state)[0]
    return tf.argmax(q_values, output_type=tf.int32)

# Training loop
for episode in range(NUM_EP):
    done = False
    total_reward = 0
    ep_len = 0

    state = tf.convert_to_tensor([env.reset()[0]], dtype=tf.float32)
    action = policy(state, EPSILON)

    while not done:
        next_state_raw, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        next_state = tf.convert_to_tensor([next_state_raw], dtype=tf.float32)
        next_action = policy(next_state, EPSILON)

        target = reward
        if not done:
            target += GAMMA * tf.reduce_max(q_net(next_state)[0])

        with tf.GradientTape() as tape:
            q_values = q_net(state)
            q_value = q_values[0][action]
            loss = tf.square(target - q_value)

        grads = tape.gradient(loss, q_net.trainable_weights)
        optimizer.apply_gradients(zip(grads, q_net.trainable_weights))

        state = next_state
        action = next_action
        total_reward += reward
        ep_len += 1

    print(f"Episode {episode + 1}: Total Reward = {total_reward}, Length = {ep_len}")

q_net.save("saras_q_net.py")
env.close()


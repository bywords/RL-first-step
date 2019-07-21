import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# create bandit arms
bandit_arms = [0.2, 0, -0.2, -2]
num_arms = len(bandit_arms)

def pullBandit(bandit):
    # draw random value
    result = np.random.rand(1)
    if result > bandit:
        # positive reward
        return 1
    else:
        # negative reward
        return -1

tf.reset_default_graph()

weights = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(weights)

# Learning
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)

responsible_output = tf.slice(output, action_holder, [1])
loss = -(tf.log(responsible_output) * reward_holder)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
update = optimizer.minimize(loss)


total_episodes = 1000
total_reward = np.zeros(num_arms)

init = tf.global_variables_initializer()

# Launch the TF graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        actions = sess.run(output)
        a = np.random.choice(actions, p=actions)
        action = np.argmax(actions == a)

        # Get a reward from a bandit arm
        reward = pullBandit(bandit_arms[action])

        # Update the network
        _, resp, ww = sess.run([update, responsible_output, weights],
                               feed_dict={reward_holder: [reward], action_holder: [action]})

        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the {} arms of the bandit: {}".format(num_arms, total_reward))

        i += 1

print()
print("The agent thinks arm {} is the most promising....".format(np.argmax(ww)+1))
if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
    print("...and it was right!")
else:
    print("...and it was wrong!")

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# Initialize Q-table with all zeros
Q = np.zeros([env.observation_space.n,
              env.action_space.n])

# Set hyperaprameters
lr = .85
y = .99
num_episodes = 2000

rList = []
for i in range(num_episodes):
    s = env.reset()
    rAll = 0
    d = False
    j = 0

    # Q-Table learning algorithm
    while j < 99:
        j += 1
        # Select actions
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
        # new state with reward
        s1, r, d, _ = env.step(a)
        # Update Q-table
        Q[s, a] = Q[s, a] + lr * (r + y*np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        if d == True:
            break
    rList.append(rAll)

print("Score over time: {}".format(sum(rList) / num_episodes))

print("Final Q-Table Values")
print(Q)

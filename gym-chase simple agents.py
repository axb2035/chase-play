"""
Created on Mon Jun 24 16:29:35 2019

Create a basic test bed for the Chase gym environment...
"""
import gym
from copy import deepcopy
import numpy as np

from datetime import datetime
import random
from random import randrange


def write_chase_log(log, agent_name):
    file_time = datetime.now().strftime("%Y%m%d - %H%M")
    f_name = 'Chase - ' + agent_name + ' - ' + file_time + '.csv'
    with open(f_name, "w", newline="") as f:
        arena_pos = ""
        for v in range(400):
            arena_pos += str(v) + ","
        h_line = 'Episode,Step,Action,Reward,Done,' + arena_pos + '\n'
        f.write(h_line)
        for i in range(len(log)):
            log[i][5] = str(log[i][5]).replace(' ', ',')[1:-1]
            l_line = '{},{},{},{},{},{}\n'.format(log[i][0],
                                                  log[i][1],
                                                  log[i][2],
                                                  log[i][3],
                                                  log[i][4],
                                                  log[i][5],)
            f.write(l_line)
        f.close


# Added to stop LF being added when convereted to string.
np.set_printoptions(linewidth=1000)
env = gym.make('gym_chase:Chase-v0')

EPISODES = 10
e = 0
state_log = []

# Simple human agent
"""
while e < EPISODES:
    done = False
    e_step = 0
    total_reward = 0
    state, info = env.reset(random_seed=e)
    state = state.ravel()

    state_log.append([e, e_step, None, None, done, deepcopy(state)])

    while not done:
        env.render()
        print('\n7   8   9')
        print('  \\ | /')
        print('4 - 5 - 6')
        print('  / | \\')
        print('1   2   3')
        p_move = input('\nYour move [1-9 move, 5 stay still]:')
        n_state, r, done, dummy, info = env.step(int(p_move))
        print('\nEpisode:', e, 'Step:', e_step)
        print('\nReward:', r)
        total_reward += r
        e_step += 1
        n_state = n_state.ravel()
        state_log.append([e, e_step, p_move, r, done, deepcopy(n_state)])
    env.render()
    if total_reward == 5:
        print("\nAll robots eliminated. Total reward =", total_reward)
    else:
        print("\nAgent eliminated. Total reward =", total_reward)
    e += 1


write_chase_log(state_log, 'Human')

"""
# Simple random agent

while e < EPISODES:
    done = False
    e_step = 0
    total_reward = 0
    state, info = env.reset(seed=e)
    state = state.ravel()
    state_log.append([e, e_step, None, None, done, deepcopy(state)])

    random.seed()
    while not done:
        rnd_move = randrange(9) + 1
        rnd_move = 5
        n_state, r, done, dummy, info = env.step(rnd_move)
        total_reward += r
        e_step += 1
        n_state = n_state.ravel()
        state_log.append([e, e_step, rnd_move, r, done, deepcopy(n_state)])
    if total_reward == 5:
        print("All robots eliminated. Total reward =", total_reward)
    else:
        print("Agent eliminated. Total reward =", total_reward)
    e += 1

write_chase_log(state_log, 'Random')

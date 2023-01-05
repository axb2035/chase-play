"""A basic human player for Chase.

Created on Wed Nov 9 2022.
"""
# from gymnasium.utils.play import play
# import gym_examples
import gymnasium as gym

# import numpy as np

EPISODES = 10
e = 0
state_log = []

# Simple human agent
# TODO: Remove human agent and make the env work with:
# play(gymnasium.make('gym_chase:Chase-v1'))
# gymnasium has change how play works to only support pygame. So larger
# changes needed.

# play(gym.make('gym_chase:Chase-v1', render_mode='human'))

# mapping = {"2": 1,  # Down.
#            "4": 0,  # Left.
#            "6": 2,  # Right.
#            "8": 3,  # Up.
#             }
# default_action = np.array([0,0,0])
# play(gym.make('gym_examples:GridWorld-v0')) #, keys_to_action=mapping))

env = gym.make("gym_chase:Chase-v1", render_mode="human")
while e < EPISODES:
    done = False
    e_step = 0
    total_reward = 0
    state, info = env.reset(seed=e)
    # TODO: Update logging to support new internal game state.
    # state = state.ravel()
    # state_log.append([e, e_step, None, None, done, deepcopy(state)])

    while not done:
        env.render()
        print("\n7   8   9")
        print("  \\ | /")
        print("4 - 5 - 6")
        print("  / | \\")
        print("1   2   3")
        p_move = input("\nYour move [1-9 move, 5 stay still]:")
        n_state, r, done, dummy, info = env.step(int(p_move))
        print("\nEpisode:", e, "Step:", e_step)
        print("\nReward:", r)
        total_reward += r
        e_step += 1
        # n_state = n_state.ravel()
        # state_log.append([e, e_step, p_move, r, done, deepcopy(n_state)])
    env.render()
    if total_reward == 5:
        print("\nAll robots eliminated. Total reward =", total_reward)
    else:
        print("\nAgent eliminated. Total reward =", total_reward)
    e += 1

# write_chase_log(state_log, 'Human')

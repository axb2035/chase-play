"""A basic random agent for Chase.

Created on Mon Jun 24 16:29:35 2019.
"""
import argparse
import random
from datetime import datetime
from typing import Optional

import gymnasium as gym

# import gym
import numpy as np


# TODO: Update logging to support new internal game state.
# from copy import deepcopy
def write_chase_log(log, agent_name):
    """Write the log to a csv file."""
    file_time = datetime.now().strftime("%Y%m%d - %H%M")
    f_name = "Chase - " + agent_name + " - " + file_time + ".csv"
    with open(f_name, "w", newline="") as f:
        arena_pos = ""
        for v in range(400):
            arena_pos += str(v) + ","
        h_line = "Episode,Step,Action,Reward,Done," + arena_pos + "\n"
        f.write(h_line)
        for i in range(len(log)):
            log[i][5] = str(log[i][5]).replace(" ", ",")[1:-1]
            l_line = "{},{},{},{},{},{}\n".format(
                log[i][0],
                log[i][1],
                log[i][2],
                log[i][3],
                log[i][4],
                log[i][5],
            )
            f.write(l_line)
        f.close


# Simple random agent
# TODO: make the env work with play(gymnasium.make('gym_chase:Chase-v1'))


def play(epsiodes: Optional[int] = 2, possum: Optional[bool] = False):
    """Play Chase using a random agent."""
    env = gym.make("gym_chase:Chase-v1")
    e = 0
    while e < epsiodes:
        terminated = False
        e_step = 0
        total_reward = 0
        state, info = env.reset(seed=e)
        # TODO: Update logging to support new internal game state.
        # state = state.ravel()
        # state_log.append([e, e_step, None, None, done, deepcopy(state)])

        random.seed()
        while not terminated:
            if possum:
                action = 4
            else:
                action = env.action_space.sample()
            n_state, r, terminated, truncated, info = env.step(action)
            total_reward += r
            e_step += 1
            # n_state = n_state.ravel()
            # state_log.append([e, e_step, rnd_move, r, done, deepcopy(n_state)])
        if total_reward == 5:
            print("All robots eliminated. Total reward =", total_reward)
        else:
            print("Agent eliminated. Total reward =", total_reward)
        e += 1

    # write_chase_log(state_log, 'Random')


def main():
    """Run the main function."""
    # Added to stop LF being added when converted to string.
    np.set_printoptions(linewidth=1000)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--episodes", type=int, default=2, help="Number of episodes to run."
    )
    parser.add_argument(
        "-p", "--possum", type=bool, default=False, help="Don't move agent. i.e. If I don't move maybe they won't see me."
    )
    args = parser.parse_args()

    play(args.episodes, args.possum)


if __name__ == "__main__":
    main()

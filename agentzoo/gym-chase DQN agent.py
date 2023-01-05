"""AXB implementation of DQN with Keras."""

import copy
import random
from collections import deque
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam

# from keras.utils import to_categorical

# from gym_chase.envs import ChaseEnv
env = gym.make("gym_chase:Chase-v1")


class DQNAgent:
    """Keras based DQN agent for Chase RL environment."""

    def __init__(self, state_size, action_size):
        """Initialize the DQN agent."""
        # Environment parameters.
        self.state_size = state_size
        self.action_size = action_size

        # Experience Replay Buffer parameters.
        self.memory = deque(maxlen=2000)

        # Discount rate.
        self.gamma = 0.99

        # ANN parameters.
        self.learning_rate = 0.001
        self.batch_size = 32

        # Exploration v exploitation (Epsilon) parameter.
        self.epsilon = 0.1

        # Epsilon decay parameters.
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.999

        self.dqn_model = self.build_dqn()

    # Build model.
    # Need to research architecture - TO DO
    def build_dqn(self):
        """Build the DQN model."""
        dqn_model = Sequential()
        dqn_model.add(Dense(128, input_dim=400, activation="relu"))
        dqn_model.add(Dense(128, activation="relu"))
        dqn_model.add(Dense(self.action_size, activation="softmax"))
        dqn_model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return dqn_model

    # Determine an action!
    def action(self, state):
        """Determine an action."""
        if random.random() <= self.epsilon:
            act = random.randrange(self.action_size) + 1
            # print('Action (random):', act)
        else:
            act_values = self.dqn_model.predict(state.reshape(1, -1) / 4)
            # Argmax will return value from 0-8 when need 1-9.
            act = np.argmax(act_values[0]) + 1
            # print('Action (model):', act)
        return act

    # Manage replay buffer.
    def update_memory(self, state, action, r, n_state, done):
        """Update the replay buffer."""
        self.memory.append((state, action, r, n_state, done))

    # Update DQN using replay buffer.
    def update_model(self):
        """Update the DQN model."""
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, r, n_state, done in minibatch:
            target = r
            if not done:
                target = r + self.gamma * np.amax(
                    self.dqn_model.predict(n_state.reshape(1, -1) / 4)[0]
                )
            target_f = self.dqn_model.predict(state.reshape(1, -1) / 4)
            target_f[0][action - 1] = target
            self.dqn_model.fit(state.reshape(1, -1) / 4, target_f, epochs=1, verbose=0)
        # Epsilon decay
        # if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load a saved model."""
        self.dqn_model = load_model(name)

    def save(self, name):
        """Save the model."""
        self.dqn_model.save(name)


def write_chase_log(log, agent_name):
    """Write the log to a csv file."""
    file_time = datetime.now().strftime("%Y%m%d - %H%M")
    f_name = "Chase - " + agent_name + " - " + file_time + ".csv"
    with open(f_name, "w", newline="") as f:
        h_line = "Episode,Step,Action,Reward,Done,{}\n".format(
            np.array2string(np.arange(0.0, 400.0))
            .replace(".", ",")
            .replace("\n", "")
            .replace(" ", "")[1:-2]
        )
        f.write(h_line)
        for i in range(len(log)):
            l_line = "{},{},{},{},{},{}\n".format(
                log[i][0],
                log[i][1],
                log[i][2],
                log[i][3],
                log[i][4],
                str(log[i][5]).replace(".", ",").replace("\n", "")[1:-2],
            )
            f.write(l_line)
        f.close


# run episodes

state_size = env.observation_space.shape[0] * env.observation_space.shape[1]

agent = DQNAgent(state_size, env.action_space)
agent.load("exp_23f_train.h5")

model_name = "exp_23g"
EPISODES = 100000
e_start = 300000
e = e_start
e_steps = []
t_rewards = []
state_log = []

# Simple DQN agent
start_time = datetime.now()

while e < EPISODES + e_start:
    done = False
    e_step = 1
    total_reward = 0
    state = env.reset(e - 1).ravel()
    state_log.append([e, e_step, None, None, done, copy.deepcopy(state)])
    # state = to_categorical(state, num_classes=5)

    # Reinitialize random seed.
    random.seed()
    #    print('------------------------------')
    # env.render()
    while not done:
        # print('Episode:', e, 'Step:', e_step)

        # Choose action based on state.
        action = agent.action(state)

        # Send action to environment and get new state.
        n_state, r, done = env.step(action)
        n_state = n_state.ravel()
        state_log.append([e, e_step, str(action), r, done, copy.deepcopy(n_state)])
        # n_state = to_categorical(n_state, num_classes=5)
        # env.render()

        # Update the memory queue.
        agent.update_memory(state, action, r, n_state, done)

        # update model/learning
        if len(agent.memory) > agent.batch_size:
            agent.update_model()

        # update the old state to the new state
        state = n_state

        # Capture info for future...
        # print('Reward:', r)
        total_reward += r
        e_step += 1

    #    if total_reward == 5:
    #        print("All robots eliminated. Total reward =", total_reward)
    #    else:
    #        print("Agent eliminated. Total reward =", total_reward)
    # print('Memory size:', len(agent.memory))
    # print('Epsilon:{:.2}'.format(agent.epsilon))
    # print('Episode:', e, 'Steps', e_step)
    e_steps.append(e_step)
    t_rewards.append(total_reward)
    if e % 100 == 0:
        print("Completed episode", e)
    e += 1


end_time = datetime.now()
print(f"Duration: {end_time - start_time}")

agent.save(model_name + "_train.h5")
# agent.to_json()

write_chase_log(state_log, model_name + "_train")

plt.figure(figsize=(14, 10))
plt.plot(e_steps, label="steps")
plt.plot(t_rewards, label="reward")
plt.yticks(
    np.arange(
        min(t_rewards),
        max(e_steps) + 1,
        step=int((max(e_steps) - min(t_rewards)) / 10 + 1),
    )
)
plt.legend(loc="upper right")
plt.title(
    "Chase DQN - "
    + model_name
    + "\nepisodes="
    + str(EPISODES)
    + " e="
    + str(agent.epsilon)
    + " gamma="
    + str(agent.gamma)
    + " lr="
    + str(agent.learning_rate)
    + " minibatch="
    + str(agent.batch_size)
    + " replay buffer="
    + str(len(agent.memory))
    + "\narch: i128 relu, h128 relu, o9 softmax loss=mse, optimiser=adam"
    + f"\nDuration: {end_time - start_time}"
)
plt.show
plt.savefig(model_name + ".png")

# write stats to file...

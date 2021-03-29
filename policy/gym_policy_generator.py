import numpy as np
import gym
from gym import spaces

from moving_luggage.constants import (
    AgentActions, KEY_AGENTS, KEY_BAGS, LATENT_HEAVY_BAGS, LATENT_LIGHT_BAGS, NUM_X_GRID, NUM_Y_GRID)
from moving_luggage.simulator import Simulator
from policy.policy_utils import conv_to_np_env

from stable_baselines.common.env_checker import check_env
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.cmd_util import make_vec_env


class MovingLuggageEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, mental_model_idx):
        super().__init__()
        self.game = Simulator()
        self.env_id = 0
        self.mental_model = mental_model_idx

        self.action_space = spaces.Discrete(len(AgentActions)**2)

        # empty: 0 / bag: 1 / a1: 2 / a2: 3 / a1&a2: 4
        # bag&a1: 5 / bag&a1h: 6 / bag&a2: 7 / bag&a2h: 8
        # bag&a1&a2: 9 / bag&a1h&a2: 10 / bag&a1&a2h: 11 / bag&a1h&a2h: 12
        num_cases = 13
        self.observation_space = spaces.Box(
            low=0, high=(num_cases - 1),
            shape=(self.game.grid_x * self.game.grid_y, ), dtype=np.uint8)

    def reset(self):
        self.game.finish_game(self.env_id)
        self.game.add_new_env(self.env_id, int(NUM_X_GRID * NUM_Y_GRID / 4))

        env = self.game.map_id_env[self.env_id]
        np_env = conv_to_np_env(
            env[KEY_BAGS], env[KEY_AGENTS][0], env[KEY_AGENTS][1])

        return np_env

    def step(self, joint_action):
        env = self.game.map_id_env[self.env_id]
        action1 = joint_action % len(AgentActions)
        action2 = int(joint_action / len(AgentActions))

        agent1 = env[KEY_AGENTS][0]
        agent2 = env[KEY_AGENTS][1]
        a1_pos_prev = agent1.coord
        a2_pos_prev = agent2.coord
        a1_hold_prev = agent1.hold
        a2_hold_prev = agent2.hold

        move_actions = [
            AgentActions.UP.value, AgentActions.DOWN.value,
            AgentActions.LEFT.value, AgentActions.RIGHT.value]

        reward = 0
        # for the case where agents perceive bags as heavy.
        if self.mental_model == LATENT_HEAVY_BAGS:
            if a1_pos_prev != a2_pos_prev:
                if a1_hold_prev and (action1 in move_actions):
                    reward -= 1
                if a2_hold_prev and (action2 in move_actions):
                    reward -= 1

        n_bags_prev = np.sum(env[KEY_BAGS])

        self.game._take_simultaneous_step(
            env, AgentActions(action1), AgentActions(action2))

        np_bags = env[KEY_BAGS]

        n_bags = np.sum(np_bags)
        np_env = conv_to_np_env(np_bags, agent1, agent2)

        n_bag_diff = n_bags_prev - n_bags
        reward += n_bag_diff * 10

        done = bool(n_bags == 0)
        info = {}

        return np_env, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise RuntimeError("Not implemented")

        env = self.game.map_id_env[self.env_id]
        np_env = conv_to_np_env(
            env[KEY_BAGS], env[KEY_AGENTS][0], env[KEY_AGENTS][1])

        print(np_env.reshape((self.game.grid_x, self.game.grid_y)))


if __name__ == "__main__":
    env = MovingLuggageEnv(LATENT_LIGHT_BAGS)
    # check_env(env, warn=True)

    # obs = env.reset()
    # env.render()

    # print(env.observation_space)
    # print(env.action_space)
    # print(env.action_space.sample())

    # # Hardcoded agent: a1 goes left, a2 goes up
    # GO_LEFT = 3 + 6 * 1
    # n_steps = 2
    # for step in range(n_steps):
    #     print("Step {}".format(step + 1))
    #     obs, reward, done, info = env.step(GO_LEFT)
    #     print('obs=', obs, 'reward=', reward, 'done=', done)
    #     env.render()
    #     if done:
    #         print("Goal reached!", "reward=", reward)
    #         break


    env = make_vec_env(lambda: env, n_envs=1)
    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("deepq_movingluggage_light")

    # del model
    # model = DQN.load("deepq_movingluggage")

    obs = env.reset()
    n_steps = 10
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print("Step {}".format(step + 1))
        action1 = action % len(AgentActions)
        action2 = int(action / len(AgentActions))
        print("Action1: " + AgentActions(action1).name)
        print("Action2: " + AgentActions(action2).name)

        obs, reward, done, info = env.step(action)
        print('reward=', reward, 'done=', done)
        env.render(mode='console')
        if done:
            break



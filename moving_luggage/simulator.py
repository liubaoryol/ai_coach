import numpy as np
from threading import Lock, Timer
import time
from moving_luggage.constants import *
from moving_luggage.transition import transition
from moving_luggage.objects import ObjAgent
# from moving_luggage.policies import get_dqn_policy
# from moving_luggage.hand_policies import get_qlearn_policy


class Simulator():
    def __init__(self, log_dir='.'):
        self.map_id_env = {}

        self.grid_x = NUM_X_GRID
        self.grid_y = NUM_Y_GRID
        self.lock_key_queue = Lock()
        # step length
        self.step_length = 500  # msec
        self.cb_renderer = None
        self.cb_game_end = None
        self.cb_get_policy_action = None
        self.log_dir = log_dir
        self.max_steps = 300 * 1000 / self.step_length  # default: 5 min

        mid = int(self.grid_y / 2 - 1)
        self.goal_pos = [(self.grid_x - 1, mid), (self.grid_x - 1, mid + 1)]
        self.agent1_pos = (self.grid_x - 1, mid - 1)
        self.agent2_pos = (self.grid_x - 1, mid + 2)

    def set_max_step(self, max_step):
        self.max_steps = max_step

    def generate_init_bags(self, num_bag):
        np_bags = np.zeros((self.grid_x, self.grid_y))

        num_grid = (self.grid_x - 1) * self.grid_y
        bag_indices = np.random.choice(num_grid, num_bag, replace=False)
        def to_coord(idx):
            x = idx % (self.grid_x - 1)
            y = idx / (self.grid_x - 1)
            return (int(x), int(y))
        
        # add bags
        for idx in range(num_bag):
            np_bags[to_coord(bag_indices[idx])] = 1

        return np_bags

    def add_new_env(self, env_id, num_bag):
        if env_id in self.map_id_env:
            raise RuntimeError("already have the same eid")
            return

        self.map_id_env[env_id] = {
            KEY_BAGS: self.generate_init_bags(num_bag),
            KEY_AGENTS: [],
            KEY_INPUT: [[], []],
            KEY_TIMER: None,
            KEY_USERNAME: "",
            KEY_STEPS: 0}
        env = self.map_id_env[env_id]

        env[KEY_AGENTS].append(ObjAgent(self.agent1_pos))
        env[KEY_AGENTS].append(ObjAgent(self.agent2_pos))

    def _get_policy_action(self, env, agent):
        np_bags = env[KEY_BAGS]
        optimal_actions = []
        if agent.hold:
            close_goal = None
            min_dist = 10000000000
            for pos in self.goal_pos:
                dist = abs(agent.coord[0] - pos[0]) + abs(agent.coord[1] - pos[1])
                if dist < min_dist:
                    min_dist = dist
                    close_goal = pos
            if close_goal is not None:
                x_diff = close_goal[0] - agent.coord[0]
                if x_diff < 0:
                    optimal_actions.append(AgentActions.LEFT)
                elif x_diff > 0:
                    optimal_actions.append(AgentActions.RIGHT)
                    
                y_diff = close_goal[1] - agent.coord[1]
                if y_diff < 0:
                    optimal_actions.append(AgentActions.UP)
                elif y_diff > 0:
                    optimal_actions.append(AgentActions.DOWN)
        else:
            if np_bags[agent.coord] != 0:
                optimal_actions.append(AgentActions.HOLD)
            else:
                bags_pos = np.argwhere(np_bags)
                close_bag = None
                min_dist = 10000000000
                for pos in bags_pos:
                    dist = abs(agent.coord[0] - pos[0]) + abs(agent.coord[1] - pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        close_bag = pos
                if close_bag is not None:
                    x_diff = close_bag[0] - agent.coord[0]
                    if x_diff < 0:
                        optimal_actions.append(AgentActions.LEFT)
                    elif x_diff > 0:
                        optimal_actions.append(AgentActions.RIGHT)
                        
                    y_diff = close_bag[1] - agent.coord[1]
                    if y_diff < 0:
                        optimal_actions.append(AgentActions.UP)
                    elif y_diff > 0:
                        optimal_actions.append(AgentActions.DOWN)

        if len(optimal_actions) == 0:
            optimal_actions.append(AgentActions.STAY)

        action = np.random.choice(optimal_actions)

        return action

    def _get_action(self, env):
        action1 = None
        action2 = None
        with self.lock_key_queue:
            if len(env[KEY_INPUT][0]) > 0:
                action1 = env[KEY_INPUT][0].pop(0)
            if len(env[KEY_INPUT][1]) > 0:
                action2 = env[KEY_INPUT][1].pop(0)
         
        if action1 is None:
            agent = env[KEY_AGENTS][0]
            if agent.id is not None:
                action1 = AgentActions.STAY
            else:
                # action1 = AgentActions.STAY
                # retrieve policy
                # action1 = get_dqn_policy(env, 0, LATENT_HEAVY_BAGS)

                # if env[KEY_STEPS] < 2:
                #     action1 = AgentActions.STAY
                # else:
                #     action1 = self._get_policy_action(env, agent)
                if self.cb_get_policy_action:
                    action1 = self.cb_get_policy_action(
                        env, 0, LATENT_HEAVY_BAGS)
                else:
                    action1 = AgentActions.STAY
                # action1 = get_qlearn_policy(
                #     env, 0, LATENT_LIGHT_BAGS, self.goal_pos)

        if action2 is None:
            agent = env[KEY_AGENTS][1]
            if agent.id is not None:
                action2 = AgentActions.STAY
            else:
                # action2 = AgentActions.STAY
                # retrieve policy
                # action2 = get_dqn_policy(env, 1, LATENT_HEAVY_BAGS)
                if self.cb_get_policy_action:
                    action2 = self.cb_get_policy_action(
                        env, 1, LATENT_HEAVY_BAGS)
                else:
                    action2 = AgentActions.STAY
                # action2 = get_qlearn_policy(
                #     env, 1, LATENT_LIGHT_BAGS, self.goal_pos)
        
        return action1, action2

    def set_callback_renderer(self, callback_renderer):
        self.cb_renderer = callback_renderer 

    def set_callback_game_end(self, cb_game_end):
        self.cb_game_end = cb_game_end 

    def set_callback_policy(self, callback_policy):
        self.cb_get_policy_action = callback_policy 

    def run_game(self, env_id):
        if env_id not in self.map_id_env:
            return
        
        env = self.map_id_env[env_id]
        self._periodic_actions(env, env_id)

    def take_a_step_and_get_objs(self, env_id, a_id, action):
        if env_id not in self.map_id_env:
            return

        self.action_input(env_id, a_id, action)
        env = self.map_id_env[env_id]
        action1, action2 = self._get_action(env)
        self._take_simultaneous_step(env, action1, action2)

        # is finished
        if self.is_finished(env_id):
            self.finish_game(env_id)
            return None
        return self._get_object_list(env)

    def is_finished(self, env_id):
        if env_id not in self.map_id_env:
            return True  # already finished

        env = self.map_id_env[env_id]
        return np.sum(env[KEY_BAGS]) == 0 or env[KEY_STEPS] > self.max_steps
    
    def _periodic_actions(self, env, env_id):
        if env_id in self.map_id_env:
            # processing. timer ...
            action1, action2 = self._get_action(env)
            self._take_simultaneous_step(env, action1, action2)

            # update scene
            obj_list = self._get_object_list(env)
            if self.cb_renderer:
                self.cb_renderer(obj_list, env_id)
            time.sleep(0)

            # is finished
            if self.is_finished(env_id):
                self.finish_game(env_id)
                return
            
            # sec, msec = divmod(time.time() * 1000, 1000)
            # time_stamp = '%s.%03d' % (
            #     time.strftime('%Y-%m-%d_%H_%M_%S', time.gmtime(sec)), msec)
            # print("action excuted: " + time_stamp)
            env[KEY_TIMER] = Timer(
                self.step_length / 1000,
                self._periodic_actions,
                [env, env_id])
            env[KEY_TIMER].start()

    def finish_game(self, env_id):
        if env_id in self.map_id_env:
            env = self.map_id_env.pop(env_id)
            # may need to check the current thread and the timer thread
            # are the same to avoid timer.cancel() being called
            # inside the timer thread itself.
            if env[KEY_TIMER] is not None:
                env[KEY_TIMER].cancel()

            if self.cb_game_end:
                self.cb_game_end(env_id)
   
    def set_user_name(self, env_id, user_name):
        if env_id not in self.map_id_env:
            return False
        
        env = self.map_id_env[env_id]
        env[KEY_USERNAME] = user_name
         
    def connect_agent_id(self, env_id, agent_id):
        if env_id not in self.map_id_env:
            return False
        
        env = self.map_id_env[env_id]

        if env[KEY_AGENTS][0].id is None:
            env[KEY_AGENTS][0].id = agent_id
            return True
        elif env[KEY_AGENTS][1].id is None:
            env[KEY_AGENTS][1].id = agent_id
            return True
        else:
            # both agents are already taken
            return False

    def _get_agent_idx_by_id(self, env, aid):
        if env[KEY_AGENTS][0].id == aid:
            return 0
        elif env[KEY_AGENTS][1].id == aid:
            return 1
        else:
            return -1
            # raise RuntimeError("no agent by that id")

    def action_input(self, eid, aid, action):
        if eid not in self.map_id_env:
            return

        env = self.map_id_env[eid]
        idx = self._get_agent_idx_by_id(env, aid)
        if idx < 0:
            return
        with self.lock_key_queue:
            current_key_list = env[KEY_INPUT][idx]
            if len(current_key_list) == 0:
                current_key_list.append(action)
        
    def _take_simultaneous_step(self, env, action1, action2):
        np_bags = env[KEY_BAGS]
        agent1 = env[KEY_AGENTS][0]
        agent2 = env[KEY_AGENTS][1]

        list_next_env = transition(
            np_bags, agent1.coord, agent2.coord, agent1.hold, agent2.hold,
            action1, action2, self.goal_pos)

        np_p = np.zeros(len(list_next_env))
        for idx, item in enumerate(list_next_env):
            np_p[idx] = item[0]
        idx_c = np.random.choice(np.arange(len(list_next_env)), 1, p=np_p)[0]
        _, np_bags_n, a1_pos, a2_pos, a1_h, a2_h = list_next_env[idx_c]
        agent1.coord = a1_pos
        agent2.coord = a2_pos
        agent1.hold = a1_h
        agent2.hold = a2_h
        env[KEY_BAGS] = np_bags_n
        env[KEY_STEPS] += 1

    def _get_object_list(self, env):
        np_bags = env[KEY_BAGS]
        agent1 = env[KEY_AGENTS][0]
        agent2 = env[KEY_AGENTS][1]

        def coord2idx(coord):
            return int(coord[1] * self.grid_x + coord[0])

        objs = {
            KEY_BAGS: [],
            KEY_A1_POS: coord2idx(agent1.coord),
            KEY_A2_POS: coord2idx(agent2.coord),
            KEY_A1_HOLD: 1 if agent1.hold else 0,
            KEY_A2_HOLD: 1 if agent2.hold else 0}

        bags_pos = np.argwhere(np_bags)
        for pos in bags_pos:
            objs[KEY_BAGS].append(coord2idx(pos))

        return objs

        
import threading
import queue
import sys
import os
import time
import pickle
from pathlib import Path
from tw_timer import *
from tw_data import *
from tw_actions import *
import tw_trajectory_util

POLICY_RANDOM = "random"

TAG_JOINT_PI = "joint_pi"
TAG_RL_AGENT = "rl_agent"
TAG_JOINT_LATENT = "joint_latent"
TAG_JOINT_VAL_FN = "joint_vale_fn"
TAG_SCORE_PREV = "score_prev"
TAG_MDP = "mdp"


class CAppTeamwork:
    def __init__(self, xml_file,
                 policy_name=POLICY_RANDOM, log_dir="trajectory_log",
                 agent1_pos=None, agent2_pos=None):
        self.log_directory = log_dir
        self.policy_name = policy_name
        self.xml_file = xml_file
        self.running = 0
        self.max_step = 100
        self.do_restart = False
        self.thread1 = None
        self.user_name = ''

        # control flow
        self.event_queue = queue.Queue()
        self.event_lock = threading.Lock()
        self.started = 0
        self.ai_mode = 0
        self.periodic_task = None

        # logging
        self.log = []
        self.episode_count = 0
        self.step_sum = 0.0

        self.agent1_pos = None
        self.agent2_pos = None

        # initialize data
        init_data_with_file(self.xml_file)

        # LCM of growing time
        self.lcm_t_g = self.lcm_grow_time()

        # policy related
        self.policydata = {}
        self.has_policy = True

        # web compatibility
        self.cb_update_web_scene = None
        self.cb_game_end = None

    def get_grid_size(self):
        return (NUM_GRID_X, NUM_GRID_Y)
    
    def reset_param(self):
        self.started = 0
        self.log.clear()
        self.event_queue = queue.Queue()

    def set_agent_pos(self, agent1_pos=None, agent2_pos=None):
        self.agent1_pos = agent1_pos
        self.agent2_pos = agent2_pos

    def add_agents(self):
        a1_pos = self.agent1_pos
        a2_pos = self.agent2_pos
        if a1_pos is None:
            a1_pos = get_random_agent_position()
        if a2_pos is None:
            a2_pos = get_random_agent_position()
        add_agent(a1_pos, a2_pos)
    
    def run_on_web(self, cb_update_web_scene, cb_game_end, user_sid, user_name, thread_fn = None):
        if self.running == 1:
            print("Already app is running")
            return False
        
        self.reset_param()
        self.user_sid = user_sid
        self.user_name = user_name

        self.cb_update_web_scene = cb_update_web_scene
        self.cb_game_end = cb_game_end

        # initialize data
        init_data_with_file(self.xml_file) 
        self.add_agents()

        step_length = 0.5     # in sec
        self.timers = CContinuousTimer(step_length, 0.1)

        # set worker
        if thread_fn:
            self.thread1 = thread_fn(target=self.worker_thread)
        else:
            self.thread1 = threading.Thread(target=self.worker_thread)
            self.thread1.start()

        # self.running = 1
        # self.on_data_changed(WORLD_OBJECTS)

        return True

    def set_max_step(self, max_step):
        self.max_step = max_step

    def set_log_directory(self, log_dir):
        self.log_directory = log_dir

    def get_policy_name(self):
        return self.policy_name

    def set_latent(self, joint_latent):
        if TAG_JOINT_LATENT in self.policydata.keys():
            self.policydata[TAG_JOINT_LATENT] = joint_latent
        else:
            print("Latent states is not available for this policy config.")

    def on_episode_end(self):
        steps = int(self.timers.get_current_time())
        self.step_sum += steps
        self.episode_count += 1
        train_episodes = 0

        if len(self.log) is not 0:
            dict_states = self.get_dict_states()
            state_coord = self.conv_states_2_coord(dict_states)
            state_coord.append("end")
            state_coord.append("end")
            self.log.append(state_coord)
            self.save_trajectory_log()

    def restart(self):
        self.timers.reset()
        self.reset_param()

        # initialize data
        dict_num_trash = {}
        dict_num_trash[OBJ_TYPE_1] = 0
        dict_num_trash[OBJ_TYPE_2] = 0
        dict_num_trash[OBJ_TYPE_3] = 3
        list_trash = generate_random_trash(dict_num_trash)
        init_data(list_trash)

        self.add_agents()


    def lcm_grow_time(self):
        set_t_g = set()
        for prop in OBJ_PROP.values():
            if prop is not None:
                if prop[2] > 0:
                    set_t_g.add(prop[2])

        num_t_g = len(set_t_g)
        if num_t_g == 0:
            return 0
        else:
            t_g_lcm = set_t_g.pop()
            for t_g in set_t_g:
                t_g_lcm = compute_lcm(t_g_lcm, t_g)

            return t_g_lcm

    def save_trajectory_log(self):
        # dump_folder = 'trajectory_log'
        dump_folder = self.log_directory
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)

        def dump_prop(prop):
            if prop is None:
                return '0, 0, 0'
            else:
                txt = ''
                if prop[0] <= 0:
                    txt += '0'
                else:
                    txt += str(prop[0])
                txt += ', '
                if prop[1] <= 0:
                    txt += '0'
                else:
                    txt += str(prop[1])
                txt += ', '
                if prop[2] <= 0:
                    txt += '0'
                else:
                    txt += str(prop[2])
                return txt

        txt_log = []
        for coords in self.log:
            txt_log.append(", ".join([str(elem) for elem in coords]))

        sec, msec = divmod(time.time() * 1000, 1000)
        time_stamp = '%s.%03d' % (
            time.strftime('%Y-%m-%d_%H_%M_%S', time.gmtime(sec)), msec)
        file_name = ('trajectory_' + str(self.user_name) + '_' + time_stamp + '.txt')
        with open(os.path.join(dump_folder, file_name),
                  'w', newline='') as txtfile:
            txtfile.write(str(self.user_name) + '\n')
            txt_prop = dump_prop(OBJ_PROP[OBJ_TYPE_1]) + ", "
            txt_prop += dump_prop(OBJ_PROP[OBJ_TYPE_2]) + ", "
            txt_prop += dump_prop(OBJ_PROP[OBJ_TYPE_3]) + "\n"
            txtfile.write(txt_prop)
            txtfile.writelines("\n".join(txt_log))

    def get_dict_states(self):
        dict_states = {AGENT_NAME_1: [], AGENT_NAME_2: [],
                       OBJ_TYPE_1: [], OBJ_TYPE_2: [], OBJ_TYPE_3: [],
                       STATE_TIMESTEP: None}
        # object states
        for dname in WORLD_OBJECTS:
            obj = OBJECT_MAP[dname]
            if obj.get_type() == TYPE_AGENT:
                dict_states[dname].append(obj.get_pos())
                dict_states[dname].append(obj.get_mode())
            else:
                dict_states[obj.get_type()].append(obj.get_pos())

        # timestep state
        if self.lcm_t_g <= 0:
            dict_states[STATE_TIMESTEP] = 0
        else:
            elapsed_time = self.timers.get_current_time()
            dict_states[STATE_TIMESTEP] = int(elapsed_time) % self.lcm_t_g

        return dict_states

    def conv_states_2_coord(self, dict_states):

        state_coord = []
        state_coord.append(dict_states[AGENT_NAME_1][0][0])  # agent1 x
        state_coord.append(dict_states[AGENT_NAME_1][0][1])  # agent1 y
        state_coord.append(dict_states[AGENT_NAME_2][0][0])  # agent2 x
        state_coord.append(dict_states[AGENT_NAME_2][0][1])  # agent2 y
        state_coord.append(dict_states[AGENT_NAME_1][1])  # agent1 mode
        state_coord.append(dict_states[AGENT_NAME_2][1])  # agent2 mode
        state_coord.append(len(dict_states[OBJ_TYPE_1]))  # numm trash_1
        for coord in dict_states[OBJ_TYPE_1]:
            state_coord.append(coord[0])  # trash_1 x
            state_coord.append(coord[1])  # trash_1 y
        state_coord.append(len(dict_states[OBJ_TYPE_2]))  # num trash_2
        for coord in dict_states[OBJ_TYPE_2]:
            state_coord.append(coord[0])  # trash_2 x
            state_coord.append(coord[1])  # trash_2 y
        state_coord.append(len(dict_states[OBJ_TYPE_3]))  # num trash_3
        for coord in dict_states[OBJ_TYPE_3]:
            state_coord.append(coord[0])  # trash_3 x
            state_coord.append(coord[1])  # trash_3 y

        # time step
        state_coord.append(dict_states[STATE_TIMESTEP])

        return state_coord

    def on_data_changed(self, data_names):
        if self.cb_update_web_scene:
            # always redraw all
            env_objs = []
            agent_objs = []
            for dname in WORLD_OBJECTS:
                obj = OBJECT_MAP[dname]
                if obj.get_type() != TYPE_AGENT:
                    env_objs = env_objs + obj.get_drawing_object()
                else:
                    agent_objs = agent_objs + obj.get_drawing_object()

            all_objs = env_objs + agent_objs
            self.cb_update_web_scene(all_objs, self.user_sid)

    def gui_key_cb(self, event):
        if self.ai_mode == 1 or self.running == 0:
            return

        agent_nm = ""
        key_nm = ""
        if event.keysym == "Left":
            agent_nm = AGENT_NAME_1
            key_nm = ACTION_LEFT
        elif event.keysym == "Right":
            agent_nm = AGENT_NAME_1
            key_nm = ACTION_RIGHT
        elif event.keysym == "Up":
            agent_nm = AGENT_NAME_1
            key_nm = ACTION_UP
        elif event.keysym == "Down":
            agent_nm = AGENT_NAME_1
            key_nm = ACTION_DOWN
        elif event.keysym == "a":
            agent_nm = AGENT_NAME_2
            key_nm = ACTION_LEFT
        elif event.keysym == "d":
            agent_nm = AGENT_NAME_2
            key_nm = ACTION_RIGHT
        elif event.keysym == "w":
            agent_nm = AGENT_NAME_2
            key_nm = ACTION_UP
        elif event.keysym == "s":
            agent_nm = AGENT_NAME_2
            key_nm = ACTION_DOWN
        elif event.keysym == "p":
            agent_nm = AGENT_NAME_1
            key_nm = ACTION_PICKUP
        elif event.keysym == "t":
            agent_nm = AGENT_NAME_2
            key_nm = ACTION_PICKUP
        elif event.keysym == "o":
            agent_nm = AGENT_NAME_1
            key_nm = ACTION_CLEAN
        elif event.keysym == "r":
            agent_nm = AGENT_NAME_2
            key_nm = ACTION_CLEAN
        elif event.keysym == "semicolon":
            agent_nm = AGENT_NAME_1
            key_nm = ACTION_DROP
        elif event.keysym == "f":
            agent_nm = AGENT_NAME_2
            key_nm = ACTION_DROP

        if agent_nm != "" and key_nm != "":
            if agent_nm == AGENT_NAME_1:
                with self.event_lock:
                    dict_states = self.get_dict_states()
                    state_coord = self.conv_states_2_coord(dict_states)
                    state_coord.append(agent_nm)
                    state_coord.append(key_nm)
                    self.log.append(state_coord)
                    self.event_queue.put((ETYPE_KEY, agent_nm, key_nm))

    def get_individual_policy(self, aname):
        a_head = OBJECT_MAP[aname].get_pos()
        a_mode = OBJECT_MAP[aname].get_mode()
        min_dist = float("inf")
        act = ACTION_STAY
        for key in WORLD_OBJECTS:
            obj = OBJECT_MAP[key]
            if obj.get_type() in [OBJ_TYPE_1, OBJ_TYPE_2, OBJ_TYPE_3]:
                # if obj.get_num_agents() < 2:
                t_head = obj.get_pos()
                x_diff = t_head[0] - a_head[0]
                y_diff = t_head[1] - a_head[1]
                dist = abs(x_diff) + abs(y_diff)
                if dist < min_dist:
                    min_dist = dist
                    if x_diff > 0:
                        act = ACTION_RIGHT
                    elif y_diff > 0:
                        act = ACTION_DOWN
                    elif x_diff < 0:
                        act = ACTION_LEFT
                    elif y_diff < 0:
                        act = ACTION_UP
                    elif (x_diff == 0 and y_diff == 0 and
                            a_mode == MODE_IDLE):
                        act = ACTION_CLEAN
                    else:
                        act = ACTION_STAY
        return act

    def can_toggle(self, dict_states, aname):
        a_pos = dict_states[aname][0]

        def has_same_pos(trash_type):
            for t_pos in dict_states[trash_type]:
                if t_pos == a_pos:
                    return True
            return False

        if has_same_pos(OBJ_TYPE_1):
            return True
        if has_same_pos(OBJ_TYPE_2):
            return True
        if has_same_pos(OBJ_TYPE_3):
            return True

        return False

    def get_possible_action(self, dict_states, aname):
        act_list = []
        if self.can_toggle(dict_states, aname):
            if OBJECT_MAP[aname].get_mode() == MODE_IDLE:
                act_list = [ACTION_STAY, ACTION_CLEAN,
                            ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN]
            else:
                act_list = [ACTION_STAY, ACTION_CLEAN]
        else:
            act_list = [ACTION_STAY,
                        ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN]

        return act_list

    def get_possible_action_pair(self, dict_states):
        act1_list = self.get_possible_action(dict_states, AGENT_NAME_1)
        act2_list = self.get_possible_action(dict_states, AGENT_NAME_2)

        act_pair_list = []
        for act1 in act1_list:
            for act2 in act2_list:
                act_pair_list.append((act1, act2))
        return act_pair_list

    # for bad teamwork test
    def get_random_policy(self, dict_states, aname):
        act_list = self.get_possible_action(dict_states, aname)
        return random.choice(act_list)

    def get_policy_action(self, dict_states):
        action_a1 = self.get_random_policy(dict_states, AGENT_NAME_1)
        action_a2 = self.get_random_policy(dict_states, AGENT_NAME_2)
        return (action_a1, action_a2)

    def start_timer_if_necessary(self):
        if self.started == 0:
            self.started = 1

            self.timers.start()

            def move_ai(t_excute, l_name):
                act = self.get_individual_policy(AGENT_NAME_2)

                with self.event_lock:
                    dict_states = self.get_dict_states()
                    state_coord = self.conv_states_2_coord(dict_states)
                    state_coord.append(AGENT_NAME_2)
                    state_coord.append(act)
                    self.log.append(state_coord)
                    self.event_queue.put((ETYPE_KEY, AGENT_NAME_2, act))

                # add new timer
                t_n = t_excute + self.timers.get_unit_time() * 2
                te = CTimingEvent(t_n, l_name,
                                    callback=(lambda t_ex, l_name, changed:
                                            move_ai(t_ex, l_name)))
                self.timers.add_timer(te)

            # add timer
            ai_event = CTimingEvent(name="ai_move")
            ai_event.set_time = self.timers.get_unit_time() * 2
            ai_event.callback = (lambda t_ex, l_name, changed:
                                        move_ai(t_ex, l_name))
            self.timers.add_timer(ai_event)

            # add grow timers
            for obj_name in WORLD_OBJECTS:
                obj = OBJECT_MAP[obj_name]
                if obj.get_type() in [OBJ_TYPE_1, OBJ_TYPE_2, OBJ_TYPE_3]:
                    if obj.get_time2grow() > 0:
                        growing_event = CTimingEvent(name=obj_name, tag="grow")
                        growing_event.set_time = (
                                        obj.get_time2grow() -
                                        obj.get_time_progressed())
                        growing_event.callback = (
                                        lambda t_ex, l_name, changed:
                                        grow_process(self.timers, t_ex,
                                                     l_name, changed))
                        self.timers.add_timer(growing_event)

    def end_condition(self):
        if self.started and self.timers.get_current_time() >= self.max_step:
            return True

        if len(OBJECT_MAP) == 2:
            return True

        return False

    def worker_thread(self):
        self.running = 1
        self.on_data_changed(WORLD_OBJECTS)
        while self.running:
            time.sleep(0.01)
            # end of episode
            if self.end_condition():
                self.on_episode_end()
                if self.do_restart:
                    self.restart()
                else:
                    self.running = 0
                    break

            data_names = set()
            # periodic update
            while self.timers.ringing():
                self.timers.do_event(data_names)

            # event based update
            # 1 event per loop
            cur_event = None
            with self.event_lock:
                if self.event_queue.qsize():
                    try:
                        cur_event = self.event_queue.get(0)
                    except queue.Empty:
                        pass

            if cur_event is not None:
                self.start_timer_if_necessary()

                # move action
                if cur_event[0] == ETYPE_KEY:
                    agent_nm = cur_event[1]
                    action_nm = cur_event[2]
                    self.do_action(agent_nm, action_nm, data_names)

            if len(data_names) != 0:
                self.on_data_changed(data_names)
        
        if self.cb_game_end:
            self.cb_game_end()

    def end_application(self):
        if self.running != 0:
            self.running = 0
            if self.thread1:
                self.thread1.join()


    def do_action(self, agent_nm, action_nm, set_changed_data=None):
        move_actions = [ACTION_LEFT, ACTION_RIGHT,
                        ACTION_UP, ACTION_DOWN]
        if action_nm in move_actions:
            if move(agent_nm, action_nm):
                if set_changed_data is not None:
                    set_changed_data.add(agent_nm)
        elif action_nm == ACTION_PICKUP:
            data_changed = []
            if pickup(agent_nm, self.timers, data_changed):
                for obj_nm in data_changed:
                    if set_changed_data is not None:
                        set_changed_data.add(obj_nm)
        elif action_nm == ACTION_CLEAN:
            data_changed = []
            if toggle_mode(agent_nm, self.timers, data_changed):
                for obj_nm in data_changed:
                    if set_changed_data is not None:
                        set_changed_data.add(obj_nm)
        elif action_nm == ACTION_DROP:
            data_changed = []
            if drop(agent_nm, self.timers, data_changed):
                if set_changed_data is not None:
                    set_changed_data.add(agent_nm)
                for obj_pu in data_changed:
                    if set_changed_data is not None:
                        set_changed_data.add(obj_pu)

    def run_without_gui(self, episode_iteration, timeout_steps):
        if not self.has_policy:
            return

        self.timers = CDiscreteTimer()
        for cnt in range(episode_iteration):
            self.restart()
            self.start_timer_if_necessary()
            while not self.end_condition():
                while self.timers.ringing():
                    self.timers.do_event()

                dict_states = self.get_dict_states()
                action_pair = self.get_policy_action(dict_states)

                state_coord = self.conv_states_2_coord(dict_states)
                state_coord.append(action_pair[0])
                state_coord.append(action_pair[1])
                self.log.append(state_coord)

                self.do_action(AGENT_NAME_1, action_pair[0])
                self.do_action(AGENT_NAME_2, action_pair[1])
                self.timers.tick()

            self.on_episode_end()
